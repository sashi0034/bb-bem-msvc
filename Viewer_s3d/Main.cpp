# include <Siv3D.hpp>

#include "LivePPAddon.h"
#include "../bb-bem-msvc/src/bb_bem.h"

namespace
{
	RenderTexture getPlaneTexture() {
		RenderTexture texture{512, 512, TextureFormat::R8G8B8A8_Unorm_SRGB, HasDepth::No, HasMipMap::Yes};
		const ScopedRenderTarget2D rt{texture.clear(ColorF{0.9}.removeSRGBCurve())};
		constexpr int step_32 = 32;
		for (int x = 0; x < texture.width(); x += step_32) {
			for (int y = 0; y < texture.height(); y += step_32) {
				RectF{Vec2{x, y}, Vec2{step_32, step_32}}.drawFrame(1.0, ColorF{0.7}.removeSRGBCurve());
			}
		}

		return texture;
	}

	struct SphereData {
		Vec3 center;
		double radius;
		ColorF color;
	};

	double compute_relative_error(const bb_result_t& a, const bb_result_t& b) {
		if (a.dim != b.dim) {
			return 0.0;
		}

		double a_b_2{};
		double a_2{};
		for (int i = 0; i < a.dim; ++i) {
			for (int n = 0; n < a.input.para_batch; ++n) {
				a_b_2 += (a.sol[i][n] - b.sol[i][n]) * (a.sol[i][n] - b.sol[i][n]);
				a_2 += (a.sol[i][n]) * (a.sol[i][n]);
			}
		}

		return Math::Sqrt(a_b_2 / a_2);
	}
}

struct Viewer_3sd : IAddon {
	// 背景色 (リニアレンダリング用なので removeSRGBCurve() で sRGB カーブを除去）
	ColorF m_backgroundColor = ColorF{0.5}.removeSRGBCurve();

	RenderTexture m_planeTexture{getPlaneTexture()};

	// 3D シーンを描く、マルチサンプリング対応レンダーテクスチャ
	// リニア色空間のレンダリング用に TextureFormat::R8G8B8A8_Unorm_SRGB
	// 奥行きの比較のための深度バッファも使うので HasDepth::Yes
	// マルチサンプル・レンダーテクスチャなので、描画内容を使う前に resolve() が必要
	MSRenderTexture m_renderTexture{Scene::Size(), TextureFormat::R8G8B8A8_Unorm_SRGB, HasDepth::Yes};

	// 3D シーンのデバッグ用カメラ
	// 縦方向の視野角 30°, カメラの位置 (10, 16, -32)
	// 前後移動: [W][S], 左右移動: [A][D], 上下移動: [E][X], 注視点移動: アローキー, 加速: [Shift][Ctrl]
	DebugCamera3D m_camera{m_renderTexture.size(), 30_deg, Vec3{10, 16, -32}};

	bb_result_t m_bb_naive{};
	bb_result_t m_bb_cuda{};
	bb_result_t m_bb_cuda_wmma{};

	Array<SphereData> m_sphereList{};

	bb_compute_t m_currentCompute{};
	int m_currentBatch{};

	bool init() override {
		Window::SetTitle(U"Viewer_3sd");

		// ウインドウとシーンを 1280x720 にリサイズ
		Window::Resize(1280, 720);

		const char* filename = "../../bb-bem-msvc/input.txt";
		if (bb_bem(filename, BB_COMPUTE_NAIVE, &m_bb_naive) == BB_OK &&
			bb_bem(filename, BB_COMPUTE_CUDA, &m_bb_cuda) == BB_OK &&
			bb_bem(filename, BB_COMPUTE_CUDA_WMMA, &m_bb_cuda_wmma) == BB_OK
		) {
			varifyResult();
			rebuildSphereList();
		}
		else {7
			std::cerr << "Error: Boundary element analysis failed: input.txt" << std::endl;
		}

		return true;
	}

	~Viewer_3sd() override {
		release_bb_result(&m_bb_naive);
		release_bb_result(&m_bb_cuda);
	}

	bool update() override {
		// デバッグカメラの更新 (カメラの移動スピード: 2.0)
		m_camera.update(2.0);

		// 3D シーンにカメラを設定
		Graphics3D::SetCameraTransform(m_camera);

		// 3D 描画
		{
			// renderTexture を背景色で塗りつぶし、
			// renderTexture を 3D 描画のレンダーターゲットに
			const ScopedRenderTarget3D target{m_renderTexture.clear(m_backgroundColor)};

			Plane{Vec3{}.withY(-10.0), 64}.draw(m_planeTexture);

			for (const auto& sphereData : m_sphereList) {
				Sphere{sphereData.center, sphereData.radius}.draw(sphereData.color);
			}
		}

		// 3D シーンを 2D シーンに描画
		{
			// renderTexture を resolve する前に 3D 描画を実行する
			Graphics3D::Flush();

			// マルチサンプル・テクスチャのリゾルブ
			m_renderTexture.resolve();

			// リニアレンダリングされた renderTexture をシーンに転送
			Shader::LinearToScreen(m_renderTexture);
		}

		if (SimpleGUI::Button(U"{}"_fmt(getComputeName()), Vec2{20, 20})) {
			m_currentCompute = static_cast<bb_compute_t>((m_currentCompute + 1) % (BB_COMPUTE_CUDA_WMMA + 1));
			rebuildSphereList();
		}

		if (SimpleGUI::Button(U"Batch {}"_fmt(m_currentBatch), Vec2{20, 60})) {
			m_currentBatch = (m_currentBatch + 1) % m_bb_naive.input.para_batch;
			rebuildSphereList();
		}

		return true;
	}

private:
	const bb_result_t& get_bb_result() const {
		switch (m_currentCompute) {
		case BB_COMPUTE_NAIVE:
			return m_bb_naive;
		case BB_COMPUTE_CUDA:
			return m_bb_cuda;
		case BB_COMPUTE_CUDA_WMMA:
			return m_bb_cuda_wmma;
		default:
			assert(false);
			return {};
		}
	}

	String getComputeName() const {
		switch (m_currentCompute) {
		case BB_COMPUTE_NAIVE:
			return U"Naive";
		case BB_COMPUTE_CUDA:
			return U"CUDA";
		case BB_COMPUTE_CUDA_WMMA:
			return U"CUDA WMMA";
		default:
			assert(false);
			return {};
		}
	}

	void rebuildSphereList() {
		m_sphereList.clear();
		const auto& bb_result = get_bb_result();
		const auto& bb_input = bb_result.input;

		double maxSolAbs{};
		for (int fc_id = 0; fc_id < bb_input.nofc; ++fc_id) {
			const double sol = bb_result.sol[fc_id][m_currentBatch];
			maxSolAbs = Math::Max(maxSolAbs, Math::Abs(sol));
		}

		if (maxSolAbs == 0.0) maxSolAbs = 1.0;

		for (int fc_id = 0; fc_id < bb_input.nofc; ++fc_id) {
			// 各要素の重心を計算
			Vec3 centroid{0.0, 0.0, 0.0};
			for (int nd_id = 0; nd_id < bb_result.input.nond_on_face; nd_id++) {
				centroid += Vec3{
					bb_input.np[bb_input.face2node[fc_id][nd_id]].x,
					bb_input.np[bb_input.face2node[fc_id][nd_id]].y,
					bb_input.np[bb_input.face2node[fc_id][nd_id]].z
				};
			}

			centroid /= bb_result.input.nond_on_face;

			const double sol = bb_result.sol[fc_id][m_currentBatch]; // TODO: インデックスの変更対応

			m_sphereList.push_back({
				.center = centroid * 10,
				.radius = (Math::Abs(sol) / maxSolAbs) * 0.25,
				.color = sol > 0 ? Palette::Orange.removeSRGBCurve() : Palette::Lightskyblue.removeSRGBCurve()
			});
		}
	}

	void varifyResult() {
		Console.writeln(U"----------------------------------------------- Result verification");

		Console.writeln(
			U"Relative error between Naive and Cuda: {}"_fmt(compute_relative_error(m_bb_naive, m_bb_cuda)));
		Console.writeln(
			U"Relative error between Naive and Cuda-WMMA: {}"_fmt(compute_relative_error(m_bb_naive, m_bb_cuda_wmma)));

		Console.writeln(
			U"Compute time (Naive): {} sec"_fmt(m_bb_naive.compute_time));
		Console.writeln(
			U"Compute time (Cuda): {} sec"_fmt(m_bb_cuda.compute_time));
		Console.writeln(
			U"Compute time (Cuda-WMMA): {} sec"_fmt(m_bb_cuda_wmma.compute_time));
	}
};

void Main() {
	Console.open();

	Util::InitLivePPAddon();

	Addon::Register<Viewer_3sd>(U"Viewer_3sd");

	while (System::Update()) {
	}
}
