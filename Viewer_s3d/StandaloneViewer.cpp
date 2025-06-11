#include "stdafx.h"
#include "StandaloneViewer.h"

#include "bb_bem_wrapper.h"
#include "TomlConfigValueWrapper.h"
#include "../bb-bem-msvc/src/bb_bem.h"

namespace
{
	RenderTexture getPlaneTexture() {
		RenderTexture texture{1024, 1024, TextureFormat::R8G8B8A8_Unorm_SRGB, HasDepth::No, HasMipMap::Yes};
		const ScopedRenderTarget2D rt{texture.clear(ColorF{0.9}.removeSRGBCurve())};
		constexpr int step_64 = 64;
		for (int x = 0; x < texture.width(); x += step_64) {
			for (int y = 0; y < texture.height(); y += step_64) {
				RectF{Vec2{x, y}, Vec2{step_64, step_64}}.drawFrame(1.0, ColorF{0.7}.removeSRGBCurve());
			}
		}

		return texture;
	}

	struct SphereData {
		Vec3 center;
		double radius;
		ColorF color;
	};

	struct TriangleData {
		Mesh mesh;
		ColorF color;
	};
}

struct StandaloneViewer : IAddon {
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

	bb_bem_wrapper::ResultHolder m_bb{};

	// Array<SphereData> m_sphereList{};
	Array<TriangleData> m_triangleList{};

	bb_compute_t m_currentCompute{};
	int m_currentBatch{};

	bool init() override {
		calculate_bem();

		return true;
	}

	~StandaloneViewer() override {
		release_bem();
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

			// for (const auto& sphereData : m_sphereList) {
			// 	Sphere{sphereData.center, sphereData.radius}.draw(sphereData.color);
			// }

			const ScopedRenderStates3D rs3d{(RasterizerState::SolidCullNone)};
			for (const auto& t : m_triangleList) {
				t.mesh.draw(t.color);
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

		if (SimpleGUI::Button(U"{}"_fmt(bb_bem_wrapper::GetName(m_currentCompute)), Vec2{20, 20})) {
			m_currentCompute = bb_bem_wrapper::NextIndex(m_currentCompute);
			rebuildTriangleList();
		}

		if (SimpleGUI::Button(U"Batch {}"_fmt(m_currentBatch), Vec2{20, 60})) {
			m_currentBatch = (m_currentBatch + 1) % Max(1, m_bb.para_batch_unaligned());
			rebuildTriangleList();
		}

		if (SimpleGUI::Button(U"Re-compute", Scene::Size().withY(20).movedBy(-200, 0))) {
			calculate_bem();
		}

		return true;
	}

private:
	void calculate_bem() {
		release_bem();

		if (not calculate_bem_internal()) {
			std::cerr << "Error: Boundary element analysis failed." << std::endl;
			return;
		}

		m_bb.varifyResult();
		rebuildTriangleList();
	}

	bool calculate_bem_internal() {
		const std::string filename = Util::GetTomlConfigValueOf<String>(U"input_path").toUTF8();;

		if (not Util::GetTomlConfigValueOf<bool>(U"skip_naive")) {
			if (bb_bem(filename.data(), BB_COMPUTE_NAIVE, &m_bb.bb_naive) != BB_OK) return false;
		}

		if (bb_bem(filename.data(), BB_COMPUTE_CUDA, &m_bb.bb_cuda) != BB_OK) return false;
		if (bb_bem(filename.data(), BB_COMPUTE_CUDA_WMMA, &m_bb.bb_cuda_wmma) != BB_OK) return false;
		return true;
	}

	void release_bem() {
		m_bb.release();
	}

	void rebuildTriangleList() {
		// m_sphereList.clear();
		m_triangleList.clear();
		const auto& bb_result = m_bb.get(m_currentCompute);
		const auto& bb_input = bb_result.input;

		double maxSolAbs{};
		for (int fc_id = 0; fc_id < bb_input.nofc_unaligned; ++fc_id) {
			const double sol = bb_result.sol[fc_id][m_currentBatch];
			maxSolAbs = Math::Max(maxSolAbs, Math::Abs(sol));

			// std::cout << fc_id << ": " << sol << std::endl;
		}

		if (maxSolAbs == 0.0) maxSolAbs = 1.0;

		for (int fc_id = 0; fc_id < bb_input.nofc_unaligned; ++fc_id) {
			// 各要素の重心を計算
			// Vec3 centroid{0.0, 0.0, 0.0};
			// for (int nd_id = 0; nd_id < bb_result.input.nond_on_face; nd_id++) {
			// 	centroid += Vec3{
			// 		bb_input.np[bb_input.face2node[fc_id][nd_id]].x,
			// 		bb_input.np[bb_input.face2node[fc_id][nd_id]].y,
			// 		bb_input.np[bb_input.face2node[fc_id][nd_id]].z
			// 	};
			// }
			//
			// centroid /= bb_result.input.nond_on_face;

			const double sol = bb_result.sol[fc_id][m_currentBatch];

			HSV color = sol > 0 ? Palette::Orangered.removeSRGBCurve() : Palette::Royalblue.removeSRGBCurve();
			// color.s = Math::Lerp(0.3, 1.0, Math::Abs(sol) / maxSolAbs);
			color.s = Math::Abs(sol) / maxSolAbs;
			color.v = 1.0;

			// m_sphereList.push_back({
			// 	.center = centroid * 10,
			// 	.radius = 0.25,
			// 	.color = color
			// });

			Array<Vec3> verticePositions{};
			Array<Vertex3D> vertices{};
			for (int nd_id = 0; nd_id < bb_result.input.nond_on_face; nd_id++) {
				Vec3 p{};
				p.x = bb_input.np[bb_input.face2node[fc_id][nd_id]].x;
				p.y = bb_input.np[bb_input.face2node[fc_id][nd_id]].y;
				p.z = bb_input.np[bb_input.face2node[fc_id][nd_id]].z;
				verticePositions.push_back(p);

				Vertex3D v{};
				v.pos = p * 10;
				vertices.push_back(v);
			}

			// const bool isClockwise =
			// 	Vec3{verticePositions[1] - verticePositions[0]}.cross(verticePositions[2] - verticePositions[0]).y > 0;
			//
			// const Array indices = {
			// 	isClockwise ? TriangleIndex32{0, 1, 2} : TriangleIndex32{0, 2, 1}, // FIXME
			// };

			const Array indices = {TriangleIndex32{0, 2, 1}}; // FIXME

			m_triangleList.push_back({
				.mesh = Mesh{MeshData{vertices, indices},},
				.color = color
			});
		}
	}
};

void Viewer_s3d::RegisterStandaloneViewer() {
	Addon::Register<StandaloneViewer>(U"Viewer_3sd");
}
