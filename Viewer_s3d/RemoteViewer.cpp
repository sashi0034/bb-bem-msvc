#include "stdafx.h"
#include "RemoteViewer.h"

#include "../bb-bem-msvc/src/bb_bem.h"
#include "../bb-bem-msvc/src/stl_wrapper.hpp"

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

	struct TriangleData {
		Mesh mesh;
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

struct RemoteViewer : IAddon {
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

	Array<TriangleData> m_triangleList{};

	bb_compute_t m_currentCompute{};
	int m_currentBatch{};

	bool init() override {
		rebuildModel();

		return true;
	}

	~RemoteViewer() override {
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

		// if (SimpleGUI::Button(U"{}"_fmt(getComputeName()), Vec2{20, 20})) {
		// 	m_currentCompute = static_cast<bb_compute_t>((m_currentCompute + 1) % (BB_COMPUTE_CUDA_WMMA + 1));
		// 	rebuildSphereList();
		// }
		//
		// if (SimpleGUI::Button(U"Batch {}"_fmt(m_currentBatch), Vec2{20, 60})) {
		// 	m_currentBatch = (m_currentBatch + 1) % m_bb_naive.input.para_batch;
		// 	rebuildSphereList();
		// }

		if (SimpleGUI::Button(U"Re-compute", Scene::Size().withY(20).movedBy(-200, 0))) {
			rebuildModel();
		}

		return true;
	}

private:
	void rebuildModel() {
		const std::string filename = "../../input_data/cube-ascii-1-8.stl";

		STLModel model{filename};
		for (const auto& facet : model.facets()) {
			Array<Vertex3D> vertices{};
			for (int j = 0; j < 3; ++j) {
				const stl_vector3_t& v = facet.v[j];
				const Vec3 p{v.x, v.y, v.z};

				Vertex3D vertex{};
				vertex.pos = p * 10;
				vertices.push_back(vertex);
			}

			const Array indices = {TriangleIndex32{0, 1, 2}};

			HSV color = Palette::Orangered.removeSRGBCurve();
			// sol > 0 ? Palette::Orangered.removeSRGBCurve() : Palette::Royalblue.removeSRGBCurve();

			m_triangleList.push_back({
				.mesh = Mesh{MeshData{vertices, indices}},
				.color = color
			});
		}
	}
};

void Viewer_s3d::RegisterRemoteViewer() {
	Addon::Register<RemoteViewer>(U"Viewer_3sd");
}
