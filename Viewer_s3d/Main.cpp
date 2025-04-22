# include <Siv3D.hpp>

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
}

struct Viewer_3sd : IAddon {
	// 背景色 (リニアレンダリング用なので removeSRGBCurve() で sRGB カーブを除去）
	ColorF m_backgroundColor = ColorF{0.5}.removeSRGBCurve();

	// UV チェック用テクスチャ (ミップマップ使用。リニアレンダリング時に正しく扱われるよう、sRGB テクスチャであると明示）
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

	bool init() override {
		// ウインドウとシーンを 1280x720 にリサイズ
		Window::Resize(1280, 720);

		return true;
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

			// 床を描画
			Plane{64}.draw(m_planeTexture);

			// ボックスを描画
			Box{-8, 2, 0, 4}.draw(ColorF{0.8, 0.6, 0.4}.removeSRGBCurve());

			// 球を描画
			Sphere{0, 2, 0, 2}.draw(ColorF{0.4, 0.8, 0.6}.removeSRGBCurve());

			// 円柱を描画
			Cylinder{8, 2, 0, 2, 4}.draw(ColorF{0.6, 0.4, 0.8}.removeSRGBCurve());
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

		return true;
	}
};

void Main() {
	Addon::Register<Viewer_3sd>(U"Viewer_3sd");

	while (System::Update()) {
	}
}
