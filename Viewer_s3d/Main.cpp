# include <Siv3D.hpp>

struct Viewer_3sd : IAddon {
	// 背景色 (リニアレンダリング用なので removeSRGBCurve() で sRGB カーブを除去）
	ColorF backgroundColor = ColorF{0.4, 0.6, 0.8}.removeSRGBCurve();

	// UV チェック用テクスチャ (ミップマップ使用。リニアレンダリング時に正しく扱われるよう、sRGB テクスチャであると明示）
	Texture uvChecker{U"example/texture/uv.png", TextureDesc::MippedSRGB};

	// 3D シーンを描く、マルチサンプリング対応レンダーテクスチャ
	// リニア色空間のレンダリング用に TextureFormat::R8G8B8A8_Unorm_SRGB
	// 奥行きの比較のための深度バッファも使うので HasDepth::Yes
	// マルチサンプル・レンダーテクスチャなので、描画内容を使う前に resolve() が必要
	MSRenderTexture renderTexture{Scene::Size(), TextureFormat::R8G8B8A8_Unorm_SRGB, HasDepth::Yes};

	// 3D シーンのデバッグ用カメラ
	// 縦方向の視野角 30°, カメラの位置 (10, 16, -32)
	// 前後移動: [W][S], 左右移動: [A][D], 上下移動: [E][X], 注視点移動: アローキー, 加速: [Shift][Ctrl]
	DebugCamera3D camera{renderTexture.size(), 30_deg, Vec3{10, 16, -32}};

	bool init() override {
		// ウインドウとシーンを 1280x720 にリサイズ
		Window::Resize(1280, 720);

		return true;
	}

	bool update() override {
		// デバッグカメラの更新 (カメラの移動スピード: 2.0)
		camera.update(2.0);

		// 3D シーンにカメラを設定
		Graphics3D::SetCameraTransform(camera);

		// 3D 描画
		{
			// renderTexture を背景色で塗りつぶし、
			// renderTexture を 3D 描画のレンダーターゲットに
			const ScopedRenderTarget3D target{renderTexture.clear(backgroundColor)};

			// 床を描画
			Plane{64}.draw(uvChecker);

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
			renderTexture.resolve();

			// リニアレンダリングされた renderTexture をシーンに転送
			Shader::LinearToScreen(renderTexture);
		}

		return true;
	}
};

void Main() {
	Addon::Register<Viewer_3sd>(U"Viewer_3sd");

	while (System::Update()) {
	}
}
