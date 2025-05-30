# include <Siv3D.hpp>

#include "LivePPAddon.h"
#include "RemoteViewer.h"
#include "StandaloneViewer.h"
#include "TomlConfigValueWrapper.h"

void Main() {
	Console.open();

	Window::SetTitle(U"Viewer_3sd");
	Window::Resize(1280, 720);

#ifdef _DEBUG
	Util::InitLivePPAddon();
#endif

	Util::InitTomlConfigValueAddon();

	if (Util::GetTomlConfigValueOf<bool>(U"use_remote_viewer")) {
		Viewer_s3d::RegisterRemoteViewer();
	}
	else {
		Viewer_s3d::RegisterStandaloneViewer();
	}

	while (System::Update()) {
	}
}
