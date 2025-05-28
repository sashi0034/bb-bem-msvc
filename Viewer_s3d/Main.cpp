# include <Siv3D.hpp>

#include "LivePPAddon.h"
#include "RemoteViewer.h"
#include "StandaloneViewer.h"

void Main() {
	Console.open();

	Window::SetTitle(U"Viewer_3sd");
	Window::Resize(1280, 720);

#ifdef _DEBUG
	Util::InitLivePPAddon();
#endif

	// Viewer_s3d::RegisterStandaloneViewer();
	Viewer_s3d::RegisterRemoteViewer();

	while (System::Update()) {
	}
}
