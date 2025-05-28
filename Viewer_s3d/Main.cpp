# include <Siv3D.hpp>

#include "LivePPAddon.h"
#include "StandaloneViewer.h"

void Main() {
	Console.open();

#ifdef _DEBUG
	Util::InitLivePPAddon();
#endif

	Viewer_s3d::RegisterStandaloneViewer();

	while (System::Update()) {
	}
}
