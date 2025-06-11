#include "pch.h"

#include "StandaloneViewer_TY.h"
#include "TY/Window.h"

void Main() {
    TY::Window::SetTitle("Viewer_TY");

    Viewer_TY::StandaloneViewer();
}
