#pragma once

// This file provides declarations for OpenGL functions used by ImGui

#include <Windows.h>
#include <gl/GL.h>

// Use custom loader for advanced OpenGL functions
#define IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#include "gl_loader.h"

// ImGui expects these to be in global namespace
// We don't declare these since they conflict with our custom loader
// Instead, the functions are implemented in imgui_gl_bindings.cpp 