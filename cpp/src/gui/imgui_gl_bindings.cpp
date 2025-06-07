#include "../include/imgui_impl_includes.h"
#include <stdexcept>
#include <string>

// This file implements OpenGL function forwarding from ImGui to our custom gl_loader
// We don't need to redefine functions here since we're including gl_loader.h
// in the imgui_impl_includes.h, which provides the necessary function pointers.

// Most OpenGL functions are resolved by gl_loader.cpp
// This file is here just to ensure the forward declarations work
// and to make sure the linker can find all the required symbols. 