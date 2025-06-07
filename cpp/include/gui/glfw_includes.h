#pragma once

// This file helps manage GLFW includes and prevent conflicts with Windows.h

// Prevent min/max macros from Windows.h
#define NOMINMAX

// Include Windows.h with appropriate defines first
#include <Windows.h>

// Define the OpenGL loader for ImGui - This is already defined in CMakeLists.txt
// #define IMGUI_IMPL_OPENGL_LOADER_CUSTOM

// Ensure we use the Windows API for windowing
#define GLFW_EXPOSE_NATIVE_WIN32 1
#define GLFW_EXPOSE_NATIVE_WGL 1

// Now include GLFW3
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

// Make sure OpenGL constants are defined
#ifndef GL_TRIANGLES
#define GL_TRIANGLES                      0x0004
#endif

#ifndef GL_TRIANGLE_STRIP
#define GL_TRIANGLE_STRIP                 0x0005
#endif

#ifndef GL_FLOAT
#define GL_FLOAT                          0x1406
#endif

#ifndef GL_UNSIGNED_BYTE
#define GL_UNSIGNED_BYTE                  0x1401
#endif 