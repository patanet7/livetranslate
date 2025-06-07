#pragma once

// This file provides a compatibility layer between ImGui's OpenGL3 backend
// and our custom OpenGL loader (gl_loader.h)

#include "glfw_includes.h"

// Make the OpenGL functions available to ImGui impl
// ImGui expects these to be available globally when using IMGUI_IMPL_OPENGL_LOADER_CUSTOM

// OpenGL function pointers used by ImGui impl
#define IMGUI_IMPL_OPENGL_LOADER_CUSTOM 1

// These functions are used by ImGui OpenGL implementation
// We don't redefine them since we're providing the actual implementation through gl_loader.h
#ifndef IMGUI_IMPL_OPENGL_ES2
    #include <GL/gl.h>
#endif

// Ensure all function calls will go through our loader
#define glBindTexture              ::glBindTexture
#define glDeleteTextures           ::glDeleteTextures
#define glGenTextures              ::glGenTextures
#define glTexImage2D               ::glTexImage2D
#define glTexParameteri            ::glTexParameteri
#define glPixelStorei              ::glPixelStorei
#define glGetIntegerv              ::glGetIntegerv
#define glGetString                ::glGetString
#define glViewport                 ::glViewport
#define glEnable                   ::glEnable
#define glDisable                  ::glDisable
#define glClear                    ::glClear
#define glClearColor               ::glClearColor
#define glScissor                  ::glScissor
#define glBlendEquation            ::glBlendEquation
#define glBlendFunc                ::glBlendFunc
#define glBlendFuncSeparate        ::glBlendFuncSeparate
#define glBlendEquationSeparate    ::glBlendEquationSeparate
#define glUseProgram               ::glUseProgram
#define glUniform1i                ::glUniform1i
#define glUniformMatrix4fv         ::glUniformMatrix4fv
#define glGenBuffers               ::glGenBuffers
#define glBindBuffer               ::glBindBuffer
#define glGenVertexArrays          ::glGenVertexArrays
#define glBindVertexArray          ::glBindVertexArray
#define glEnableVertexAttribArray  ::glEnableVertexAttribArray
#define glVertexAttribPointer      ::glVertexAttribPointer
#define glBindSampler              ::glBindSampler 
#define glActiveTexture            ::glActiveTexture
#define glBufferData               ::glBufferData
#define glDrawElements             ::glDrawElements
#define glPolygonMode              ::glPolygonMode
#define glDeleteBuffers            ::glDeleteBuffers
#define glDeleteVertexArrays       ::glDeleteVertexArrays 
#define glIsEnabled                ::glIsEnabled
#define glIsProgram                ::glIsProgram

// Define any additional ImGui keys that might be missing
#ifndef ImGuiKey_Oem102
#define ImGuiKey_Oem102 ImGuiKey_GraveAccent
#endif

#ifndef ImGuiKey_F13
#define ImGuiKey_F13 ImGuiKey_F12
#define ImGuiKey_F14 ImGuiKey_F12
#define ImGuiKey_F15 ImGuiKey_F12
#define ImGuiKey_F16 ImGuiKey_F12
#define ImGuiKey_F17 ImGuiKey_F12
#define ImGuiKey_F18 ImGuiKey_F12
#define ImGuiKey_F19 ImGuiKey_F12
#define ImGuiKey_F20 ImGuiKey_F12
#define ImGuiKey_F21 ImGuiKey_F12
#define ImGuiKey_F22 ImGuiKey_F12
#define ImGuiKey_F23 ImGuiKey_F12
#define ImGuiKey_F24 ImGuiKey_F12
#endif

// Disable viewport platform implementation for older ImGui versions
#ifndef ImGuiPlatformIO
struct ImGuiPlatformIO {};
#define GetPlatformIO() (*((ImGuiPlatformIO*)nullptr))
#endif

#ifndef ImGuiViewport_PlatformHandle
#define ImGuiViewport_PlatformHandle(viewport) (NULL)
#endif

// Make M_PI available if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif 