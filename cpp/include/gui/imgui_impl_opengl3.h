#pragma once

// Minimal OpenGL3 implementation for ImGui

// If you are new to Dear ImGui, read documentation from the docs/ folder + read the top of imgui.cpp.
// Read online: https://github.com/ocornut/imgui/tree/master/docs

// About GLSL version:
// None of this is ideal. You also want to check '#define IMGUI_IMPL_OPENGL_ES2' below.
// The GL ES 3.0 shader should work on most modern GLES3+ devices with the exception of WebGL 2.0.
// iOS before version 11 may need '#define IMGUI_IMPL_OPENGL_LOADER_GLES2'.
#define IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    #define GLSL_VERSION "#version 100"
#elif defined(IMGUI_IMPL_OPENGL_ES3)
    // GL ES 3.0 + GLSL 300 es
    #define GLSL_VERSION "#version 300 es"
#else
    // GL 3.2 + GLSL 150
    #define GLSL_VERSION "#version 150"
#endif

IMGUI_IMPL_API bool     ImGui_ImplOpenGL3_Init(const char* glsl_version = GLSL_VERSION);
IMGUI_IMPL_API void     ImGui_ImplOpenGL3_Shutdown();
IMGUI_IMPL_API void     ImGui_ImplOpenGL3_NewFrame();
IMGUI_IMPL_API void     ImGui_ImplOpenGL3_RenderDrawData(ImDrawData* draw_data);

// Called by Init/NewFrame/Shutdown
IMGUI_IMPL_API bool     ImGui_ImplOpenGL3_CreateFontsTexture();
IMGUI_IMPL_API void     ImGui_ImplOpenGL3_DestroyFontsTexture();
IMGUI_IMPL_API bool     ImGui_ImplOpenGL3_CreateDeviceObjects();
IMGUI_IMPL_API void     ImGui_ImplOpenGL3_DestroyDeviceObjects(); 