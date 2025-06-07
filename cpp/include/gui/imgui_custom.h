#pragma once

// This file provides additional definitions for ImGui
// to help with compilation without conflicts

// Include standard ImGui header
#include "../external/imgui/imgui.h"

// Define any missing ImGui key enums that might be needed
#ifndef ImGuiMod_Ctrl
#define ImGuiMod_Ctrl (1 << 12)
#endif

#ifndef ImGuiMod_Shift
#define ImGuiMod_Shift (1 << 13)
#endif

#ifndef ImGuiMod_Alt
#define ImGuiMod_Alt (1 << 14)
#endif

#ifndef ImGuiMod_Super
#define ImGuiMod_Super (1 << 15)
#endif

// Define any additional ImGui flags if needed
#ifndef ImGuiBackendFlags_RendererHasVtxOffset
#define ImGuiBackendFlags_RendererHasVtxOffset (1 << 4)
#endif

// Define Oem102 key if it doesn't exist
#ifndef ImGuiKey_Oem102
enum {
    ImGuiKey_Oem102 = ImGuiKey_COUNT + 1
};
#endif 