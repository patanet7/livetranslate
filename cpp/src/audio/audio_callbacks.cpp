#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

// Windows headers first
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <shlobj.h>
#include <objbase.h>
#include <comdef.h>

#include <functional>
#include "../include/audio/audio_capture.hpp"

// Define the global callback variables
std::function<void(const float*, size_t)> g_audioLevelCallback;
std::function<void(const float*, const float*, size_t)> g_spectrumCallback; 