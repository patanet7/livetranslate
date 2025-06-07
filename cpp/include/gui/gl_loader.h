#pragma once

#include <glad/glad.h>

// Optionally include error message helpers
#include <windows.h>
#include <string>

bool loadOpenGLFunctions(); 

inline std::string GetLastErrorAsString() {
    DWORD error = GetLastError();
    if (error == 0) return "";

    LPSTR messageBuffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);

    std::string message(messageBuffer, size);
    LocalFree(messageBuffer);
    return message;
}
