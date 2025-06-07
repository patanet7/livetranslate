#pragma once

// Fix for std::complex and __uuidof issues
// This header should be included in all source files that use these features

#include <windows.h>
#include <initguid.h>
#include <cmath>
#include <string>

// Define GUID constants for audio interfaces
DEFINE_GUID(CLSID_MMDeviceEnumerator, 0xBCDE0395, 0xE52F, 0x467C, 0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E);
DEFINE_GUID(IID_IMMDeviceEnumerator, 0xA95664D2, 0x9614, 0x4F35, 0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6);
DEFINE_GUID(IID_IAudioClient, 0x1CB9AD4C, 0xDBFA, 0x4C32, 0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2);
DEFINE_GUID(IID_IAudioCaptureClient, 0xC8ADBD64, 0xE71E, 0x48A0, 0xA4, 0xDE, 0x18, 0x5C, 0x39, 0x5C, 0xD3, 0x17);
DEFINE_GUID(IID_IAudioRenderClient, 0xF294ACFC, 0x3146, 0x4483, 0xA7, 0xBF, 0xAD, 0xDC, 0xA7, 0xC2, 0x60, 0xE2);

// Replacement for std::complex
struct ComplexFloat {
    float real;
    float imag;
    
    ComplexFloat() : real(0.0f), imag(0.0f) {}
    ComplexFloat(float r, float i) : real(r), imag(i) {}
    
    // Basic operations
    ComplexFloat operator+(const ComplexFloat& other) const {
        return ComplexFloat(real + other.real, imag + other.imag);
    }
    
    ComplexFloat operator-(const ComplexFloat& other) const {
        return ComplexFloat(real - other.real, imag - other.imag);
    }
    
    ComplexFloat operator*(const ComplexFloat& other) const {
        return ComplexFloat(
            real * other.real - imag * other.imag,
            real * other.imag + imag * other.real
        );
    }
    
    // Magnitude calculation
    float abs() const {
        return std::sqrt(real * real + imag * imag);
    }
};

// String conversion utility
inline std::string wstring_to_string(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string str(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), &str[0], size_needed, NULL, NULL);
    return str;
} 