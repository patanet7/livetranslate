#pragma once

#include <cmath>
#include <algorithm>

// Define M_PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Define local math functions to avoid Windows-specific issues
namespace math {
    inline float sin(float x) { return std::sin(x); }
    inline float cos(float x) { return std::cos(x); }
    inline float tan(float x) { return std::tan(x); }
    inline float asin(float x) { return std::asin(x); }
    inline float acos(float x) { return std::acos(x); }
    inline float atan(float x) { return std::atan(x); }
    inline float atan2(float y, float x) { return std::atan2(y, x); }
    inline float sqrt(float x) { return std::sqrt(x); }
    inline float pow(float x, float y) { return std::pow(x, y); }
    inline float log(float x) { return std::log(x); }
    inline float log10(float x) { return std::log10(x); }
    inline float abs(float x) { return std::abs(x); }
    inline float floor(float x) { return std::floor(x); }
    inline float ceil(float x) { return std::ceil(x); }
    inline float round(float x) { return std::round(x); }
    
    // Min/max functions 
    template<typename T>
    inline T min(T a, T b) { return std::min(a, b); }
    
    template<typename T>
    inline T max(T a, T b) { return std::max(a, b); }
    
    template<typename T>
    inline T clamp(T value, T min_val, T max_val) { 
        return std::max(min_val, std::min(value, max_val)); 
    }
    
    // Degree/radian conversion
    inline float degrees(float radians) { return radians * (180.0f / M_PI); }
    inline float radians(float degrees) { return degrees * (M_PI / 180.0f); }
} 