cmake_minimum_required(VERSION 3.15)
project(livetranslate_capture)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add special compiler options to fix complex initialization
if(MSVC)
    # Force standard-compliant constructor initialization
    add_compile_options(/permissive-)
    # Define _USE_MATH_DEFINES for M_PI
    add_definitions(-D_USE_MATH_DEFINES)
    # Disable deprecation warnings for standard library functions
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Created math_helpers.h for consistent math functions across Windows

# Set Boost options
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(BOOST_ROOT "C:/boost_1_88_0") # Update this path to your Boost installation
set(BOOST_INCLUDEDIR "${BOOST_ROOT}")
set(BOOST_LIBRARYDIR "${BOOST_ROOT}/stage/lib")

# Find required packages
find_package(Boost REQUIRED COMPONENTS system date_time)

# Add nlohmann/json
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Core audio capture source files
set(CORE_SOURCES
    cpp/src/audio/audio_capture.cpp
    cpp/src/audio/nvidia_effects.cpp
    cpp/src/audio/audio_callbacks.cpp
)

# GUI source files
set(GUI_SOURCES
    cpp/src/gui/gui.cpp
    cpp/src/gui/gl_loader.cpp
)

# Updated NVIDIA effects SDK path
set(NVIDIA_EFFECTS_SDK_PATH "${CMAKE_SOURCE_DIR}/external/NVIDIA Audio Effects SDK")
message(STATUS "NVIDIA SDK Path: ${NVIDIA_EFFECTS_SDK_PATH}")
include_directories("${NVIDIA_EFFECTS_SDK_PATH}/include")

# Updated ImGui path
set(IMGUI_PATH "${CMAKE_SOURCE_DIR}/external/imgui")
message(STATUS "ImGui Path: ${IMGUI_PATH}")
include_directories("${IMGUI_PATH}")
include_directories("${IMGUI_PATH}/backends")

# Updated GLFW path
set(GLFW_PATH "${CMAKE_SOURCE_DIR}/external/glfw")
message(STATUS "GLFW Path: ${GLFW_PATH}")
include_directories("${GLFW_PATH}/include")

# Glad (OpenGL loader)
set(GLAD_SOURCES "${CMAKE_SOURCE_DIR}/external/glad/src/glad.c")
include_directories("${CMAKE_SOURCE_DIR}/external/glad/include/")

# ImGui source files
set(IMGUI_SOURCES
    "${IMGUI_PATH}/imgui.cpp"
    "${IMGUI_PATH}/imgui_demo.cpp"
    "${IMGUI_PATH}/imgui_draw.cpp"
    "${IMGUI_PATH}/imgui_tables.cpp"
    "${IMGUI_PATH}/imgui_widgets.cpp"
    "${IMGUI_PATH}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_PATH}/backends/imgui_impl_opengl3.cpp"
)

# Add muFFT source files
set(MUFFT_SOURCES
    ${CMAKE_SOURCE_DIR}/external/muFFT/fft.c
    ${CMAKE_SOURCE_DIR}/external/muFFT/kernel.c
    ${CMAKE_SOURCE_DIR}/external/muFFT/cpu.c
)

add_library(mufft STATIC ${MUFFT_SOURCES})
target_include_directories(mufft PUBLIC ${CMAKE_SOURCE_DIR}/external/muFFT)
target_compile_definitions(mufft PRIVATE MUFFT_USE_AVX MUFFT_USE_SSE3 MUFFT_USE_SSE)

# Include paths
include_directories(cpp/include)
include_directories(${Boost_INCLUDE_DIRS})

# Output paths for executable and directories
message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")

# Add the main audio capture executable
add_executable(audio_capture
    cpp/src/main.cpp
    ${CORE_SOURCES}
)

target_link_libraries(audio_capture 
    ${Boost_LIBRARIES}
    mufft
    nlohmann_json::nlohmann_json
)

# Add the Boost test executable
add_executable(boost_test
    cpp/src/utils/boost_test.cpp
)

target_link_libraries(boost_test ${Boost_LIBRARIES})

# Add the GUI application
add_executable(imgui_gui
    cpp/src/gui/gui_main.cpp
    ${CORE_SOURCES}
    ${GUI_SOURCES}
    ${IMGUI_SOURCES}
    ${GLAD_SOURCES}   
)

target_link_libraries(imgui_gui 
    ${Boost_LIBRARIES}
    mufft  
    ${NVIDIA_EFFECTS_SDK_PATH}/lib/windows/nvAudioEffects.lib
    ${GLFW_PATH}/lib-vc2022/glfw3.lib
    opengl32.lib
    comdlg32.lib
    nlohmann_json::nlohmann_json
)

# Link against the NVIDIA Effects library
if(WIN32)
    target_link_libraries(audio_capture 
        ${NVIDIA_EFFECTS_SDK_PATH}/lib/windows/nvAudioEffects.lib
    )
    
    target_link_libraries(imgui_gui 
        ${NVIDIA_EFFECTS_SDK_PATH}/lib/windows/nvAudioEffects.lib
        ${GLFW_PATH}/lib-vc2022/glfw3.lib
        opengl32.lib
        comdlg32.lib
    )
    
    # Use the /bigobj flag for ImGui files which can be large
    target_compile_options(imgui_gui PRIVATE /bigobj)
    
    # Define ImGui to use standard OpenGL loader
    target_compile_definitions(imgui_gui PRIVATE 
    IMGUI_IMPL_OPENGL_LOADER_GLAD
    "_CRT_SECURE_NO_WARNINGS"
    "_USE_MATH_DEFINES"
    "UNICODE"
    "_UNICODE"
    "WINVER=0x0A00"
    "_WIN32_WINNT=0x0A00"
    )

endif()

# Updated NVIDIA DLL copy commands
if(WIN32)
    add_custom_command(TARGET audio_capture POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${NVIDIA_EFFECTS_SDK_PATH}/lib/windows/nvAudioEffects.dll"
            $<TARGET_FILE_DIR:audio_capture>
    )
    
    add_custom_command(TARGET imgui_gui POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${NVIDIA_EFFECTS_SDK_PATH}/lib/windows/nvAudioEffects.dll"
            $<TARGET_FILE_DIR:imgui_gui>
    )
endif()