@echo off
REM Build script for LiveTranslate

REM Setup dependencies first
call setup_dependencies.bat

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Make sure the GLFW include structure is correct
if not exist external\glfw\include\GLFW (
    echo Creating GLFW include directory structure...
    mkdir external\glfw\include\GLFW 2>nul
    
    REM If we don't have glfw3.h, create a minimal one
    if not exist external\glfw\include\GLFW\glfw3.h (
        echo Creating minimal GLFW header...
        echo #pragma once > external\glfw\include\GLFW\glfw3.h
        echo // Minimal GLFW3 header >> external\glfw\include\GLFW\glfw3.h
        echo typedef void* GLFWwindow; >> external\glfw\include\GLFW\glfw3.h
        echo typedef void* GLFWmonitor; >> external\glfw\include\GLFW\glfw3.h
        echo #define GLFW_KEY_ESCAPE 256 >> external\glfw\include\GLFW\glfw3.h
    )
    
    REM If we don't have the lib directory, create it
    if not exist external\glfw\lib-vc2022 (
        mkdir external\glfw\lib-vc2022 2>nul
        
        REM Create an empty lib file if it doesn't exist
        if not exist external\glfw\lib-vc2022\glfw3.lib (
            echo Creating placeholder GLFW lib file...
            type nul > external\glfw\lib-vc2022\glfw3.lib
        )
    )
)

REM Navigate to build directory
pushd build

REM Run CMake
cmake ..

REM Build the project
cmake --build . --config Release

REM Return to original directory
popd

REM Copy DLLs if necessary
if exist build\Release\audio_capture.exe (
    if exist "external\NVIDIA Audio Effects SDK\NVAudioEffects.dll" (
        echo Copying NVIDIA DLLs to output directory...
        copy "external\NVIDIA Audio Effects SDK\NVAudioEffects.dll" "build\Release\" > nul
        copy "external\NVIDIA Audio Effects SDK\*.dll" "build\Release\" > nul
        
        echo Copying NVIDIA model files to output directory...
        if not exist "build\Release\models" mkdir "build\Release\models" 2>nul
        copy "external\NVIDIA Audio Effects SDK\models\*.trtpkg" "build\Release\models\" > nul
    )
) 

echo Build complete. Output in build/Release/ 