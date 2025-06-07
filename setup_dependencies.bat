@echo off
REM Setup script for LiveTranslate dependencies
REM This script will download and extract ImGui and GLFW

set IMGUI_VERSION=v1.89.9
set GLFW_VERSION=3.3.8

echo Setting up dependencies for LiveTranslate...
echo.

REM Create external directory if it doesn't exist
if not exist external mkdir external
pushd external

REM Download and extract ImGui
echo Downloading ImGui...
if not exist imgui (
    powershell -Command "Invoke-WebRequest -Uri https://github.com/ocornut/imgui/archive/refs/tags/%IMGUI_VERSION%.zip -OutFile imgui.zip"
    echo Extracting ImGui...
    powershell -Command "Expand-Archive -Path imgui.zip -DestinationPath ."
    mkdir imgui
    xcopy /E /I imgui-%IMGUI_VERSION:~1%\* imgui\
    rmdir /s /q imgui-%IMGUI_VERSION:~1%
    del imgui.zip
    
    REM Create backends directory if missing
    if not exist imgui\backends mkdir imgui\backends
    
    REM Download backend implementation files
    echo Downloading ImGui backend files...
    powershell -Command "Invoke-WebRequest -Uri https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_glfw.cpp -OutFile imgui\backends\imgui_impl_glfw.cpp"
    powershell -Command "Invoke-WebRequest -Uri https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_glfw.h -OutFile imgui\backends\imgui_impl_glfw.h"
    powershell -Command "Invoke-WebRequest -Uri https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3.cpp -OutFile imgui\backends\imgui_impl_opengl3.cpp"
    powershell -Command "Invoke-WebRequest -Uri https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3.h -OutFile imgui\backends\imgui_impl_opengl3.h"
    powershell -Command "Invoke-WebRequest -Uri https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3_loader.h -OutFile imgui\backends\imgui_impl_opengl3_loader.h"
) else (
    echo ImGui directory already exists, checking backends...
    if not exist imgui\backends (
        echo ImGui backends directory is missing, downloading backend files...
        mkdir imgui\backends
        powershell -Command "Invoke-WebRequest -Uri https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_glfw.cpp -OutFile imgui\backends\imgui_impl_glfw.cpp"
        powershell -Command "Invoke-WebRequest -Uri https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_glfw.h -OutFile imgui\backends\imgui_impl_glfw.h"
        powershell -Command "Invoke-WebRequest -Uri https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3.cpp -OutFile imgui\backends\imgui_impl_opengl3.cpp"
        powershell -Command "Invoke-WebRequest -Uri https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3.h -OutFile imgui\backends\imgui_impl_opengl3.h"
        powershell -Command "Invoke-WebRequest -Uri https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3_loader.h -OutFile imgui\backends\imgui_impl_opengl3_loader.h"
    ) else (
        echo Checking for OpenGL3 loader...
        if not exist imgui\backends\imgui_impl_opengl3_loader.h (
            echo Downloading OpenGL3 loader...
            powershell -Command "Invoke-WebRequest -Uri https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3_loader.h -OutFile imgui\backends\imgui_impl_opengl3_loader.h"
        )
    )
)

REM Download and extract GLFW
echo Downloading GLFW...
if not exist glfw (
    mkdir glfw
    powershell -Command "Invoke-WebRequest -Uri https://github.com/glfw/glfw/releases/download/%GLFW_VERSION%/glfw-%GLFW_VERSION%.bin.WIN64.zip -OutFile glfw.zip"
    echo Extracting GLFW...
    powershell -Command "Expand-Archive -Path glfw.zip -DestinationPath ."
    
    REM Check structure of extracted files
    if exist glfw-%GLFW_VERSION%.bin.WIN64\include\GLFW (
        echo GLFW structure looks good, copying files...
        mkdir glfw\include\GLFW
        xcopy /E /I glfw-%GLFW_VERSION%.bin.WIN64\include\GLFW\* glfw\include\GLFW\
    ) else (
        echo GLFW structure is different, adapting...
    )
    
    REM Make sure to copy all needed files including libs
    if exist glfw-%GLFW_VERSION%.bin.WIN64\lib-vc2022 (
        mkdir glfw\lib-vc2022
        xcopy /E /I glfw-%GLFW_VERSION%.bin.WIN64\lib-vc2022\* glfw\lib-vc2022\
    )
    
    rmdir /s /q glfw-%GLFW_VERSION%.bin.WIN64
    del glfw.zip
) else (
    echo GLFW directory already exists, checking structure...
    if not exist glfw\include\GLFW (
        echo GLFW include directory is missing or has wrong structure.
        echo Re-downloading GLFW...
        rmdir /s /q glfw
        mkdir glfw
        powershell -Command "Invoke-WebRequest -Uri https://github.com/glfw/glfw/releases/download/%GLFW_VERSION%/glfw-%GLFW_VERSION%.bin.WIN64.zip -OutFile glfw.zip"
        echo Extracting GLFW...
        powershell -Command "Expand-Archive -Path glfw.zip -DestinationPath ."
        
        REM Create proper directory structure
        mkdir glfw\include
        if exist glfw-%GLFW_VERSION%.bin.WIN64\include\GLFW (
            mkdir glfw\include\GLFW
            xcopy /E /I glfw-%GLFW_VERSION%.bin.WIN64\include\GLFW\* glfw\include\GLFW\
        )
        
        REM Copy library files
        if exist glfw-%GLFW_VERSION%.bin.WIN64\lib-vc2022 (
            mkdir glfw\lib-vc2022
            xcopy /E /I glfw-%GLFW_VERSION%.bin.WIN64\lib-vc2022\* glfw\lib-vc2022\
        )
        
        rmdir /s /q glfw-%GLFW_VERSION%.bin.WIN64
        del glfw.zip
    )
)

popd

echo.
echo Dependencies setup complete!
echo.
echo Please make sure the following paths are correct in CMakeLists.txt:
echo - IMGUI_PATH = "${CMAKE_SOURCE_DIR}/external/imgui"
echo - GLFW_PATH = "${CMAKE_SOURCE_DIR}/external/glfw"
echo.
echo Note: You need to install the NVIDIA Audio Effects SDK and update the path in CMakeLists.txt:
echo - NVIDIA_SDK_PATH = "C:/Program Files/NVIDIA Corporation/NVIDIA Audio Effects SDK"
echo.
echo Now you can build the project with CMake. 