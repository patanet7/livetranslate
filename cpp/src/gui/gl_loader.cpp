#include "../include/gui/gl_loader.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

bool loadOpenGLFunctions() {
    // Use glad to load all OpenGL function pointers
    return gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
}