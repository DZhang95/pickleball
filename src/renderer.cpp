#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cstdlib>

// Vertex shader source code
const char* vertexShaderSource = 
"#version 330 core\n"
"layout (location = 0) in vec2 aPos;\n"
"\n"
"void main() {\n"
"    gl_Position = vec4(aPos, 0.0, 1.0);\n"
"}\n";

// Fragment shader source code
const char* fragmentShaderSource = 
"#version 330 core\n"
"out vec4 FragColor;\n"
"\n"
"void main() {\n"
"    FragColor = vec4(0.2f, 0.5f, 0.8f, 1.0f);\n"
"}\n";

// Function to compile shader
unsigned int compileShader(unsigned int type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    
    // Check for compilation errors
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
        return 0;
    }
    return shader;
}

// Function to create shader program
unsigned int createShaderProgram() {
    unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    unsigned int fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    
    if (vertexShader == 0 || fragmentShader == 0) {
        return 0;
    }
    
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    // Check for linking errors
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        return 0;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return shaderProgram;
}

// Function to render a 2D rectangular box
void renderRectangle(float centerX, float centerY, float width, float height) {
    // Define vertices for a rectangle (2 triangles)
    // Rectangle centered at (centerX, centerY) with given width and height
    float halfWidth = width / 2.0f;
    float halfHeight = height / 2.0f;
    
    float vertices[] = {
        // First triangle
        centerX - halfWidth, centerY - halfHeight,  // Bottom-left
        centerX + halfWidth, centerY - halfHeight,  // Bottom-right
        centerX - halfWidth, centerY + halfHeight,  // Top-left
        // Second triangle
        centerX - halfWidth, centerY + halfHeight,  // Top-left
        centerX + halfWidth, centerY - halfHeight,  // Bottom-right
        centerX + halfWidth, centerY + halfHeight   // Top-right
    };
    
    // Create and bind VAO (Vertex Array Object)
    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    
    // Create and bind VBO (Vertex Buffer Object)
    unsigned int VBO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    // Set vertex attribute pointers
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Draw the rectangle
    glDrawArrays(GL_TRIANGLES, 0, 6);
    
    // Clean up
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    
    // Create window
    GLFWwindow* window = glfwCreateWindow(800, 600, "2D Rectangle Renderer", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    
    // Create shader program
    unsigned int shaderProgram = createShaderProgram();
    if (shaderProgram == 0) {
        std::cerr << "Failed to create shader program" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    // Main render loop
    while (!glfwWindowShouldClose(window)) {
        // Clear the screen
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // Dark gray background
        glClear(GL_COLOR_BUFFER_BIT);
        
        // Use shader program
        glUseProgram(shaderProgram);
        
        // Render a rectangle at the center of the screen
        // Coordinates are in normalized device coordinates (-1 to 1)
        renderRectangle(0.0f, 0.0f, 1.5f, 1.3f); // Center, width=0.5, height=0.3
        
        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    // Cleanup
    glDeleteProgram(shaderProgram);
    glfwTerminate();
    
    return 0;
}

