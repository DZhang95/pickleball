#define GL_SILENCE_DEPRECATION
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>

// Vertex shader source code
const char* vertexShaderSource = 
"#version 330 core\n"
"layout (location = 0) in vec2 aPos;\n"
"\n"
"void main() {\n"
"    gl_Position = vec4(aPos, 0.0, 1.0);\n"
"}\n";

// Fragment shader source code for rectangle
const char* fragmentShaderSource = 
"#version 330 core\n"
"out vec4 FragColor;\n"
"\n"
"void main() {\n"
"    FragColor = vec4(0.2f, 0.5f, 0.8f, 1.0f);\n"
"}\n";

// Fragment shader source code for circle (different color)
const char* circleFragmentShaderSource = 
"#version 330 core\n"
"out vec4 FragColor;\n"
"\n"
"void main() {\n"
"    FragColor = vec4(0.8f, 0.3f, 0.2f, 1.0f);\n"
"}\n";

// Fragment shader source code for net (white)
const char* netFragmentShaderSource = 
"#version 330 core\n"
"out vec4 FragColor;\n"
"\n"
"void main() {\n"
"    FragColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);\n"
"}\n";

// Fragment shader source code for air particles (light gray)
const char* airParticleFragmentShaderSource = 
"#version 330 core\n"
"out vec4 FragColor;\n"
"\n"
"void main() {\n"
"    FragColor = vec4(0.7f, 0.7f, 0.7f, 1.0f);\n"
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

// Function to render a 2D circle
void renderCircle(float centerX, float centerY, float radius, int segments) {
    // Generate vertices for a circle using triangles
    // segments determines the number of triangles used to approximate the circle
    std::vector<float> vertices;
    
    // Generate vertices as triangles (center + two adjacent points on circle)
    for (int i = 0; i < segments; i++) {
        // Center vertex
        vertices.push_back(centerX);
        vertices.push_back(centerY);
        
        // First point on circle
        float angle1 = 2.0f * 3.14159265359f * i / segments;
        float x1 = centerX + radius * cosf(angle1);
        float y1 = centerY + radius * sinf(angle1);
        vertices.push_back(x1);
        vertices.push_back(y1);
        
        // Second point on circle
        float angle2 = 2.0f * 3.14159265359f * (i + 1) / segments;
        float x2 = centerX + radius * cosf(angle2);
        float y2 = centerY + radius * sinf(angle2);
        vertices.push_back(x2);
        vertices.push_back(y2);
    }
    
    // Create and bind VAO (Vertex Array Object)
    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    
    // Create and bind VBO (Vertex Buffer Object)
    unsigned int VBO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    
    // Set vertex attribute pointers
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Draw the circle using triangles (3 vertices per triangle, segments triangles)
    glDrawArrays(GL_TRIANGLES, 0, segments * 3);
    
    // Clean up
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
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

// Function to render a net (vertical line splitting the court)
void renderNet(float centerX, float topY, float bottomY, float lineWidth) {
    // Render the net as a vertical line made of a thin rectangle
    float halfWidth = lineWidth / 2.0f;
    
    float vertices[] = {
        // First triangle
        centerX - halfWidth, bottomY,  // Bottom-left
        centerX + halfWidth, bottomY,  // Bottom-right
        centerX - halfWidth, topY,     // Top-left
        // Second triangle
        centerX - halfWidth, topY,     // Top-left
        centerX + halfWidth, bottomY,  // Bottom-right
        centerX + halfWidth, topY      // Top-right
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
    
    // Draw the net
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
    
    // Create window with square aspect ratio (1:1) so circles appear circular
    GLFWwindow* window = glfwCreateWindow(600, 600, "2D Rectangle Renderer", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    
    // Create shader program for rectangle
    unsigned int shaderProgram = createShaderProgram();
    if (shaderProgram == 0) {
        std::cerr << "Failed to create shader program" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    // Create shader program for circle (with different color)
    unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    unsigned int circleFragmentShader = compileShader(GL_FRAGMENT_SHADER, circleFragmentShaderSource);
    unsigned int circleShaderProgram = 0;
    if (vertexShader != 0 && circleFragmentShader != 0) {
        circleShaderProgram = glCreateProgram();
        glAttachShader(circleShaderProgram, vertexShader);
        glAttachShader(circleShaderProgram, circleFragmentShader);
        glLinkProgram(circleShaderProgram);
        
        int success;
        char infoLog[512];
        glGetProgramiv(circleShaderProgram, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(circleShaderProgram, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
            circleShaderProgram = 0;
        }
        glDeleteShader(vertexShader);
        glDeleteShader(circleFragmentShader);
    }
    
    if (circleShaderProgram == 0) {
        std::cerr << "Failed to create circle shader program" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    // Create shader program for net
    unsigned int netVertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    unsigned int netFragmentShader = compileShader(GL_FRAGMENT_SHADER, netFragmentShaderSource);
    unsigned int netShaderProgram = 0;
    if (netVertexShader != 0 && netFragmentShader != 0) {
        netShaderProgram = glCreateProgram();
        glAttachShader(netShaderProgram, netVertexShader);
        glAttachShader(netShaderProgram, netFragmentShader);
        glLinkProgram(netShaderProgram);
        
        int success;
        char infoLog[512];
        glGetProgramiv(netShaderProgram, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(netShaderProgram, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
            netShaderProgram = 0;
        }
        glDeleteShader(netVertexShader);
        glDeleteShader(netFragmentShader);
    }
    
    if (netShaderProgram == 0) {
        std::cerr << "Failed to create net shader program" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    // Create shader program for air particles
    unsigned int airVertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    unsigned int airFragmentShader = compileShader(GL_FRAGMENT_SHADER, airParticleFragmentShaderSource);
    unsigned int airParticleShaderProgram = 0;
    if (airVertexShader != 0 && airFragmentShader != 0) {
        airParticleShaderProgram = glCreateProgram();
        glAttachShader(airParticleShaderProgram, airVertexShader);
        glAttachShader(airParticleShaderProgram, airFragmentShader);
        glLinkProgram(airParticleShaderProgram);
        
        int success;
        char infoLog[512];
        glGetProgramiv(airParticleShaderProgram, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(airParticleShaderProgram, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
            airParticleShaderProgram = 0;
        }
        glDeleteShader(airVertexShader);
        glDeleteShader(airFragmentShader);
    }
    
    if (airParticleShaderProgram == 0) {
        std::cerr << "Failed to create air particle shader program" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    // Rectangle boundaries (half width and half height)
    float rectHalfWidth = 1.5f / 2.0f;  // 0.75
    float rectHalfHeight = 1.3f / 2.0f;  // 0.65
    
    // Generate air particles (static, small circles)
    const int numAirParticles = 1000;
    float airParticleRadius = 0.02f;  // Much smaller than the ball
    std::vector<std::pair<float, float> > airParticles;
    
    // Random number generator for positioning air particles
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> xDist(-rectHalfWidth + airParticleRadius, rectHalfWidth - airParticleRadius);
    std::uniform_real_distribution<float> yDist(-rectHalfHeight + airParticleRadius, rectHalfHeight - airParticleRadius);
    
    // Generate random positions for air particles within the court
    for (int i = 0; i < numAirParticles; i++) {
        float x = xDist(gen);
        float y = yDist(gen);
        airParticles.push_back(std::make_pair(x, y));
    }
    
    // Circle physics variables
    float circleX = 0.0f;
    float circleY = 0.0f;
    float circleRadius = 0.1f;
    float circleVelX = 0.01f;  // Velocity in x direction
    float circleVelY = 0.005f;   // Velocity in y direction
    
    // Main render loop
    while (!glfwWindowShouldClose(window)) {
        // Update circle position
        circleX += circleVelX;
        circleY += circleVelY;
        
        // Check for collisions with rectangle boundaries and bounce
        // Left boundary
        if (circleX - circleRadius <= -rectHalfWidth) {
            circleX = -rectHalfWidth + circleRadius;
            circleVelX = -circleVelX;
        }
        // Right boundary
        if (circleX + circleRadius >= rectHalfWidth) {
            circleX = rectHalfWidth - circleRadius;
            circleVelX = -circleVelX;
        }
        // Bottom boundary
        if (circleY - circleRadius <= -rectHalfHeight) {
            circleY = -rectHalfHeight + circleRadius;
            circleVelY = -circleVelY;
        }
        // Top boundary
        if (circleY + circleRadius >= rectHalfHeight) {
            circleY = rectHalfHeight - circleRadius;
            circleVelY = -circleVelY;
        }
        
        // Clear the screen
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // Dark gray background
        glClear(GL_COLOR_BUFFER_BIT);
        
        // Render a rectangle at the center of the screen (the court)
        // Coordinates are in normalized device coordinates (-1 to 1)
        glUseProgram(shaderProgram);
        renderRectangle(0.0f, 0.0f, 1.5f, 1.3f); // Center, width=1.5, height=1.3
        
        // Render the net splitting the court in half (vertical line at x=0)
        glUseProgram(netShaderProgram);
        renderNet(0.0f, rectHalfHeight, -rectHalfHeight, 0.01f); // Center x=0, full height, thin line
        
        // Render air particles (static, small circles)
        glUseProgram(airParticleShaderProgram);
        for (size_t i = 0; i < airParticles.size(); i++) {
            renderCircle(airParticles[i].first, airParticles[i].second, airParticleRadius, 16);
        }
        
        // Render a circle (ball) inside the rectangle at its current position
        glUseProgram(circleShaderProgram);
        renderCircle(circleX, circleY, circleRadius, 32);
        
        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    // Cleanup
    glDeleteProgram(shaderProgram);
    glDeleteProgram(circleShaderProgram);
    glDeleteProgram(netShaderProgram);
    glDeleteProgram(airParticleShaderProgram);
    glfwTerminate();
    
    return 0;
}

