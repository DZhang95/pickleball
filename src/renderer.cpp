#define GL_SILENCE_DEPRECATION
#define GL_SILENCE_DEPRECATION
#ifdef __APPLE__
  #include <OpenGL/gl3.h>
#else
  #include <GL/glew.h>
#endif
#include <GLFW/glfw3.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cmath>
#include <random>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <numeric>
#include <fstream>

// Comment out to disable timing info
#define DEBUG

// Optional CUDA test API (scaffolded). See src/cuda/*
#include "cuda/cuda_kernels.h"

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
"    FragColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);\n"
"}\n";

// Fragment shader source code for air particles (light gray)
const char* airParticleFragmentShaderSource = 
"#version 330 core\n"
"out vec4 FragColor;\n"
"\n"
"void main() {\n"
"    FragColor = vec4(0.7f, 0.7f, 0.7f, 1.0f);\n"
"}\n";


// Fragment shader for path trace (thin red line)
const char* pathFragmentShaderSource =
"#version 330 core\n"
"out vec4 FragColor;\n"
"\n"
"void main() {\n"
"    FragColor = vec4(0.9f, 0.1f, 0.1f, 1.0f);\n"
"}\n"
;

// ******** Some world constants ********
// config: world physical size (meters)
const float WORLD_W = 13.4112f; // length (m)
const float WORLD_H = 6.096f;   // width (m)
const float world_cx = 0.0f, world_cy = 0.0f;
// Uniform NDC scale to preserve aspect (use same scale on X and Y)
const float NDC_SCALE = std::min(2.0f / WORLD_W, 2.0f / WORLD_H);
// Rectangle boundaries (half width and half height) in world meters
float rectHalfWidth = WORLD_W * 0.5f; // half-length
float rectHalfHeight = WORLD_H * 0.5f;   // half-width

// Physics constants
const float BALL_MASS = 0.026f;  // kg (26g)
const float AIR_PARCEL_MASS = 0.0001f;  // kg (0.1g) - represents many air molecules
const float BALL_RADIUS = 0.185f;
const float BALL_MOMENT_OF_INERTIA = (2.0f / 5.0f) * BALL_MASS * BALL_RADIUS * BALL_RADIUS;

float airParticleRadius = 0.005f;

const float IMPULSE_SCALE_FACTOR = 1;  // Can scale down impulses
const float AIR_AIR_IMPULSE_SCALE_FACTOR = 0.1f;

// Wind parameters (wind velocity in m/s)
// Set these to non-zero values to simulate wind
// For still air, set both to 0.0f
const float WIND_VELOCITY_X = 0.0f;  // Wind speed in x direction (m/s)
const float WIND_VELOCITY_Y = 0.0f;  // Wind speed in y direction (m/s)
// Wind turbulence: small random variations in wind speed
const float WIND_TURBULENCE = 0.1f;  // Random variation as fraction of wind speed

const float timestepSize = 0.001f;  // 0.001s time step
// **************************************    

// ******** Other Globals ********
// Generate air particles
int numAirParticles = 3000;
// Circle initial position and velocity
float circleX = -4.0f;
float circleY = -2.0f;
float circleVelX = 40.0f;  // Velocity in x direction
float circleVelY = 10.0f;   // Velocity in y direction
float circleSpin = 100.0f;  // Angular velocity (spin) - scalar in 2D
// If true the ball is allowed to leave the rectangular world (no bounce)
const bool allowBallEscape = true;
// If true air parcels are allowed to leave the rectangular world (no bounce)
const bool allowAirEscape = true;
// Path trace: store recent ball center positions in NDC space (only used if showPath)
std::vector<std::pair<float,float>> ballPath;
const size_t maxPathPoints = 5000; // cap to avoid unbounded growth
// **************************************    

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

// Timing helpers are enabled only when DEBUG is defined. When disabled we
// provide small no-op replacements so the rest of the code compiles and runs
// without the profiling overhead.
#ifdef DEBUG
static std::mutex g_timers_mutex;

// Running totals (ms) and counts per phase
static std::map<std::string, double> g_phase_total_ms;
static std::map<std::string, size_t> g_phase_counts;

struct ScopedTimer {
    std::string name;
    double sim_time; // -1 means unavailable
    std::chrono::steady_clock::time_point t0;

    // When created, track the time
    ScopedTimer(const std::string &n, double sim_time_ = -1.0)
      : name(n), sim_time(sim_time_), t0(std::chrono::steady_clock::now()) {}

    // When destroyed, print out the duration
    ~ScopedTimer() {
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::lock_guard<std::mutex> lk(g_timers_mutex);
        // update running totals and counts
        g_phase_total_ms[name] += ms;
        g_phase_counts[name] += 1;
        // print sim time if available, then phase and ms
        if (sim_time >= 0.0) {
            std::cerr << "TS=" << std::fixed << std::setprecision(6) << sim_time
                      << " " << name << " " << ms << " ms" << std::endl;
        } else {
            std::cerr << name << " " << ms << " ms" << std::endl;
        }
    }
};

// Print collected totals (called at end of simulation)
static void print_timing_totals() {
    std::lock_guard<std::mutex> lk(g_timers_mutex);
    std::cerr << "==== TIMING TOTALS ====" << std::endl;
    // Per-phase totals
    for (auto &kv : g_phase_total_ms) {
        const std::string &phase = kv.first;
        double total_ms = kv.second;
        size_t cnt = g_phase_counts[phase];
        double mean = (cnt > 0) ? (total_ms / (double)cnt) : 0.0;
        std::cerr << phase << ": total=" << total_ms << " ms, count=" << cnt << ", mean=" << mean << " ms" << std::endl;
    }

    // Totals of interest
    double frames_ms = g_phase_total_ms.count("frame_total") ? g_phase_total_ms["frame_total"] : 0.0;
    size_t frames_count = g_phase_counts.count("frame_total") ? g_phase_counts["frame_total"] : 0;
    double physics_total_ms = g_phase_total_ms.count("physics_total") ? g_phase_total_ms["physics_total"] : 0.0;

    // Sum all physics sub-phases (those that start with "physics_" but are not physics_total)
    double physics_subsum_ms = 0.0;
    for (auto &kv : g_phase_total_ms) {
        const std::string &phase = kv.first;
        if (phase.rfind("physics_", 0) == 0 && phase != std::string("physics_total")) {
            physics_subsum_ms += kv.second;
        }
    }

    std::cerr << "-- Summary --" << std::endl;
    std::cerr << "frames: count=" << frames_count << ", total_ms=" << frames_ms << ", mean_ms=" << (frames_count ? frames_ms / (double)frames_count : 0.0) << std::endl;
    std::cerr << "physics_total (timer): total_ms=" << physics_total_ms << std::endl;
    std::cerr << "physics_subtasks sum: total_ms=" << physics_subsum_ms << std::endl;
    std::cerr << "======================" << std::endl;
}
#else
// No-op versions when DEBUG is not defined
struct ScopedTimer {
    ScopedTimer(const std::string & /*n*/, double /*sim_time_*/ = -1.0) {}
};

static inline void print_timing_totals() {}
#endif

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
// Parameters: centerX, bottomY, topY, lineWidth (bottom before top - matches vertex ordering)
void renderNet(float centerX, float bottomY, float topY, float lineWidth) {
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

// Framebuffer resize callback: keep the GL viewport in sync with window framebuffer size
static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    // Avoid setting a zero-sized viewport
    if (width <= 0 || height <= 0) return;
    glViewport(0, 0, width, height);
}

// Air particle structure (designed for future upgrade to realistic physics)
struct AirParticle {
    float x, y;           // Position
    float velX, velY;      // Velocity
    float mass;            // Mass (for future realistic physics)
    
    AirParticle(float px, float py, float vx, float vy, float m) 
        : x(px), y(py), velX(vx), velY(vy), mass(m) {}
};

// World shaders structure to hold different shader programs
struct WorldShaders {
    unsigned int rectangleShader;
    unsigned int circleShader;
    unsigned int netShader;
    unsigned int airParticleShader;
    unsigned int pathShader;
};

GLFWwindow* gl_init(bool showPath, WorldShaders &shaders) {
        // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        throw std::runtime_error("Failed to initialize GLFW");
    }
    
    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    
    // 600x600 window
    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* vidMode = primaryMonitor ? glfwGetVideoMode(primaryMonitor) : NULL;
    int winW = 1200, winH = 1200;
    GLFWwindow* window = NULL;
    window = glfwCreateWindow(winW, winH, "2D Rectangle Renderer", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window);
#ifndef __APPLE__
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwTerminate();
        throw std::runtime_error("Failed to initialize GLEW");
    }
#endif    

    // Set initial viewport to the actual framebuffer size and register resize callback
    int fbW, fbH;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    if (fbW > 0 && fbH > 0) glViewport(0, 0, fbW, fbH);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    
    // Create shader program for rectangle
    shaders.rectangleShader = createShaderProgram();
    if (shaders.rectangleShader == 0) {
        std::cerr << "Failed to create shader program" << std::endl;
        glfwTerminate();
        throw std::runtime_error("Failed to create shader program");
    }
    
    // Create shader program for circle (with different color)
    unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    unsigned int circleFragmentShader = compileShader(GL_FRAGMENT_SHADER, circleFragmentShaderSource);
    if (vertexShader != 0 && circleFragmentShader != 0) {
        shaders.circleShader = glCreateProgram();
        glAttachShader(shaders.circleShader, vertexShader);
        glAttachShader(shaders.circleShader, circleFragmentShader);
        glLinkProgram(shaders.circleShader);
        
        int success;
        char infoLog[512];
        glGetProgramiv(shaders.circleShader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shaders.circleShader, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
            shaders.circleShader = 0;
        }
        glDeleteShader(vertexShader);
        glDeleteShader(circleFragmentShader);
    }
    
    if (shaders.circleShader == 0) {
        std::cerr << "Failed to create circle shader program" << std::endl;
        glfwTerminate();
        throw std::runtime_error("Failed to create circle shader program");
    }
    
    // Create shader program for net
    unsigned int netVertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    unsigned int netFragmentShader = compileShader(GL_FRAGMENT_SHADER, netFragmentShaderSource);
    if (netVertexShader != 0 && netFragmentShader != 0) {
        shaders.netShader = glCreateProgram();
        glAttachShader(shaders.netShader, netVertexShader);
        glAttachShader(shaders.netShader, netFragmentShader);
        glLinkProgram(shaders.netShader);
        
        int success;
        char infoLog[512];
        glGetProgramiv(shaders.netShader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shaders.netShader, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
            shaders.netShader = 0;
        }
        glDeleteShader(netVertexShader);
        glDeleteShader(netFragmentShader);
    }
    
    if (shaders.netShader == 0) {
        std::cerr << "Failed to create net shader program" << std::endl;
        glfwTerminate();
        throw std::runtime_error("Failed to create net shader program");
    }
    
    // Create shader program for air particles
    unsigned int airVertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    unsigned int airFragmentShader = compileShader(GL_FRAGMENT_SHADER, airParticleFragmentShaderSource);
    if (airVertexShader != 0 && airFragmentShader != 0) {
        shaders.airParticleShader = glCreateProgram();
        glAttachShader(shaders.airParticleShader, airVertexShader);
        glAttachShader(shaders.airParticleShader, airFragmentShader);
        glLinkProgram(shaders.airParticleShader);
        
        int success;
        char infoLog[512];
        glGetProgramiv(shaders.airParticleShader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shaders.airParticleShader, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
            shaders.airParticleShader = 0;
        }
        glDeleteShader(airVertexShader);
        glDeleteShader(airFragmentShader);
    }
    
    if (shaders.airParticleShader == 0) {
        std::cerr << "Failed to create air particle shader program" << std::endl;
        glfwTerminate();
        throw std::runtime_error("Failed to create air particle shader program");
    }

    // Optionally create shader program for the path trace (thin red line)
    if (showPath) {
        unsigned int pathVertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
        unsigned int pathFragmentShader = compileShader(GL_FRAGMENT_SHADER, pathFragmentShaderSource);
        if (pathVertexShader != 0 && pathFragmentShader != 0) {
            shaders.pathShader = glCreateProgram();
            glAttachShader(shaders.pathShader, pathVertexShader);
            glAttachShader(shaders.pathShader, pathFragmentShader);
            glLinkProgram(shaders.pathShader);

            int success;
            char infoLog[512];
            glGetProgramiv(shaders.pathShader, GL_LINK_STATUS, &success);
            if (!success) {
                glGetProgramInfoLog(shaders.pathShader, 512, NULL, infoLog);
                std::cerr << "WARNING: path shader linking failed; disabling path trace\n" << infoLog << std::endl;
                glDeleteProgram(shaders.pathShader);
                shaders.pathShader = 0;
                showPath = false;
            }
            glDeleteShader(pathVertexShader);
            glDeleteShader(pathFragmentShader);
        } else {
            std::cerr << "WARNING: path shaders failed to compile; disabling path trace" << std::endl;
            showPath = false;
        }
    }

    return window;
}

// Cleanup function to delete shaders and terminate GLFW
void cleanup(GLFWwindow* window, WorldShaders &shaders) {
    // Delete shader programs
    glDeleteProgram(shaders.rectangleShader);
    glDeleteProgram(shaders.circleShader);
    glDeleteProgram(shaders.netShader);
    glDeleteProgram(shaders.airParticleShader);
    if (shaders.pathShader != 0) {
        glDeleteProgram(shaders.pathShader);
    }
    
    // Terminate GLFW
    glfwDestroyWindow(window);
    glfwTerminate();
}

void initializeAirParticles(std::vector<AirParticle> &airParticles) {
    // Random number generator for positioning air particles
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> xDist(-rectHalfWidth + airParticleRadius, rectHalfWidth - airParticleRadius);
    std::uniform_real_distribution<float> yDist(-rectHalfHeight + airParticleRadius, rectHalfHeight - airParticleRadius);
    
    // Calculate wind velocity with turbulence
    float windSpeed = sqrtf(WIND_VELOCITY_X * WIND_VELOCITY_X + WIND_VELOCITY_Y * WIND_VELOCITY_Y);
    
    // Generate random positions and velocities for air particles within the court
    for (int i = 0; i < numAirParticles; i++) {
        float x = xDist(gen);
        float y = yDist(gen);
        
        // Set initial velocity based on wind
        float vx, vy;
        if (windSpeed > 0.0f) {
            // Wind is present: add wind velocity with some turbulence
            std::uniform_real_distribution<float> turbDist(-WIND_TURBULENCE, WIND_TURBULENCE);
            float turbX = turbDist(gen) * windSpeed;
            float turbY = turbDist(gen) * windSpeed;
            vx = WIND_VELOCITY_X + turbX;
            vy = WIND_VELOCITY_Y + turbY;
        } else {
            // Still air: particles have near-zero velocity
            // They only move due to collisions with the ball
            std::uniform_real_distribution<float> velDist(-0.00001f, 0.00001f);
            vx = velDist(gen);
            vy = velDist(gen);
        }
        
        airParticles.push_back(AirParticle(x, y, vx, vy, AIR_PARCEL_MASS));
    }
}

// Physics step extracted from main loop. Updates globals: circleX, circleY, circleVelX, circleVelY, circleSpin,
// and modifies the provided airParticles vector. spinDeltaAccumulator is updated with the total spin change
// produced during this timestep. use_cuda_requested is accepted for compatibility but this implementation
// currently uses the CPU path.
void simulatePhysics(double timestep, std::vector<AirParticle> &airParticles, bool use_cuda_requested, float &spinDeltaAccumulator, bool showPath) {
    ScopedTimer timer_physics("physics_total", timestep);

    // Update ball position based on velocity
    circleX += circleVelX * timestepSize;
    circleY += circleVelY * timestepSize;

    // Append current ball center (in NDC) to path trace if enabled
    if (showPath) {
        ballPath.emplace_back((circleX - world_cx) * NDC_SCALE, (circleY - world_cy) * NDC_SCALE);
        if (ballPath.size() > maxPathPoints) {
            size_t removeCount = ballPath.size() - maxPathPoints;
            ballPath.erase(ballPath.begin(), ballPath.begin() + removeCount);
        }
    }

    // If CUDA requested, attempt to run the whole particle-step + ball-particle collisions on GPU.
    // Track whether the GPU actually completed the work successfully (gpu_ok). If it fails
    // we must still perform the CPU ball-particle collision step later.
    bool gpu_ok = false;
    if (use_cuda_requested) {
        int n = (int)airParticles.size();
        std::vector<float> px(n), py(n), pvx(n), pvy(n);
        for (int i = 0; i < n; ++i) {
            px[i] = airParticles[i].x;
            py[i] = airParticles[i].y;
            pvx[i] = airParticles[i].velX;
            pvy[i] = airParticles[i].velY;
        }

        // ballState: [circleX, circleY, circleVelX, circleVelY, circleSpin]
        float ballState[5] = { circleX, circleY, circleVelX, circleVelY, circleSpin };

        bool ok = cuda_physics_run(px.data(), py.data(), pvx.data(), pvy.data(), n, ballState, (float)timestepSize);
        if (ok) {
            // copy back particle arrays
            for (int i = 0; i < n; ++i) {
                airParticles[i].x = px[i];
                airParticles[i].y = py[i];
                airParticles[i].velX = pvx[i];
                airParticles[i].velY = pvy[i];
            }
            // Update ball velocities and spin from returned state (positions are handled on host)
            circleVelX = ballState[2];
            circleVelY = ballState[3];
            circleSpin = ballState[4];
            gpu_ok = true;
        } else {
            // If GPU run failed, fall back to CPU integration for this small step
            for (size_t i = 0; i < airParticles.size(); i++) {
                airParticles[i].x += airParticles[i].velX * timestepSize;
                airParticles[i].y += airParticles[i].velY * timestepSize;
            }
        }
    } else {
        for (size_t i = 0; i < airParticles.size(); i++) {
            airParticles[i].x += airParticles[i].velX * timestepSize;
            airParticles[i].y += airParticles[i].velY * timestepSize;
        }
    }

    // Check for collisions between air particles (AIR-AIR collisions)
    // If GPU did the physics step successfully (gpu_ok) then the GPU already handled
    // air-air collisions; otherwise run the CPU fallback.
    if (!use_cuda_requested || !gpu_ok) {
        for (size_t i = 0; i < airParticles.size(); i++) {
            for (size_t j = i + 1; j < airParticles.size(); j++) {
                AirParticle& particle1 = airParticles[i];
                AirParticle& particle2 = airParticles[j];

                float dx = particle2.x - particle1.x;
                float dy = particle2.y - particle1.y;
                float distance = sqrtf(dx * dx + dy * dy);
                float minDistance = 2.0f * airParticleRadius;

                if (distance < minDistance && distance > 0.0001f) {
                    float nx = dx / distance;
                    float ny = dy / distance;

                    float overlap = minDistance - distance;
                    float separationX = nx * overlap * 0.5f;
                    float separationY = ny * overlap * 0.5f;

                    particle1.x -= separationX;
                    particle1.y -= separationY;
                    particle2.x += separationX;
                    particle2.y += separationY;

                    float relVelX = particle2.velX - particle1.velX;
                    float relVelY = particle2.velY - particle1.velY;

                    float v_rel_dot_n = relVelX * nx + relVelY * ny;
                    float impulseX = 2.0f * AIR_PARCEL_MASS * v_rel_dot_n * nx;
                    float impulseY = 2.0f * AIR_PARCEL_MASS * v_rel_dot_n * ny;

                    impulseX *= AIR_AIR_IMPULSE_SCALE_FACTOR;
                    impulseY *= AIR_AIR_IMPULSE_SCALE_FACTOR;

                    particle1.velX += impulseX / AIR_PARCEL_MASS;
                    particle1.velY += impulseY / AIR_PARCEL_MASS;
                    particle2.velX -= impulseX / AIR_PARCEL_MASS;
                    particle2.velY -= impulseY / AIR_PARCEL_MASS;
                }
            }
        }
    }

    // Check for collisions between ball and air particles.
    // If CUDA was requested and successfully performed the whole physics step (gpu_ok==true)
    // then the GPU already handled ball-particle collisions and we can skip the CPU code.
    // Otherwise (CUDA not requested or GPU run failed), fall back to the CPU collision code.
    if (!use_cuda_requested || !gpu_ok) {
        for (size_t i = 0; i < airParticles.size(); i++) {
            AirParticle& particle = airParticles[i];

            float dx = particle.x - circleX;
            float dy = particle.y - circleY;
            float distance = sqrtf(dx * dx + dy * dy);
            float minDistance = BALL_RADIUS + airParticleRadius;

            if (distance < minDistance && distance > 0.0001f) {
                float r_mag = sqrtf(dx * dx + dy * dy);
                float nx = dx / r_mag;
                float ny = dy / r_mag;

                float rx = nx * BALL_RADIUS;
                float ry = ny * BALL_RADIUS;

                float overlap = minDistance - distance;
                float separationX = nx * overlap * 0.5f;
                float separationY = ny * overlap * 0.5f;

                float totalMass = BALL_MASS + particle.mass;
                circleX -= separationX * (particle.mass / totalMass);
                circleY -= separationY * (particle.mass / totalMass);
                particle.x += separationX * (BALL_MASS / totalMass);
                particle.y += separationY * (BALL_MASS / totalMass);

                float surfaceVelX = circleVelX - circleSpin * ry;
                float surfaceVelY = circleVelY + circleSpin * rx;

                float relVelX = particle.velX - surfaceVelX;
                float relVelY = particle.velY - surfaceVelY;

                float v_rel_n = relVelX * nx + relVelY * ny;
                if (v_rel_n < 0.0f) {
                    float invMassBall = 1.0f / BALL_MASS;
                    float invMassPart = 1.0f / particle.mass;

                    const float restitution = 0.0f;
                    float Jn_mag = -(1.0f + restitution) * v_rel_n / (invMassBall + invMassPart);

                    float tx = -ny;
                    float ty = nx;
                    float v_rel_t = relVelX * tx + relVelY * ty;

                    float denom_t = invMassBall + invMassPart + (BALL_RADIUS * BALL_RADIUS) / BALL_MOMENT_OF_INERTIA;
                    float Jt_unc = - v_rel_t / denom_t;

                    const float mu = 0.05f;
                    float Jt = 0.0f;
                    if (fabsf(Jt_unc) <= mu * Jn_mag) {
                        Jt = Jt_unc;
                    } else {
                        Jt = (Jt_unc > 0.0f ? 1.0f : -1.0f) * mu * Jn_mag;
                    }

                    float impulseX = Jn_mag * nx + Jt * tx;
                    float impulseY = Jn_mag * ny + Jt * ty;

                    impulseX *= IMPULSE_SCALE_FACTOR;
                    impulseY *= IMPULSE_SCALE_FACTOR;

                    circleVelX += impulseX / BALL_MASS;
                    circleVelY += impulseY / BALL_MASS;

                    float r_cross_J = rx * impulseY - ry * impulseX;
                    float spin_change = r_cross_J / BALL_MOMENT_OF_INERTIA;
                    circleSpin += spin_change;
                    spinDeltaAccumulator += spin_change;

                    particle.velX -= impulseX / particle.mass;
                    particle.velY -= impulseY / particle.mass;
                }
            }
        }

        // Check collisions with rectangle boundaries for ball
        if (!allowBallEscape) {
            if (circleX - BALL_RADIUS <= -rectHalfWidth) {
                circleX = -rectHalfWidth + BALL_RADIUS;
                circleVelX = -circleVelX;
            }
            if (circleX + BALL_RADIUS >= rectHalfWidth) {
                circleX = rectHalfWidth - BALL_RADIUS;
                circleVelX = -circleVelX;
            }
            if (circleY - BALL_RADIUS <= -rectHalfHeight) {
                circleY = -rectHalfHeight + BALL_RADIUS;
                circleVelY = -circleVelY;
            }
            if (circleY + BALL_RADIUS >= rectHalfHeight) {
                circleY = rectHalfHeight - BALL_RADIUS;
                circleVelY = -circleVelY;
            }
        }

        // Check for collisions with rectangle boundaries (air particles)
        for (size_t i = 0; i < airParticles.size(); ) {
            AirParticle& particle = airParticles[i];

            if (!allowAirEscape) {
                if (particle.x - airParticleRadius <= -rectHalfWidth) {
                    particle.x = -rectHalfWidth + airParticleRadius;
                    particle.velX = -particle.velX;
                }
                if (particle.x + airParticleRadius >= rectHalfWidth) {
                    particle.x = rectHalfWidth - airParticleRadius;
                    particle.velX = -particle.velX;
                }
                if (particle.y - airParticleRadius <= -rectHalfHeight) {
                    particle.y = -rectHalfHeight + airParticleRadius;
                    particle.velY = -particle.velY;
                }
                if (particle.y + airParticleRadius >= rectHalfHeight) {
                    particle.y = rectHalfHeight - airParticleRadius;
                    particle.velY = -particle.velY;
                }

                ++i;
            } else {
                bool outsideLeft = (particle.x + airParticleRadius < -rectHalfWidth);
                bool outsideRight = (particle.x - airParticleRadius > rectHalfWidth);
                bool outsideBottom = (particle.y + airParticleRadius < -rectHalfHeight);
                bool outsideTop = (particle.y - airParticleRadius > rectHalfHeight);

                if (outsideLeft || outsideRight || outsideBottom || outsideTop) {
                    airParticles.erase(airParticles.begin() + i);
                } else {
                    ++i;
                }
            }
        }

        // Continuous Magnus effect
        {
            const float k_magnus = 0.0005f;
            float magnusAx = k_magnus * (-circleSpin * circleVelY) / BALL_MASS;
            float magnusAy = k_magnus * ( circleSpin * circleVelX) / BALL_MASS;
            circleVelX += magnusAx * timestepSize;
            circleVelY += magnusAy * timestepSize;
        }
    }
}

// Rendering function extracted from main loop. Renders the court, net, particles,
// ball and optional path. Uses globals for world-to-NDC constants and ball state.
void renderFrame(GLFWwindow* window, WorldShaders &shaders, std::vector<AirParticle> &airParticles, double timestep, bool showPath) {
    ScopedTimer timer_render("render_total", timestep);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // Dark gray background
    glClear(GL_COLOR_BUFFER_BIT);

    // Render a rectangle at the center of the screen (the court)
    glUseProgram(shaders.rectangleShader);
    float rect_w_ndc = WORLD_W * NDC_SCALE;
    float rect_h_ndc = WORLD_H * NDC_SCALE;
    renderRectangle(0.0f, 0.0f, rect_w_ndc, rect_h_ndc); // Center in NDC

    // Render the net splitting the court in half (vertical line at x=0)
    glUseProgram(shaders.netShader);
    // Draw net with a thickness of ~3 pixels (framebuffer-aware)
    {
        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        float pixels_per_ndc = (fbW > 0) ? (fbW / 2.0f) : 600.0f;
        const float desired_pixels = 3.0f;
        float net_ndc_width = desired_pixels / pixels_per_ndc;
        renderNet((0.0f - world_cx) * NDC_SCALE, ( -rectHalfHeight - world_cy) * NDC_SCALE, (rectHalfHeight - world_cy) * NDC_SCALE, net_ndc_width);
    }

    // Render air particles (now moving, small circles)
    glUseProgram(shaders.airParticleShader);
    for (size_t i = 0; i < airParticles.size(); i++) {
        float px = (airParticles[i].x - world_cx) * NDC_SCALE;
        float py = (airParticles[i].y - world_cy) * NDC_SCALE;
        float pr = airParticleRadius * NDC_SCALE;
        renderCircle(px, py, pr, 16);
    }

    // Render a circle (ball) inside the rectangle at its current position
    glUseProgram(shaders.circleShader);
    renderCircle((circleX - world_cx) * NDC_SCALE, (circleY - world_cy) * NDC_SCALE, BALL_RADIUS * NDC_SCALE, 32);

    // Render the ball path as a thin red line (GL_LINE_STRIP) if enabled
    if (showPath && ballPath.size() >= 2 && shaders.pathShader != 0) {
        glUseProgram(shaders.pathShader);
        std::vector<float> pathVerts;
        pathVerts.reserve(ballPath.size() * 2);
        for (const auto &pt : ballPath) {
            pathVerts.push_back(pt.first);
            pathVerts.push_back(pt.second);
        }

        unsigned int pathVAO, pathVBO;
        glGenVertexArrays(1, &pathVAO);
        glBindVertexArray(pathVAO);

        glGenBuffers(1, &pathVBO);
        glBindBuffer(GL_ARRAY_BUFFER, pathVBO);
        glBufferData(GL_ARRAY_BUFFER, pathVerts.size() * sizeof(float), pathVerts.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        glLineWidth(1.5f);
        glDrawArrays(GL_LINE_STRIP, 0, (GLsizei)ballPath.size());

        glDeleteBuffers(1, &pathVBO);
        glDeleteVertexArrays(1, &pathVAO);
    }
}

int main(int argc, char** argv) {
    // Parse command-line arguments
    bool showPath = false; // off by default
    bool use_cuda_requested = false;
    for (int ai = 1; ai < argc; ++ai) {
        if (std::strcmp(argv[ai], "--trace-path") == 0 || std::strcmp(argv[ai], "-t") == 0 || std::strcmp(argv[ai], "--path") == 0) {
            showPath = true;
            break;
        }
        if (std::strcmp(argv[ai], "--use-cuda") == 0 || std::strcmp(argv[ai], "--cuda") == 0) {
            use_cuda_requested = true;
        }
    }
    // Quick sanity write to stderr so we can verify logging is captured
    std::cerr << "TIMING: program_start" << std::endl;

    // Initialize OpenGL and create window
    WorldShaders shaders;
    GLFWwindow* window = gl_init(showPath, shaders);
    
    // World to NDC conversion lambdas
    auto world_to_ndc_x = [&](float x){
        return (x - world_cx) * NDC_SCALE;
    };
    auto world_to_ndc_y = [&](float y){
        return (y - world_cy) * NDC_SCALE;
    };
    auto world_to_ndc_scale = [&](float s){
        return s * NDC_SCALE;
    };
        
    std::vector<AirParticle> airParticles;
    initializeAirParticles(airParticles);
    // If requested, try to initialize CUDA persistent buffers for physics
    if (use_cuda_requested) {
        if (!cuda_physics_init((int)airParticles.size())) {
            std::cerr << "WARNING: cuda_physics_init failed; disabling CUDA path" << std::endl;
            use_cuda_requested = false;
        } else {
            std::cerr << "INFO: cuda_physics_init succeeded for n=" << airParticles.size() << std::endl;
        }
    }
    
    // Main render loop
    float timestep = 0.0f;
    while (!glfwWindowShouldClose(window)) {
        ScopedTimer timer_frame("frame_total", timestep);
        // Accumulate spin changes for HUD/debugging this frame
        float spinDeltaAccumulator = 0.0f;
        // --- Physics section (timed) ---
        simulatePhysics(timestep, airParticles, use_cuda_requested, spinDeltaAccumulator, showPath);
        
        // Clear the screen and render (timed)
        renderFrame(window, shaders, airParticles, timestep, showPath);

        // Update an on-screen HUD via the window title with ball position, velocity, spin, dspin and timestepSize
        {
            std::ostringstream oss;
            // Increase precision so small spin changes are visible in the HUD
            oss << std::fixed << std::setprecision(4);
            oss << "Ball pos=(" << circleX << "," << circleY << ") ";
            oss << "vel=(" << circleVelX << "," << circleVelY << ") ";
            oss << "spin=" << circleSpin << " ";
            oss << "dspin=" << spinDeltaAccumulator << " ";
            // Show timestep in seconds (same numeric precision)
            oss << "dt=" << timestep << "s";
            std::string title = oss.str();
            glfwSetWindowTitle(window, title.c_str());
        }

        // If ball is allowed to escape, terminate the program when it fully leaves
        // the rectangular world. We consider the ball to have escaped when its
        // entire circle (center +/- radius) lies outside the rectangle on any side.
        if (allowBallEscape) {
            bool ballOutsideLeft = (circleX + BALL_RADIUS < -rectHalfWidth);
            bool ballOutsideRight = (circleX - BALL_RADIUS > rectHalfWidth);
            bool ballOutsideBottom = (circleY + BALL_RADIUS < -rectHalfHeight);
            bool ballOutsideTop = (circleY - BALL_RADIUS > rectHalfHeight);

            if (ballOutsideLeft || ballOutsideRight || ballOutsideBottom || ballOutsideTop) {
                break;
            }
        }        

        // Increment timestep
        timestep += timestepSize;

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Print totals before entering the pause/keep-open loop
    print_timing_totals();

    while(1){
        // Keep window open after simulation ends
        if (glfwWindowShouldClose(window)) {
            break;
        }
        glfwPollEvents();
    }

    // Cleanup
    // Destroy CUDA persistent buffers if allocated
    if (use_cuda_requested) cuda_physics_destroy();

    cleanup(window, shaders);
    
    return 0;
}

