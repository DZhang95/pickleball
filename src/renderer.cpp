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


// Fragment shader for path trace (thin red line)
const char* pathFragmentShaderSource =
"#version 330 core\n"
"out vec4 FragColor;\n"
"\n"
"void main() {\n"
"    FragColor = vec4(0.9f, 0.1f, 0.1f, 1.0f);\n"
"}\n"
;
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

int main(int argc, char** argv) {
    // Parse command-line arguments
    bool showPath = false; // off by default
    for (int ai = 1; ai < argc; ++ai) {
        if (std::strcmp(argv[ai], "--trace-path") == 0 || std::strcmp(argv[ai], "-t") == 0 || std::strcmp(argv[ai], "--path") == 0) {
            showPath = true;
            break;
        }
    }
    // Quick sanity write to stderr so we can verify logging is captured
    std::cerr << "TIMING: program_start" << std::endl;
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
    
    // 600x600 window
    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* vidMode = primaryMonitor ? glfwGetVideoMode(primaryMonitor) : NULL;
    int winW = 1200, winH = 1200;
    GLFWwindow* window = NULL;
    window = glfwCreateWindow(winW, winH, "2D Rectangle Renderer", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
#ifndef __APPLE__
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwTerminate();
        return -1;
    }
#endif    

    // Set initial viewport to the actual framebuffer size and register resize callback
    int fbW, fbH;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    if (fbW > 0 && fbH > 0) glViewport(0, 0, fbW, fbH);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    
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

    // Optionally create shader program for the path trace (thin red line)
    unsigned int pathShaderProgram = 0;
    if (showPath) {
        unsigned int pathVertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
        unsigned int pathFragmentShader = compileShader(GL_FRAGMENT_SHADER, pathFragmentShaderSource);
        if (pathVertexShader != 0 && pathFragmentShader != 0) {
            pathShaderProgram = glCreateProgram();
            glAttachShader(pathShaderProgram, pathVertexShader);
            glAttachShader(pathShaderProgram, pathFragmentShader);
            glLinkProgram(pathShaderProgram);

            int success;
            char infoLog[512];
            glGetProgramiv(pathShaderProgram, GL_LINK_STATUS, &success);
            if (!success) {
                glGetProgramInfoLog(pathShaderProgram, 512, NULL, infoLog);
                std::cerr << "WARNING: path shader linking failed; disabling path trace\n" << infoLog << std::endl;
                glDeleteProgram(pathShaderProgram);
                pathShaderProgram = 0;
                showPath = false;
            }
            glDeleteShader(pathVertexShader);
            glDeleteShader(pathFragmentShader);
        } else {
            std::cerr << "WARNING: path shaders failed to compile; disabling path trace" << std::endl;
            showPath = false;
        }
    }
    
    // config: world physical size (meters)
    const float WORLD_W = 13.4112f; // length (m)
    const float WORLD_H = 6.096f;   // width (m)
    const float world_cx = 0.0f, world_cy = 0.0f;

    // Uniform NDC scale to preserve aspect (use same scale on X and Y)
    const float NDC_SCALE = std::min(2.0f / WORLD_W, 2.0f / WORLD_H);

    auto world_to_ndc_x = [&](float x){
        return (x - world_cx) * NDC_SCALE;
    };
    auto world_to_ndc_y = [&](float y){
        return (y - world_cy) * NDC_SCALE;
    };
    auto world_to_ndc_scale = [&](float s){
        return s * NDC_SCALE;
    };
    
    // Rectangle boundaries (half width and half height) in world meters
    float rectHalfWidth = WORLD_W * 0.5f; // half-length
    float rectHalfHeight = WORLD_H * 0.5f;   // half-width
    
    // Generate air particles (now with velocities for collision response)
    const int numAirParticles = 3000;
    
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
    
    // Use proper physics constants
    float ballMass = BALL_MASS;
    float particleMass = AIR_PARCEL_MASS;
    float circleRadius = BALL_RADIUS;  // Use actual ball radius
    
    std::vector<AirParticle> airParticles;
    
    // Random number generator for positioning air particles
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> xDist(-rectHalfWidth + airParticleRadius, rectHalfWidth - airParticleRadius);
    std::uniform_real_distribution<float> yDist(-rectHalfHeight + airParticleRadius, rectHalfHeight - airParticleRadius);
    
    // Calculate wind velocity with turbulence
    float windVx = WIND_VELOCITY_X;
    float windVy = WIND_VELOCITY_Y;
    float windSpeed = sqrtf(windVx * windVx + windVy * windVy);
    
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
            vx = windVx + turbX;
            vy = windVy + turbY;
        } else {
            // Still air: particles have near-zero velocity
            // They only move due to collisions with the ball
            std::uniform_real_distribution<float> velDist(-0.00001f, 0.00001f);
            vx = velDist(gen);
            vy = velDist(gen);
        }
        
        airParticles.push_back(AirParticle(x, y, vx, vy, particleMass));
    }
    
    // Circle physics variables
    float circleX = -4.0f;
    float circleY = -2.0f;
    // circleRadius is now set above from BALL_RADIUS
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
    
    // Main render loop
    const float timestepSize = 0.001f;  // 0.001s time step (from physics.txt)
    float timestep = 0.0f;
    while (!glfwWindowShouldClose(window)) {
        ScopedTimer timer_frame("frame_total", timestep);
        // Accumulate spin changes for HUD/debugging this frame
        float spinDeltaAccumulator = 0.0f;
        // --- Physics section (timed) ---
        {
            ScopedTimer timer_physics("physics_total", timestep);
            // Update ball position based on velocity
            circleX += circleVelX * timestepSize;
            circleY += circleVelY * timestepSize;

            // Append current ball center (in NDC) to path trace if enabled
            if (showPath) {
                ballPath.emplace_back(world_to_ndc_x(circleX), world_to_ndc_y(circleY));
                if (ballPath.size() > maxPathPoints) {
                    // remove oldest points to keep vector size bounded
                    size_t removeCount = ballPath.size() - maxPathPoints;
                    ballPath.erase(ballPath.begin(), ballPath.begin() + removeCount);
                }
            }

            // Update air particle positions
            {
                ScopedTimer timer_air_integrate("physics_air_integrate", timestep);
                for (size_t i = 0; i < airParticles.size(); i++) {
                    airParticles[i].x += airParticles[i].velX * timestepSize;
                    airParticles[i].y += airParticles[i].velY * timestepSize;
                }
            }

            // Check for collisions between air particles (AIR-AIR collisions)
            // Using impulse-based physics: calculate relative velocity and impulse
            {
                ScopedTimer timer_air_air("physics_air_air");
                for (size_t i = 0; i < airParticles.size(); i++) {
                    for (size_t j = i + 1; j < airParticles.size(); j++) {
                    AirParticle& particle1 = airParticles[i];
                    AirParticle& particle2 = airParticles[j];
                    
                    // Calculate distance between particle centers
                    float dx = particle2.x - particle1.x;
                    float dy = particle2.y - particle1.y;
                    float distance = sqrtf(dx * dx + dy * dy);
                    float minDistance = 2.0f * airParticleRadius;  // Both particles have same radius
                    
                    if (distance < minDistance && distance > 0.0001f) {  // Collision detected
                        // Normalize collision vector (from particle1 to particle2)
                        float nx = dx / distance;  // Unit normal n
                        float ny = dy / distance;

                        // Separate overlapping particles
                        float overlap = minDistance - distance;
                        float separationX = nx * overlap * 0.5f;
                        float separationY = ny * overlap * 0.5f;

                        // Push particles apart (equal mass, so equal separation)
                        particle1.x -= separationX;
                        particle1.y -= separationY;
                        particle2.x += separationX;
                        particle2.y += separationY;

                        // Calculate relative velocity: v_rel = v2 - v1
                        float relVelX = particle2.velX - particle1.velX;
                        float relVelY = particle2.velY - particle1.velY;

                        // Calculate impulse: J = 2 * m * dot(v_rel, n) * n
                        // For air-air collisions, both particles have mass m = AIR_PARCEL_MASS
                        float v_rel_dot_n = relVelX * nx + relVelY * ny;
                        float impulseX = 2.0f * AIR_PARCEL_MASS * v_rel_dot_n * nx;
                        float impulseY = 2.0f * AIR_PARCEL_MASS * v_rel_dot_n * ny;

                        // Scale down impulse for air-air collisions (they're less important)
                        impulseX *= AIR_AIR_IMPULSE_SCALE_FACTOR;
                        impulseY *= AIR_AIR_IMPULSE_SCALE_FACTOR;

                        // Update velocities: v_new = v + (J / m)
                        // Particle 1 gets +J/m (in direction of n)
                        // Particle 2 gets -J/m (opposite direction)
                        particle1.velX += impulseX / AIR_PARCEL_MASS;
                        particle1.velY += impulseY / AIR_PARCEL_MASS;
                        particle2.velX -= impulseX / AIR_PARCEL_MASS;
                        particle2.velY -= impulseY / AIR_PARCEL_MASS;
                    }
                    }
                }
            }

            // Check for collisions between ball and air particles
            // Using the physics model: calculate surface velocity, relative velocity, and impulse
            {
                ScopedTimer timer_ball_air("physics_ball_air", timestep);
                for (size_t i = 0; i < airParticles.size(); i++) {
                    AirParticle& particle = airParticles[i];
                
                // Calculate distance between ball and particle centers
                float dx = particle.x - circleX;  // Vector from ball to particle
                float dy = particle.y - circleY;
                float distance = sqrtf(dx * dx + dy * dy);
                float minDistance = circleRadius + airParticleRadius;
                
                if (distance < minDistance && distance > 0.0001f) {  // Collision detected
                    // Normalize collision vector (r: from ball center to collision point)
                    float r_mag = sqrtf(dx * dx + dy * dy);
                    float nx = dx / r_mag;  // Unit normal n = r / |r|
                    float ny = dy / r_mag;

                    // r vector with magnitude R (ball radius) pointing to collision point
                    float rx = nx * circleRadius;
                    float ry = ny * circleRadius;

                    // Separate overlapping objects
                    float overlap = minDistance - distance;
                    float separationX = nx * overlap * 0.5f;
                    float separationY = ny * overlap * 0.5f;

                    // Push objects apart (proportional to mass)
                    float totalMass = ballMass + particle.mass;
                    circleX -= separationX * (particle.mass / totalMass);
                    circleY -= separationY * (particle.mass / totalMass);
                    particle.x += separationX * (ballMass / totalMass);
                    particle.y += separationY * (ballMass / totalMass);

                    // Calculate surface velocity at point of impact: v_surface = v_ball + (ω × r)
                    // In 2D: ω × r = (-ω * r.y, ω * r.x) where ω is scalar angular velocity
                    float surfaceVelX = circleVelX - circleSpin * ry;
                    float surfaceVelY = circleVelY + circleSpin * rx;

                    // Calculate relative velocity: v_rel = u_air - v_surface
                    float relVelX = particle.velX - surfaceVelX;
                    float relVelY = particle.velY - surfaceVelY;

                    // Coulomb-friction impulse model (normal + tangential)
                    // Compute normal relative velocity
                    float v_rel_n = relVelX * nx + relVelY * ny;
                    if (v_rel_n < 0.0f) { // only apply impulse if bodies are approaching
                        // Effective inverse masses
                        float invMassBall = 1.0f / ballMass;
                        float invMassPart = 1.0f / particle.mass;

                        // Normal impulse magnitude (restitution e = 0 for inelastic)
                        const float restitution = 0.0f;
                        float Jn_mag = -(1.0f + restitution) * v_rel_n / (invMassBall + invMassPart);

                        // Tangent vector (unit)
                        float tx = -ny;
                        float ty = nx;
                        // Relative tangential velocity
                        float v_rel_t = relVelX * tx + relVelY * ty;

                        // Denominator for tangential impulse includes rotational coupling: R^2 / I
                        float denom_t = invMassBall + invMassPart + (circleRadius * circleRadius) / BALL_MOMENT_OF_INERTIA;
                        // Unclamped tangential impulse that would cancel tangential velocity
                        float Jt_unc = - v_rel_t / denom_t;

                        // Coulomb friction coefficient (tuneable)
                        const float mu = 0.05f;
                        float Jt = 0.0f;
                        if (fabsf(Jt_unc) <= mu * Jn_mag) {
                            // Sticking: use full tangential impulse
                            Jt = Jt_unc;
                        } else {
                            // Sliding: clamp to Coulomb limit, direction opposes motion
                            Jt = (Jt_unc > 0.0f ? 1.0f : -1.0f) * mu * Jn_mag;
                        }

                        // Total impulse = normal + tangential
                        float impulseX = Jn_mag * nx + Jt * tx;
                        float impulseY = Jn_mag * ny + Jt * ty;

                        // Scale down overall impulse if desired (keeps prior behavior adjustable)
                        impulseX *= IMPULSE_SCALE_FACTOR;
                        impulseY *= IMPULSE_SCALE_FACTOR;

                        // Apply linear impulse to ball and particle
                        circleVelX += impulseX / ballMass;
                        circleVelY += impulseY / ballMass;

                        // Angular effect from tangential component (normal component gives zero torque because r is radial)
                        float r_cross_J = rx * impulseY - ry * impulseX;
                        float spin_change = r_cross_J / BALL_MOMENT_OF_INERTIA;
                        circleSpin += spin_change;
                        spinDeltaAccumulator += spin_change;

                        // Particle receives opposite impulse
                        particle.velX -= impulseX / particle.mass;
                        particle.velY -= impulseY / particle.mass;
                    }
                }
            }

            // Check for collisions with rectangle boundaries and bounce (ball)
            // If `allowBallEscape` is true we skip bouncing and let the ball leave the world.
            if (!allowBallEscape) {
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
            }

            // Check for collisions with rectangle boundaries (air particles)
            // If allowAirEscape == false -> bounce at boundaries
            // If allowAirEscape == true -> delete particles once they have fully left the world
            for (size_t i = 0; i < airParticles.size(); ) {
                AirParticle& particle = airParticles[i];

                if (!allowAirEscape) {
                    // Left boundary
                    if (particle.x - airParticleRadius <= -rectHalfWidth) {
                        particle.x = -rectHalfWidth + airParticleRadius;
                        particle.velX = -particle.velX;
                    }
                    // Right boundary
                    if (particle.x + airParticleRadius >= rectHalfWidth) {
                        particle.x = rectHalfWidth - airParticleRadius;
                        particle.velX = -particle.velX;
                    }
                    // Bottom boundary
                    if (particle.y - airParticleRadius <= -rectHalfHeight) {
                        particle.y = -rectHalfHeight + airParticleRadius;
                        particle.velY = -particle.velY;
                    }
                    // Top boundary
                    if (particle.y + airParticleRadius >= rectHalfHeight) {
                        particle.y = rectHalfHeight - airParticleRadius;
                        particle.velY = -particle.velY;
                    }

                    // keep this particle, advance index
                    ++i;
                } else {
                    // Determine if the particle has fully left the rectangular world.
                    // We treat a particle as escaped when its entire circle (center +/- radius)
                    // lies outside the rectangle on any side.
                    bool outsideLeft = (particle.x + airParticleRadius < -rectHalfWidth);
                    bool outsideRight = (particle.x - airParticleRadius > rectHalfWidth);
                    bool outsideBottom = (particle.y + airParticleRadius < -rectHalfHeight);
                    bool outsideTop = (particle.y - airParticleRadius > rectHalfHeight);

                    if (outsideLeft || outsideRight || outsideBottom || outsideTop) {
                        // Erase this particle. We do not increment i so the next element
                        // that shifted into position i will be processed on the next iteration.
                        airParticles.erase(airParticles.begin() + i);
                    } else {
                        // Particle still inside (or partially overlapping); keep it
                        ++i;
                    }
                }
            }

            // Continuous Magnus effect: add a lateral acceleration proportional to (omega x v)
            // Simple 2D approximation: F_magnus = k_magnus * (omega x v), where omega is scalar
            // In 2D: omega x v = (-omega * v.y, omega * v.x)
            // a = F / m -> velocity update: v += (k_magnus * (omega x v) / m) * dt
            {
                ScopedTimer timer_magnus("physics_magnus");
                const float k_magnus = 0.0005f; // tune this constant to change effect strength
                // Compute magnus acceleration components
                float magnusAx = k_magnus * (-circleSpin * circleVelY) / ballMass;
                float magnusAy = k_magnus * ( circleSpin * circleVelX) / ballMass;
                // Integrate into velocity
                circleVelX += magnusAx * timestepSize;
                circleVelY += magnusAy * timestepSize;
            }

            }
        } // end physics_total scope
        
        // Clear the screen and render (timed)
        {
            ScopedTimer timer_render("render_total", timestep);
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // Dark gray background
            glClear(GL_COLOR_BUFFER_BIT);
        
            // Render a rectangle at the center of the screen (the court)
            // Convert world meters -> NDC using uniform scale so the whole court fits
            glUseProgram(shaderProgram);
            float rect_w_ndc = WORLD_W * NDC_SCALE;
            float rect_h_ndc = WORLD_H * NDC_SCALE;
            renderRectangle(0.0f, 0.0f, rect_w_ndc, rect_h_ndc); // Center in NDC
            
            // Render the net splitting the court in half (vertical line at x=0)
            glUseProgram(netShaderProgram);
            renderNet(world_to_ndc_x(0.0f), world_to_ndc_y(rectHalfHeight), world_to_ndc_y(-rectHalfHeight), world_to_ndc_scale(0.01f));
            
            // Render air particles (now moving, small circles)
            glUseProgram(airParticleShaderProgram);
            for (size_t i = 0; i < airParticles.size(); i++) {
                renderCircle(world_to_ndc_x(airParticles[i].x), world_to_ndc_y(airParticles[i].y), world_to_ndc_scale(airParticleRadius), 16);
            }

            // Render a circle (ball) inside the rectangle at its current position
            glUseProgram(circleShaderProgram);
            renderCircle(world_to_ndc_x(circleX), world_to_ndc_y(circleY), world_to_ndc_scale(circleRadius), 32);

            // Render the ball path as a thin red line (GL_LINE_STRIP) if enabled
            if (showPath && ballPath.size() >= 2 && pathShaderProgram != 0) {
                glUseProgram(pathShaderProgram);
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
            bool ballOutsideLeft = (circleX + circleRadius < -rectHalfWidth);
            bool ballOutsideRight = (circleX - circleRadius > rectHalfWidth);
            bool ballOutsideBottom = (circleY + circleRadius < -rectHalfHeight);
            bool ballOutsideTop = (circleY - circleRadius > rectHalfHeight);

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
    glDeleteProgram(shaderProgram);
    glDeleteProgram(circleShaderProgram);
    glDeleteProgram(netShaderProgram);
    glDeleteProgram(airParticleShaderProgram);
    if (pathShaderProgram != 0) glDeleteProgram(pathShaderProgram);
    glfwTerminate();
    
    return 0;

}

