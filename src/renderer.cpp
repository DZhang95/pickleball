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

// Air particle structure (designed for future upgrade to realistic physics)
struct AirParticle {
    float x, y;           // Position
    float velX, velY;      // Velocity
    float mass;            // Mass (for future realistic physics)
    
    AirParticle(float px, float py, float vx, float vy, float m) 
        : x(px), y(py), velX(vx), velY(vy), mass(m) {}
};

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
#ifndef __APPLE__
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwTerminate();
        return -1;
    }
#endif    
    
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
    
    // Generate air particles (now with velocities for collision response)
    const int numAirParticles = 3000;
    
    // Physics constants
    const float BALL_MASS = 0.026f;  // kg (26g)
    const float AIR_PARCEL_MASS = 0.0001f;  // kg (0.1g) - represents many air molecules
    const float BALL_RADIUS = 0.185f;
    const float BALL_MOMENT_OF_INERTIA = (2.0f / 5.0f) * BALL_MASS * BALL_RADIUS * BALL_RADIUS;
    
    float airParticleRadius = 0.005f;
    
    const float IMPULSE_SCALE_FACTOR = 0.01f;  // Scale down impulses by 100x
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
    float circleX = 0.0f;
    float circleY = 0.0f;
    // circleRadius is now set above from BALL_RADIUS
    float circleVelX = 10.0f;  // Velocity in x direction
    float circleVelY = 10.0f;   // Velocity in y direction
    float circleSpin = 0.5f;  // Angular velocity (spin) - scalar in 2D
    
    // Main render loop
    const float timestep = 0.001f;  // 0.001s time step (from physics.txt)
    while (!glfwWindowShouldClose(window)) {
        // Update ball position based on velocity
        circleX += circleVelX * timestep;
        circleY += circleVelY * timestep;
        
        // Update air particle positions
        for (size_t i = 0; i < airParticles.size(); i++) {
            airParticles[i].x += airParticles[i].velX * timestep;
            airParticles[i].y += airParticles[i].velY * timestep;
        }
        
        // Check for collisions between air particles (AIR-AIR collisions)
        // Using impulse-based physics: calculate relative velocity and impulse
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
        
        // Check for collisions between ball and air particles
        // Using the physics model: calculate surface velocity, relative velocity, and impulse
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
                
                // Calculate impulse: J = 2 * m * dot(v_rel, n) * n
                float v_rel_dot_n = relVelX * nx + relVelY * ny;
                float impulseX = 2.0f * AIR_PARCEL_MASS * v_rel_dot_n * nx;
                float impulseY = 2.0f * AIR_PARCEL_MASS * v_rel_dot_n * ny;
                
                // Scale down impulse: each air parcel represents many molecules
                // In reality, there are billions of tiny collisions that average out
                // Since we have fewer, larger parcels, we scale down the effect
                impulseX *= IMPULSE_SCALE_FACTOR;
                impulseY *= IMPULSE_SCALE_FACTOR;
                
                // Update ball linear velocity: v_ball_new = v_ball + (J / M)
                circleVelX += impulseX / ballMass;
                circleVelY += impulseY / ballMass;
                
                // Update ball angular velocity: ω_new = ω + (cross(r, J) / I)
                // In 2D: cross(r, J) = r.x * J.y - r.y * J.x (scalar)
                // Note: spin change already uses the scaled impulse, so it's automatically scaled
                float r_cross_J = rx * impulseY - ry * impulseX;
                float spin_change = r_cross_J / BALL_MOMENT_OF_INERTIA;
                circleSpin += spin_change;
                
                // Update air particle velocity (for still air, this would be minimal)
                // For now, we'll apply the impulse to the particle as well
                particle.velX -= impulseX / particle.mass;
                particle.velY -= impulseY / particle.mass;
            }
        }
        
        // Check for collisions with rectangle boundaries and bounce (ball)
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
        
        // Check for collisions with rectangle boundaries (air particles)
        for (size_t i = 0; i < airParticles.size(); i++) {
            AirParticle& particle = airParticles[i];
            
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
        
        // Render air particles (now moving, small circles)
        glUseProgram(airParticleShaderProgram);
        for (size_t i = 0; i < airParticles.size(); i++) {
            renderCircle(airParticles[i].x, airParticles[i].y, airParticleRadius, 16);
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

