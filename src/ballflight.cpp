#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>

#include "ballflight.h"

//float ballVelocityX;
//float ballVelocityY;
//float ballVelocityZ; // For 3d
Vec2 ballVelocity;
//float ballSpinX;
//float ballSpinY;
//float ballSpinZ; // For 3d
Vec2 ballSpin;
double currentTime; // seconds
// Current world state
std::vector<std::vector<Location>> grid;
// Historical world states for output
// Stores just the ObjectType at each grid location for each timestep
// THIS IS MASSIVE
std::vector<std::vector<std::vector<ObjectType>>> gridHistory;

// helper to produce canonical 64-bit key
static inline uint64_t make_pair_key(int a, int b) {
    assert(a >= 0 && b >= 0); // ensure non-negative IDs, or add offset
    uint32_t ua = static_cast<uint32_t>(a);
    uint32_t ub = static_cast<uint32_t>(b);
    if (ua > ub) std::swap(ua, ub);
    return (static_cast<uint64_t>(ua) << 32) | static_cast<uint64_t>(ub);
};

// Initialize the world
// Create all objects (air parcels and ball) and place them in the grid
// Each object must have a unique ID
// Objects can be looked up from the object array using their ID as the index
void initialize_world(double timestep_size, double total_time, double ball_velocity, double ball_spin, int mode, int air_parcel_count) {
    // Implementation goes here
}

// Detect all collisions in the current timestep
std::vector<Collision> detect_collisions() {
    std::vector<Collision> detected_collisions;
    // TODO: implementation
    return detected_collisions;
}

// Calculate the net impulse on the ball given a single collision
Vec2 calculate_impulse(const Collision& collision) {
    Vec2 impulse = {0.0f, 0.0f};
    // TODO: implementation
    return impulse;
}

// Calculate the total impulse on the ball from all collisions
Vec2 calculate_total_impulse(const std::vector<Collision>& collisions) {
    Vec2 total_impulse = {0.0f, 0.0f};
    for (const auto& collision : collisions) {
        Vec2 impulse = calculate_impulse(collision);
        total_impulse.x += impulse.x;
        total_impulse.y += impulse.y;
    }
    return total_impulse;
}

// Update ball velocity based on impulse
// Update ball spin based on impulse
// Update ball position based on new velocity and spin
void update_ball(Vec2 impulse) {
  // TODO: implementation
}

// Save the current grid state to history
// Store only ObjectType at each grid location to save space
void save_grid_state() {
  // TODO: implementation
}

// Print the grid history to a file
void print_grid_history(const std::string& filename) {
  // TODO: implementation
}

/// Entry point for the ball flight simulation
int main(int argc, char** argv)
{

    // Default parameters
    double timestep_size = 1; // milliseconds
    double total_time = 3; //seconds
    double ball_velocity = 0; // m/s
    double ball_spin = 0; // rpm
    int mode = 50;
    int air_parcel_count = 3; // millions

    int opt;
    while ((opt = getopt(argc, argv, "t:T:v:w:a:p:")) != -1) {
      switch (opt) {
        case 't': timestep_size = atof(optarg); break;
        case 'T': total_time = atof(optarg); break;
        case 'v': ball_velocity = atof(optarg); break;
        case 'w': ball_spin = atof(optarg); break;
        case 'a': mode = atoi(optarg); break;
        case 'p': air_parcel_count = atoi(optarg); break;
        default:
          std::cerr << "Usage: " << argv[0] << " [-t timestep_size] [-T total_time] [-v ball_velocity] [-w ball_spin] [-a mode] [-p air_parcel_count]\n";
          return 1;
      }
    }    

    // Initialization
    initialize_world(timestep_size, total_time, ball_velocity, ball_spin, mode, air_parcel_count);
    currentTime = 0.0;

    while(currentTime < total_time) {
        // Calculate all collisions in the current timestep
        std::vector<Collision> collisions = detect_collisions();

        // Calculate the impulse on the ball from all collisions
        Vec2 total_impulse = calculate_total_impulse(collisions);

        // Modify ball velocity and spin based on impulse
        // Update ball position based on velocity
        update_ball(total_impulse);

        // Save grid state
        save_grid_state();
        
        currentTime += timestep_size / 1000.0; // Convert ms to s
    }

    // Print grid history to file
    print_grid_history("grid_history.txt");

    return 0;
}