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
std::vector<std::vector<Location>> grid;

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
        ////////////////////////////////////////////////////////////////////////////////////////
        // Start of timestep
        ////////////////////////////////////////////////////////////////////////////////////////

        // Calculate all collisions in the current timestep

        // Calculate the impulse on the ball from all collisions

        // Modify ball velocity and spin based on impulse

        currentTime += timestep_size / 1000.0; // Convert ms to s

        ////////////////////////////////////////////////////////////////////////////////////////
        // End of timestep
        ////////////////////////////////////////////////////////////////////////////////////////

        // Update ball position based on velocity
    }

    return 0;
}