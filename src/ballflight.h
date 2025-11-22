#include <cstdint>

struct Vec2 {
    float x;
    float y;
};

// All global simulation variables
struct SimulationParameters {
    double timestepSize; // seconds
    double totalTime; // seconds
    int mode;
    int airParcelCount; // millions
};

// All object types in the simulation
typedef enum {
    VOID,
    AIR_PARCEL,
    BALL
} ObjectType;

// Information about an object in the simulation
struct Object {
    ObjectType type;
    int id; // unique identifier
    Vec2 position; // coordinates on the grid for the center of the object
    Vec2 velocity; // velocity components
    
};

// No additional properties for air parcels at this time
struct AirParcel : Object {
};

// Additional properties for the ball
struct Ball : Object {
    double radius; // cm, is this needed here?
    Vec2 spin; // spin components
};

// Information about a specific location in the simulation grid
struct Location {
    //int x; int y; // coordinates on the grid
    ObjectType type;
    int objectID; // unique identifier for the object
};

// Different types of collisions that can occur
enum CollisionType {
    GROUND,
    AIR_AIR,
    AIR_BALL,
};

// Information about a collision event
// Ensure uniqueness of collisions by always having objectID1 < objectID2
// Encode the pair key as a 64-bit integer: (objectID1 << 32) | objectID2
struct Collision {
    CollisionType type;
    int objectID1;
    int objectID2; // Not needed for ground collisions
    uint64_t pair_key; // unique key for the object pair
};
