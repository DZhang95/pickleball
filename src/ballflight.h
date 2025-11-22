

// All global simulation variables
struct World {
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
    int x; int y; // coordinates on the grid for the center of the object
    int velocityX; int velocityY; // velocity components
    
};

// No additional properties for air parcels at this time
struct AirParcel : Object {
};

// Additional properties for the ball
struct Ball : Object {
    double radius; // cm, is this needed here?
    int spinX; int spinY; // spin components
};

// Information about a specific location in the simulation grid
struct Location {
    //int x; int y; // coordinates on the grid
    ObjectType type;
    int objectID; // unique identifier for the object
};