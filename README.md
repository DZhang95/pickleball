# pickleball, under construction

ballflight usage
-t: timestep size in units of ms. Default 1ms
-T: total time in ms, default 3s
-v: Initial ball velocity - (int, int, int)
-w: Initial ball angular velocity (spin)
-a: Approach used for simulation
    50 - 50% approach
    75 - 75% approach
    100 - 100% approach
    125 - 125% approach
-p: Number of millions of air parcels, default 3


[-t timestep_size] [-T total_time] -v ball_velocity -w ball_spin -a mode -p [air_parcel_count]