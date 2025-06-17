#pragma once

#include <cuda_runtime.h>
#include <vector>

class MechSim {
public:
    MechSim(int nx_, int ny_, float* fiber_angles_, float damping_);
    ~MechSim();

    void step(float dt, float ks_edge, float ks_radial, float ks_boundary, float* T_a);

    void download_positions(std::vector<float2>& h_pos);

    MechSim(const MechSim&) = delete;
    MechSim& operator=(const MechSim&) = delete;

    static constexpr float rest_length_edge = 1.0f; // Rest length for axial springs
    static constexpr float rest_length_diagonal = 1.41421356237f; // Rest length for diagonal springs

    int nx, ny;      // grid dimensions
    int N;           // total vertices = nx * ny
    int C;           // total cells    = (nx-1) * (ny-1)
    float* fiber_angles; // [N] fiber angles for each vertex
    float damping;   // damping factor

    // Device arrays
    float2* d_pos_c;          // [N] current positions
    float2* d_pos_p;          // [N] previous positions
    float2* d_vel;          // [N] current velocities
    float2* d_force;        // [N] accumulated forces
    float* d_fiber_angles; // [N] fiber angles for each vertex
    
    int* d_boundaryIdx;
    float2* d_boundaryPositions;
    int numBoundary; // Number of boundary vertices
    

    int*    d_cell_vidx;    // [C*4] vertex indices for each cell
};
