#pragma once

#include <cuda_runtime.h>
#include <vector>

class MechSim {
public:
    MechSim(int nx_, int ny_, float dx = 1.0f, float dy = 1.0f);
    ~MechSim();

    void step(float dt, float ks_edge, float ks_radial, float* u,
              float T0, float beta, float ua, float active_force_scaling);

    void download_positions(std::vector<float2>& h_pos);

    MechSim(const MechSim&) = delete;
    MechSim& operator=(const MechSim&) = delete;

    static constexpr float rest_length_edge = 1.0f; // Rest length for axial springs
    static constexpr float rest_length_diagonal = 1.41421356237f; // Rest length for diagonal springs

    int nx, ny;      // grid dimensions
    int N;           // total vertices = nx * ny
    int C;           // total cells    = (nx-1) * (ny-1)

    // Device arrays
    float2* d_pos_c;          // [N] current positions
    float2* d_pos_p;          // [N] previous positions
    float2* d_vel;          // [N] current velocities
    float2* d_force;        // [N] accumulated forces

    int*    d_cell_vidx;    // [C*4] vertex indices for each cell
};
