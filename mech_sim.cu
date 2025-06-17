#include "mech_sim.hpp"
#include <stdexcept>
#include <cmath> // For hypotf
#include <cuda_runtime.h>

// CUDA error checking helper
static void check(cudaError_t e) {
    if (e != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(e));
    }
}

MechSim::MechSim(int nx_, int ny_, float* fiber_angles_, float damping_)
    : nx(nx_), ny(ny_), damping(damping_) {
    N = nx * ny;
    fiber_angles = fiber_angles_;
    C = (nx - 1) * (ny - 1);
    if (N <= 0 || C < 0) {
        throw std::runtime_error("Invalid grid dimensions");
    }
    size_t szN    = N * sizeof(float2);
    size_t szC4i  = C > 0 ? C * 4 * sizeof(int)   : 0;
    size_t szC    = C > 0 ? C * sizeof(float)     : 0;
    size_t szC4f  = C > 0 ? C * 4 * sizeof(float) : 0;

    // Allocate device arrays
    check(cudaMalloc(&d_pos_c,     szN));
    check(cudaMalloc(&d_pos_p,     szN));
    check(cudaMalloc(&d_vel,       szN));
    check(cudaMalloc(&d_force,     szN));
    if (C > 0) {
        check(cudaMalloc(&d_cell_vidx, szC4i));
    } else {
        d_cell_vidx = nullptr;
    }


    // Initialize positions on host
    std::vector<float2> h_pos(N);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = j * nx + i;
            h_pos[idx] = make_float2(static_cast<float>(i) * rest_length_edge, static_cast<float>(j) * rest_length_edge);
        }
    }
    h_pos[0] = make_float2(0.5f, 0.5f);
    // Copy initial positions to both current and previous device arrays
    check(cudaMemcpy(d_pos_c, h_pos.data(), szN, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_pos_p, h_pos.data(), szN, cudaMemcpyHostToDevice)); // Initialize prev pos too
    check(cudaMemset(d_vel, 0, szN)); // Initialize velocities to zero
    check(cudaMemset(d_force, 0, szN)); // Initialize forces to zero


    // Build connectivity and rest lengths on host if there are cells
    if (C > 0) {
        std::vector<int>   h_idx(C*4);
        int ci = 0;
        for (int i = 0; i < nx-1; ++i) {
            for (int j = 0; j < ny-1; ++j, ++ci) {
                int i0 = i*nx + j;
                int i1 = i*nx + j+1;
                int i2 = (i+1)*nx + j+1;
                int i3 = (i+1)*nx + j;

                // Check bounds (optional safety)
                // if (i0 >= N || i1 >= N || i2 >= N || i3 >= N) continue;

                h_idx[ci*4 + 0] = i0;
                h_idx[ci*4 + 1] = i1;
                h_idx[ci*4 + 2] = i2;
                h_idx[ci*4 + 3] = i3;
            }
        }
        check(cudaMemcpy(d_cell_vidx, h_idx.data(), szC4i, cudaMemcpyHostToDevice));
    }

    std::vector<int> boundaryIdx;
    std::vector<float2> boundaryPositions;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                int idx = j * nx + i;
                boundaryIdx.push_back(idx);
                boundaryPositions.push_back(make_float2(static_cast<float>(i) * rest_length_edge, static_cast<float>(j) * rest_length_edge));
            }
        }
    }

    numBoundary = boundaryIdx.size();
    cudaMalloc(&d_boundaryIdx, numBoundary * sizeof(int));
    cudaMalloc(&d_boundaryPositions, numBoundary * 2 * sizeof(float2));
    cudaMemcpy(d_boundaryIdx, boundaryIdx.data(), numBoundary * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_boundaryPositions, boundaryPositions.data(), numBoundary * 2 * sizeof(float), cudaMemcpyHostToDevice);
}

MechSim::~MechSim() {
    if (d_pos_c) cudaFree(d_pos_c);
    if (d_pos_p) cudaFree(d_pos_p);
    if (d_vel) cudaFree(d_vel);
    if (d_force) cudaFree(d_force);
    if (d_cell_vidx) cudaFree(d_cell_vidx);
}


__global__ void position_verlet(
    float2* pos,
    float2* prev_pos,
    const float2* force,
    float dt,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float2 current_pos = pos[i];

        float2 accel = force[i];

        float2 next_pos;
        float dt_sq = dt * dt;

        next_pos.x = 2.0f * current_pos.x - prev_pos[i].x + accel.x * dt_sq;
        next_pos.y = 2.0f * current_pos.y - prev_pos[i].y + accel.y * dt_sq;

        prev_pos[i] = current_pos;

        pos[i] = next_pos;
    }
}

__global__ void calculate_velocity_pv(
        float2* vel,
        const float2* pos_np1,   // x_{n+1}
        const float2* pos_nm1,   // x_{n-1}
        float dt, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float inv2dt = 0.5f / dt;
        vel[i].x = (pos_np1[i].x - pos_nm1[i].x) * inv2dt;
        vel[i].y = (pos_np1[i].y - pos_nm1[i].y) * inv2dt;
    }
}

__global__ void clear_forces(float2* force, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        force[i] = make_float2(0.0f, 0.0f);
    }
}

__global__ void calculate_structural_forces(
        const float2* __restrict__ pos, const float2* __restrict__ vel,
        float2* __restrict__ force,
        int nx, int ny, float ks, float rest, float damping)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;

    int idx = iy * nx + ix;
    float2 pa = pos[idx];
    float2 f  = make_float2(0.f, 0.f);

    const int nbh[4][2] = { {-1,0},{+1,0},{0,-1},{0,+1} };

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int jx = ix + nbh[k][0];
        int jy = iy + nbh[k][1];
        if (jx < 0 || jx >= nx || jy < 0 || jy >= ny) continue;

        float2 pb  = pos[jy * nx + jx];
        float2 d   = make_float2(pb.x - pa.x, pb.y - pa.y);
        float  len = hypotf(d.x, d.y);
        float  mag = ks * (len - rest);
        f.x += mag * d.x / len;
        f.y += mag * d.y / len;
    }
    force[idx].x += f.x;
    force[idx].y += f.y;

    // Damping force
    float2 damping_force = make_float2(-damping * vel[idx].x, -damping * vel[idx].y);
    force[idx].x += damping_force.x;
    force[idx].y += damping_force.y;
}

__global__ void calculate_boundry_forces(const float2* __restrict__ pos,
                                         float2* __restrict__ force,
                                         const int* __restrict__ boundaryIdx,
                                         const float2* __restrict__ boundaryPositions, float ks_boundary,
                                         int numBoundary)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numBoundary) {
        int idx = boundaryIdx[i];
        float2 p = pos[idx];
        float2 f = make_float2(0.0f, 0.0f);

        // Calculate force based on boundary position
        f.x = ks_boundary * (boundaryPositions[i].x - p.x);
        f.y = ks_boundary * (boundaryPositions[i].y - p.y);

        force[idx].x += f.x;
        force[idx].y += f.y;
    }
}

__global__ void calculate_orthogonal_forces()
{
    //Passive spring

    //Active Spring
}

void MechSim::step(float dt, float ks_edge, float ks_radial, float ks_boundary, float* T_a) 
{
    if (dt <= 0.0f) {
        throw std::runtime_error("Time step must be positive");
    }

    // Clear forces
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    clear_forces<<<numBlocks, blockSize>>>(d_force, N);
    check(cudaGetLastError());

    // Calculate structural forces
    dim3 gridDim((nx + 15) / 16, (ny + 15) / 16);
    dim3 blockDim(16, 16);
    calculate_structural_forces<<<gridDim, blockDim>>>(d_pos_c, d_vel, d_force, nx, ny, ks_edge, MechSim::rest_length_edge, damping);
    check(cudaGetLastError());

    // Calculate boundary forces
    calculate_boundry_forces<<<(numBoundary + blockSize - 1) / blockSize, blockSize>>>(d_pos_c, d_force, d_boundaryIdx, d_boundaryPositions, ks_boundary, numBoundary);


    // Position Verlet integration
    position_verlet<<<(N + blockSize - 1) / blockSize, blockSize>>>(d_pos_c, d_pos_p, d_force, dt, N);
    check(cudaGetLastError());

    // Calculate velocities
    calculate_velocity_pv<<<(N + blockSize - 1) / blockSize, blockSize>>>(d_vel, d_pos_c, d_pos_p, dt, N);
    check(cudaGetLastError());
}