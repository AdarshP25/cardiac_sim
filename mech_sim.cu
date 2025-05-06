#include "mech_sim.hpp"
#include <stdexcept>
#include <cmath> // For hypotf
#include <cuda_runtime.h>

// Forward Declarations
__global__ void clear_forces(float2* force, int N);
__global__ void compute_cell_forces(
    const float2* pos,          // Input: Current vertex positions
    float2* force,              // Output: Accumulated forces (atomic adds)
    const int* cell_vidx,       // Input: Cell connectivity (vertex indices)
    float ks_edge,              // Input: Stiffness of edge springs
    float rest_length_edge,     // Input: Rest length for edge springs
    int C,                      // Input: Number of cells
    const float* u);             // Input: Auxiliary field (unused here)
__global__ void position_verlet(
    float2* pos, float2* prev_pos, const float2* force,
    float dt, int N);
__global__ void calculate_velocity_pv(
    float2* vel, const float2* pos_c, const float2* pos_p,
    float dt, int N);



// CUDA error checking helper
static void check(cudaError_t e) {
    if (e != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(e));
    }
}

// Kernels

__global__ void clear_forces(float2* force, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        force[i] = make_float2(0.0f, 0.0f);
    }
}

__device__ inline void calculate_and_apply_spring_force(
    int va, int vb,
    float2 pa, float2 pb,
    float2* force,
    float ks,
    float rest_length)
{
    // Calculate the vector from A to B
    float2 vec_ab = make_float2(pb.x - pa.x, pb.y - pa.y);

    // Calculate the current length of the spring
    float current_length = hypotf(vec_ab.x, vec_ab.y);

    // Avoid division by zero
    if (current_length < 1e-6f) {
        return;
    }

    float displacement = current_length - rest_length;

    // Calculate the magnitude of the restoring force (Hooke's Law: F = -k * displacement)
    float force_magnitude = -ks * displacement;

    // normalized direction
    float2 direction = make_float2(vec_ab.x / current_length, vec_ab.y / current_length);

    // Calculate the force vector
    float2 force_vector = make_float2(force_magnitude * direction.x, force_magnitude * direction.y);

    // Apply the forces to the vertices using atomic operations
    atomicAdd(&force[vb].x, force_vector.x);
    atomicAdd(&force[vb].y, force_vector.y);
    atomicAdd(&force[va].x, -force_vector.x);
    atomicAdd(&force[va].y, -force_vector.y);
}

__global__ void compute_cell_forces(
    const float2* pos,
    float2* force,
    const int* cell_vidx,
    float ks_edge,
    float rest_length_edge,
    int C)
{
    
    int ci = blockIdx.x * blockDim.x + threadIdx.x;

    
    if (ci >= C) return;

    
    int base_idx = ci * 4;
    int v_indices[4];
    v_indices[0] = cell_vidx[base_idx + 0]; // top-left (v0)
    v_indices[1] = cell_vidx[base_idx + 1]; // top-right (v1)
    v_indices[2] = cell_vidx[base_idx + 2]; // bot-right (v2)
    v_indices[3] = cell_vidx[base_idx + 3]; // bot-left (v3)

    // Check for invalid indices
    if (v_indices[0] < 0 || v_indices[1] < 0 || v_indices[2] < 0 || v_indices[3] < 0) return;

    // Get Vertex Positions
    float2 p[4];
    p[0] = pos[v_indices[0]];
    p[1] = pos[v_indices[1]];
    p[2] = pos[v_indices[2]];
    p[3] = pos[v_indices[3]];

    // --- Calculate forces for the perimeter connections ---

    // Connection 0-1 (Top Edge)
    calculate_and_apply_spring_force(v_indices[0], v_indices[1], p[0], p[1], force, ks_edge, rest_length_edge);

    // Connection 1-2 (Right Edge)
    calculate_and_apply_spring_force(v_indices[1], v_indices[2], p[1], p[2], force, ks_edge, rest_length_edge);

    // Connection 2-3 (Bot Edge)
    calculate_and_apply_spring_force(v_indices[2], v_indices[3], p[2], p[3], force, ks_edge, rest_length_edge);

    // Connection 3-0 (Left Edge)
    calculate_and_apply_spring_force(v_indices[3], v_indices[0], p[3], p[0], force, ks_edge, rest_length_edge);
}

__global__ void compute_active_stress_force(
    const float2* pos,          // Input: Current vertex positions
    float2* force,              // Output: Accumulated forces (atomic adds)
    const int* cell_vidx,       // Input: Cell connectivity (vertex indices)
    const float* u,             // Input: Activation field
    float T0,                   // Input: Max tension
    float beta,                 // Input: Activation steepness
    float ua,                   // Input: Activation threshold
    float active_force_scaling, // Input: Scaling factor from tension to force
    int C)                      // Input: Number of cells
{
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= C) return;

    int base_idx = ci * 4;
    int v_indices[4];
    float u_vals[4];
    float2 p[4];

    // Gather vertex indices, positions, and u values for the cell
    float u_sum = 0.0f;
    for (int k = 0; k < 4; ++k) {
        v_indices[k] = cell_vidx[base_idx + k];
        if (v_indices[k] < 0) return; // Invalid index check
        p[k] = pos[v_indices[k]];
        u_vals[k] = u[v_indices[k]];
        u_sum += u_vals[k];
    }

    // Calculate average u for the cell
    float u_avg = u_sum * 0.25f;

    // Calculate active tension using tanh based on average u
    // tanhf is the float version
    float Ta = T0 * tanhf(beta * (u_avg - ua));

    // Optional: Add a small threshold to avoid tiny forces if needed
    // if (fabsf(Ta) < 1e-4f) return;

    // Calculate cell centroid
    float2 centroid = make_float2(
        (p[0].x + p[1].x + p[2].x + p[3].x) * 0.25f,
        (p[0].y + p[1].y + p[2].y + p[3].y) * 0.25f
    );

    // Apply force to each vertex based on Ta and centroid
    for (int k = 0; k < 4; ++k) {
        int vk = v_indices[k];
        float2 pk = p[k];

        // Vector from centroid to vertex
        float2 vec_centroid_to_p = make_float2(pk.x - centroid.x, pk.y - centroid.y);
        float len = hypotf(vec_centroid_to_p.x, vec_centroid_to_p.y);

        if (len > 1e-6f) { // Avoid division by zero
            // Normalized direction from centroid to vertex
            float2 dir = make_float2(vec_centroid_to_p.x / len, vec_centroid_to_p.y / len);

            // Force magnitude: Proportional to Ta.
            // Negative sign means contraction (Ta > 0) pulls towards centroid (opposite to dir).
            float force_magnitude = -active_force_scaling * Ta;

            // Force vector
            float2 force_k = make_float2(force_magnitude * dir.x, force_magnitude * dir.y);

            // Atomically add force to the vertex
            atomicAdd(&force[vk].x, force_k.x);
            atomicAdd(&force[vk].y, force_k.y);
        }
    }
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
    const float2* pos_c,
    const float2* pos_p,
    float dt,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (dt > 0.0f) {
             vel[i].x = (pos_c[i].x - pos_p[i].x) / dt;
             vel[i].y = (pos_c[i].y - pos_p[i].y) / dt;
        } else {
             vel[i] = make_float2(0.0f, 0.0f);
        }
    }
}

// Class Methods

MechSim::MechSim(int nx_, int ny_, float dx, float dy)
    : nx(nx_), ny(ny_) {
    N = nx * ny;
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
            h_pos[idx] = make_float2(static_cast<float>(i) * dx, static_cast<float>(j) * dy);
        }
    }
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
}

MechSim::~MechSim() {
    if (d_pos_c) cudaFree(d_pos_c);
    if (d_pos_p) cudaFree(d_pos_p);
    if (d_vel) cudaFree(d_vel);
    if (d_force) cudaFree(d_force);
    if (d_cell_vidx) cudaFree(d_cell_vidx);
}


void MechSim::step(float dt, float ks_edge, float ks_radial, float* u,
                   float T0, float beta, float ua, float active_force_scaling) {
    if (N <= 0) return;

    int block = 256;
    int gridV = (N + block - 1) / block;
    int gridC = (C > 0) ? (C + block - 1) / block : 0;

    clear_forces<<<gridV, block>>>(d_force, N);
    check(cudaGetLastError());


    if (gridC > 0) {
        compute_cell_forces<<<gridC, block>>>(
            d_pos_c, d_force,
            d_cell_vidx,
            ks_edge, MechSim::rest_length_edge,
            C
        );
        check(cudaGetLastError());
    }

    compute_active_stress_force<<<gridC, block>>>(
            d_pos_c, d_force, d_cell_vidx, u,
            T0, beta, ua, active_force_scaling, C
        );
    check(cudaGetLastError());

    position_verlet<<<gridV, block>>>(d_pos_c, d_pos_p, d_force, dt, N);
    check(cudaGetLastError());

    // --- Apply Constraints (e.g., boundary conditions) ---

    calculate_velocity_pv<<<gridV, block>>>(d_vel, d_pos_c, d_pos_p, dt, N);
    check(cudaGetLastError());

    check(cudaDeviceSynchronize());
}

// Download CURRENT positions to host
void MechSim::download_positions(std::vector<float2>& h_pos) {
    if (N <= 0) {
        h_pos.clear();
        return;
    }
    h_pos.resize(N);
    // Download from d_pos_c, which holds the most recent positions
    check(cudaMemcpy(h_pos.data(), d_pos_c, N*sizeof(float2), cudaMemcpyDeviceToHost));
}