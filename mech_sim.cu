#include "mech_sim.hpp"
#include <stdexcept>
#include <math.h>
#include <math_constants.h>

#include <cuda_runtime.h>

// CUDA error checking helper
static void check(cudaError_t e)
{
    if (e != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(e));
    }
}

MechSim::MechSim(int nx_, int ny_, float2 *fiber_angles_, float damping_)
    : nx(nx_), ny(ny_), damping(damping_)
{
    N = nx * ny;
    fiber_angles = fiber_angles_;
    C = (nx - 1) * (ny - 1);
    if (N <= 0 || C < 0)
    {
        throw std::runtime_error("Invalid grid dimensions");
    }
    size_t szN = N * sizeof(float2);
    size_t szC4i = C > 0 ? C * 4 * sizeof(int) : 0;
    size_t szC = C > 0 ? C * sizeof(float) : 0;
    size_t szC4f = C > 0 ? C * 4 * sizeof(float) : 0;
    size_t szC4f2 = C > 0 ? C * 4 * sizeof(float2) : 0;
    size_t szC4b = C > 0 ? C * 4 * sizeof(bool) : 0;

    // Allocate device arrays
    check(cudaMalloc(&d_pos_c, szN));
    check(cudaMalloc(&d_pos_p, szN));
    check(cudaMalloc(&d_vel, szN));
    check(cudaMalloc(&d_force, szN));
    if (C > 0)
    {
        check(cudaMalloc(&d_cell_vidx, szC4i));
        check(cudaMalloc(&d_fiber_angles, szC4f2));
        check(cudaMalloc(&d_orthogonal_rest_lengths, szC4f));
        check(cudaMalloc(&d_active_spring_is_horizontal, szC4b));
    }
    else
    {
        d_cell_vidx = nullptr;
        d_fiber_angles = nullptr;
        d_orthogonal_rest_lengths = nullptr;
        d_active_spring_is_horizontal = nullptr;
    }

    // Initialize positions on host
    std::vector<float2> h_pos(N);
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            int idx = j * nx + i;
            h_pos[idx] = make_float2(static_cast<float>(i) * rest_length_edge, static_cast<float>(j) * rest_length_edge);
        }
    }
    //h_pos[0] = make_float2(1.0f, 0.5f);
    // Copy initial positions to both current and previous device arrays
    check(cudaMemcpy(d_pos_c, h_pos.data(), szN, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_pos_p, h_pos.data(), szN, cudaMemcpyHostToDevice)); // Initialize prev pos too
    check(cudaMemset(d_vel, 0, szN));                                      // Initialize velocities to zero
    check(cudaMemset(d_force, 0, szN));                                    // Initialize forces to zero

    // Build connectivity and rest lengths on host if there are cells
    if (C > 0)
    {
        std::vector<int> h_idx(C * 4);
        std::vector<float4> h_rest_lengths(C);
        std::vector<char> h_active_spring_is_horizontal(C, 0);
        std::vector<float> h_orthogonal_rest_lengths(C);
        int ci = 0;
        for (int i = 0; i < nx - 1; ++i)
        {
            for (int j = 0; j < ny - 1; ++j, ++ci)
            {
                int i0 = i * nx + j;
                int i1 = i * nx + j + 1;
                int i2 = (i + 1) * nx + j + 1;
                int i3 = (i + 1) * nx + j;

                // Check bounds (optional safety)
                // if (i0 >= N || i1 >= N || i2 >= N || i3 >= N) continue;

                h_idx[ci * 4 + 0] = i0;
                h_idx[ci * 4 + 1] = i1;
                h_idx[ci * 4 + 2] = i2;
                h_idx[ci * 4 + 3] = i3;

                // Calculate rest lengths for each orthogonal spring
                float2 p0 = h_pos[i0];
                float2 p1 = h_pos[i1];
                float2 p2 = h_pos[i2];
                float2 p3 = h_pos[i3];

                float2 barycenter = make_float2(
                    (p0.x + p1.x + p2.x + p3.x) / 4.0f,
                    (p0.y + p1.y + p2.y + p3.y) / 4.0f);

                float fiber_angle = atan2(fiber_angles[ci].y, fiber_angles[ci].x);

                if (fiber_angle <= M_PI / 4 || fiber_angle >= 3 * M_PI / 4)
                {
                    h_active_spring_is_horizontal[ci] = true;
                }

                if (fiber_angle >= M_PI / 2)
                {
                    fiber_angle -= M_PI / 2;
                }

                h_orthogonal_rest_lengths[ci] = fabs((rest_length_edge / 2) / (cos(fiber_angle)));
            }
        }
        check(cudaMemcpy(d_cell_vidx, h_idx.data(), szC4i, cudaMemcpyHostToDevice));
        check(cudaMemcpy(d_fiber_angles, fiber_angles, szC4f2, cudaMemcpyHostToDevice));
        check(cudaMemcpy(d_active_spring_is_horizontal, h_active_spring_is_horizontal.data(), szC4b, cudaMemcpyHostToDevice));
        check(cudaMemcpy(d_orthogonal_rest_lengths, h_orthogonal_rest_lengths.data(), szC4f, cudaMemcpyHostToDevice));
    }

    std::vector<int> boundaryIdx;
    std::vector<float2> boundaryPositions;

    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
            {
                int idx = j * nx + i;
                boundaryIdx.push_back(idx);
                boundaryPositions.push_back(make_float2(static_cast<float>(i) * rest_length_edge, static_cast<float>(j) * rest_length_edge));
            }
        }
    }

    numBoundary = boundaryIdx.size();
    cudaMalloc(&d_boundaryIdx, numBoundary * sizeof(int));
    cudaMalloc(&d_boundaryPositions, numBoundary * 2 * sizeof(float));
    cudaMemcpy(d_boundaryIdx, boundaryIdx.data(), numBoundary * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_boundaryPositions, boundaryPositions.data(), numBoundary * 2 * sizeof(float), cudaMemcpyHostToDevice);
}

MechSim::~MechSim()
{
    if (d_pos_c)
        cudaFree(d_pos_c);
    if (d_pos_p)
        cudaFree(d_pos_p);
    if (d_vel)
        cudaFree(d_vel);
    if (d_force)
        cudaFree(d_force);
    if (d_cell_vidx)
        cudaFree(d_cell_vidx);
}

__global__ void position_verlet(
    float2 *pos,
    float2 *prev_pos,
    const float2 *force,
    float dt,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
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
    float2 *vel,
    const float2 *pos_np1, // x_{n+1}
    const float2 *pos_nm1, // x_{n-1}
    float dt, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float inv2dt = 0.5f / dt;
        vel[i].x = (pos_np1[i].x - pos_nm1[i].x) * inv2dt;
        vel[i].y = (pos_np1[i].y - pos_nm1[i].y) * inv2dt;
    }
}

__global__ void clear_forces(float2 *force, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        force[i] = make_float2(0.0f, 0.0f);
    }
}

__global__ void calculate_structural_forces(
    const float2 *__restrict__ pos, const float2 *__restrict__ vel,
    float2 *__restrict__ force,
    int nx, int ny, float ks, float rest, float damping)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;

    int idx = iy * nx + ix;
    float2 pa = pos[idx];
    float2 f = make_float2(0.f, 0.f);

    const int nbh[4][2] = {{-1, 0}, {+1, 0}, {0, -1}, {0, +1}};

#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        int jx = ix + nbh[k][0];
        int jy = iy + nbh[k][1];
        if (jx < 0 || jx >= nx || jy < 0 || jy >= ny)
            continue;

        float2 pb = pos[jy * nx + jx];
        float2 d = make_float2(pb.x - pa.x, pb.y - pa.y);
        float len = sqrtf(d.x * d.x + d.y * d.y);
        float mag = ks * (len - rest);
        f.x += mag * d.x / len;
        f.y += mag * d.y / len;
    }
    force[idx].x += f.x;
    force[idx].y += f.y;

    // Damping force
    float2 damping_force = make_float2(-damping * vel[idx].x, -damping * vel[idx].y);
    atomicAdd(&force[idx].x, damping_force.x);
    atomicAdd(&force[idx].y, damping_force.y);
}

__global__ void calculate_boundry_forces(const float2 *__restrict__ pos,
                                         float2 *__restrict__ force,
                                         const int *__restrict__ boundaryIdx,
                                         const float2 *__restrict__ boundaryPositions, float ks_boundary,
                                         int numBoundary)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numBoundary)
    {
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

// 2‑D perpendicular dot (scalar cross product)
__device__ __forceinline__ float perp_dot(const float2 &a, const float2 &b)
{
    return a.x * b.y - a.y * b.x;
}

__device__ __forceinline__ bool edge_intersects(const float2 v0,
                                                const float2 v1,
                                                const float2 barycenter,
                                                const float2 fiber_angle,
                                                float2 &intersection)
{
    // Edge and ray directions
    float2 de = make_float2(v1.x - v0.x, v1.y - v0.y); // edge vector
    float2 df = fiber_angle;                           // ray vector (unit)

    // Solve  (v0 + t·de = c + s·df)
    float2 cv0 = make_float2(barycenter.x - v0.x,
                             barycenter.y - v0.y);

    float den = perp_dot(de, df); // det |de  -df|
    if (fabsf(den) < 1e-12f)
        return false; // parallel → no hit

    float t = perp_dot(cv0, df) / den; // position on edge
    if (t < 0.0f || t > 1.0f)
        return false; // outside segment

    float s = perp_dot(cv0, de) / den; // position on ray
    if (s < 0.0f)
        return false; // behind barycenter

    // Intersection point q = v0 + t·de
    intersection.x = v0.x + t * de.x;
    intersection.y = v0.y + t * de.y;
    return true;
}

__global__ void calculate_orthogonal_forces(const float2 *__restrict__ pos,
                                            const float2 *__restrict__ fiber_angles,
                                            const float *T_a,
                                            const float *orthogonal_rest_lengths,
                                            const int *cell_vidx,
                                            const bool *active_spring_is_horizontal,
                                            float2 *force, int C, float ks_radial, float c_f)
{
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= C)
        return;

    // Get cell vertex indices
    int i0 = cell_vidx[ci * 4 + 0];
    int i1 = cell_vidx[ci * 4 + 1];
    int i2 = cell_vidx[ci * 4 + 2];
    int i3 = cell_vidx[ci * 4 + 3];

    // Get positions of the vertices
    float2 v0 = pos[i0];
    float2 v1 = pos[i1];
    float2 v2 = pos[i2];
    float2 v3 = pos[i3];

    // Barycenter
    float2 barycenter = make_float2(
        (v0.x + v1.x + v2.x + v3.x) / 4.0f,
        (v0.y + v1.y + v2.y + v3.y) / 4.0f);

    // Fiber angle
    float2 fiber_angle = fiber_angles[ci];

    // Check if the active spring is horizontal
    bool is_horizontal = active_spring_is_horizontal[ci];

    // Calculate intersection point with edge
    // vertices of the current quad      (counter‑clockwise)
    const int   vidx[4] = { i0, i1, i2, i3 };
    const float2  v[4]  = { v0, v1, v2, v3 };

    float2 intersection;

    // pre‑compute once
    float2 dir_ortho = make_float2(-fiber_angle.y,  fiber_angle.x);
    float  rest_a = orthogonal_rest_lengths[ci];                       // passive
    float  rest_b = rest_a / (1.0f + c_f * T_a[ci]);                   // active

    #pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        int a = k;
        int b = (k + 1) % 4; // next vertex index

        bool edge_is_horz   = ((k & 1) == 0);
        bool edge_is_active = edge_is_horz ^ is_horizontal;
        float2 dir          = edge_is_active ? fiber_angle : dir_ortho;
        float  rest_len     = edge_is_active ? rest_b      : rest_a;

        if (!edge_intersects(v[a], v[b], barycenter, dir, intersection))
            continue;

        float2 d = make_float2(
            intersection.x - barycenter.x,
            intersection.y - barycenter.y);
        float  len = sqrtf(d.x*d.x + d.y*d.y);
        if (len == 0.0f) continue;

        float mag = ks_radial * (len - rest_len) / len;

        atomicAdd(&force[vidx[a]].x, d.x * mag);
        atomicAdd(&force[vidx[a]].y, d.y * mag);
        atomicAdd(&force[vidx[b]].x, d.x * mag);
        atomicAdd(&force[vidx[b]].y, d.y * mag);
    }
}

void MechSim::step(float dt, float ks_edge, float ks_radial, float ks_boundary, float *T_a, float c_f)
{
    if (dt <= 0.0f)
    {
        throw std::runtime_error("Time step must be positive");
    }

    // Clear forces
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    clear_forces<<<numBlocks, blockSize>>>(d_force, N);
    check(cudaGetLastError());

    // Calculate velocities
    calculate_velocity_pv<<<(N + blockSize - 1) / blockSize, blockSize>>>(d_vel, d_pos_c, d_pos_p, dt, N);
    check(cudaGetLastError());

    // Calculate orthogonal forces
    if (C > 0)
    {
        int numCells = C;
        dim3 gridDim((numCells + blockSize - 1) / blockSize);
        calculate_orthogonal_forces<<<gridDim, blockSize>>>(d_pos_c, d_fiber_angles, T_a, d_orthogonal_rest_lengths, d_cell_vidx, d_active_spring_is_horizontal, d_force, numCells, ks_radial, c_f);
        check(cudaGetLastError());
    }

    // Calculate structural forces
    dim3 gridDim((nx + 15) / 16, (ny + 15) / 16);
    dim3 blockDim(16, 16);
    calculate_structural_forces<<<gridDim, blockDim>>>(d_pos_c, d_vel, d_force, nx, ny, ks_edge, MechSim::rest_length_edge, damping);
    check(cudaGetLastError());

    // Calculate boundary forces
    calculate_boundry_forces<<<(numBoundary + blockSize - 1) / blockSize, blockSize>>>(d_pos_c, d_force, d_boundaryIdx, d_boundaryPositions, ks_boundary, numBoundary);
    check(cudaGetLastError());

    // Position Verlet integration
    position_verlet<<<(N + blockSize - 1) / blockSize, blockSize>>>(d_pos_c, d_pos_p, d_force, dt, N);
    check(cudaGetLastError());

}