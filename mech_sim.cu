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

__host__ __device__ __forceinline__
int vid(int x,int y,int nx){ return y*nx + x; }

MechSim::MechSim(int nx_, int ny_, float *h_fiber_angles, float damping_)
    : nx(nx_), ny(ny_), damping(damping_)
{
    N = nx*ny;                   // vertices
    C = (nx-1)*(ny-1);           // cells
    if(N==0 || C<0) throw std::runtime_error("bad grid");

    // ---- device buffers ---------------------------------------------
    check(cudaMalloc(&d_pos_c , N*sizeof(float2)));
    check(cudaMalloc(&d_pos_p , N*sizeof(float2)));
    check(cudaMalloc(&d_vel   , N*sizeof(float2)));
    check(cudaMalloc(&d_force , N*sizeof(float2)));

    check(cudaMalloc(&d_cell_vidx , C*4*sizeof(int)));
    check(cudaMalloc(&d_intersection_ratio, C*sizeof(float)));
    check(cudaMalloc(&d_orthogonal_rest_lengths, C*sizeof(float)));
    check(cudaMalloc(&d_active_spring_is_horizontal, C*sizeof(char)));

    // ---- initial vertex positions in row‑major order ----------------
    std::vector<float2> h_pos(N);
    for(int y=0;y<ny;++y)
        for(int x=0;x<nx;++x)
            h_pos[vid(x,y,nx)] = make_float2(x*rest_length_edge,
                                             y*rest_length_edge);
    h_pos[vid(0,2,nx)].x += 0.1f;
    h_pos[vid(2,1,nx)].x += 0.0f;

    check(cudaMemcpy(d_pos_c,h_pos.data(),N*sizeof(float2),cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_pos_p,h_pos.data(),N*sizeof(float2),cudaMemcpyHostToDevice));
    check(cudaMemset (d_vel  ,0,N*sizeof(float2)));
    check(cudaMemset (d_force,0,N*sizeof(float2)));

    std::vector<int>    h_idx(C*4);
    std::vector<char>   h_asph(C,0);
    std::vector<float>  h_rest(C);
    std::vector<float> h_intersection_ratio(C);

    int ci=0;
    for(int y=0;y<ny-1;++y)
    {
        for(int x=0;x<nx-1;++x,++ci)
        {
            int i0 = vid(x  ,y  ,nx);
            int i1 = vid(x+1,y  ,nx);
            int i2 = vid(x+1,y+1,nx);
            int i3 = vid(x  ,y+1,nx);

            h_idx[ci*4+0]=i0; h_idx[ci*4+1]=i1;
            h_idx[ci*4+2]=i2; h_idx[ci*4+3]=i3;

            float phi = h_fiber_angles[ci];
            float cos_phi = cosf(phi);
            float sin_phi = sinf(phi);

            h_rest[ci] = fabsf( (rest_length_edge*0.5f) / (cos_phi) );

            if(phi <= M_PI_4) {
                h_asph[ci] = 1;
                h_intersection_ratio[ci] = (0.5f * rest_length_edge - h_rest[ci] * sin_phi) / rest_length_edge;
            } else if (phi <= 3 * M_PI_4) {
                h_intersection_ratio[ci] = (0.5f * rest_length_edge + h_rest[ci] * cos_phi) / rest_length_edge;
            } else {
                h_asph[ci] = 1;
                h_intersection_ratio[ci] = (0.5f * rest_length_edge + h_rest[ci] * sin_phi) / rest_length_edge;
            }
        }
    }
    check(cudaMemcpy(d_cell_vidx ,h_idx .data(),C*4*sizeof(int)  ,cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_intersection_ratio, h_intersection_ratio.data(),C*sizeof(float),cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_active_spring_is_horizontal,h_asph.data(),C*sizeof(char),cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_orthogonal_rest_lengths   ,h_rest.data(),C*sizeof(float) ,cudaMemcpyHostToDevice));


    std::vector<int>    bIdx;
    std::vector<float2> bPos;
    for(int y=0;y<ny;++y)
        for(int x=0;x<nx;++x)
            if(x==0||x==nx-1||y==0||y==ny-1){
                bIdx.push_back(vid(x,y,nx));
                bPos.push_back(make_float2(x*rest_length_edge,
                                           y*rest_length_edge));
            }

    numBoundary = (int)bIdx.size();
    check(cudaMalloc(&d_boundaryIdx      , numBoundary*sizeof(int)));
    check(cudaMalloc(&d_boundaryPositions, numBoundary*sizeof(float2)));
    check(cudaMemcpy(d_boundaryIdx      ,bIdx.data(),numBoundary*sizeof(int)   ,cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_boundaryPositions,bPos.data(),numBoundary*sizeof(float2),cudaMemcpyHostToDevice));
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

        float2 previous_pos = prev_pos[i];

        float2 accel = force[i];

        float2 next_pos;
        float dt_sq = dt * dt;

        next_pos.x = 2.0f * current_pos.x - previous_pos.x + accel.x * dt_sq;
        next_pos.y = 2.0f * current_pos.y - previous_pos.y + accel.y * dt_sq;

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

__global__ void calculate_boundary_forces(const float2 *__restrict__ pos,
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

// averages Ta over the four vertices *inside the kernel*  (no extra buffers)
__global__ void calculate_orthogonal_forces(
        const float2* __restrict__ pos,
        const float*  __restrict__ intersect_ratios,         // per cell (C)
        const float*  __restrict__ Ta_vertex,            // per vertex (N)
        const float*  __restrict__ rest_passive,         // per cell (C)
        const int*    __restrict__ cell_vidx,            // 4 vtx / cell (C*4)
        const bool*   __restrict__ active_is_horz,       // per cell (C)
        float2*       __restrict__ force,                // per vertex (N)
        int   C,
        float ks_radial, float c_f)
{
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= C) return;

    // -------- vertex indices & positions -----------------------------
    const int i0 = cell_vidx[4*ci+0];
    const int i1 = cell_vidx[4*ci+1];
    const int i2 = cell_vidx[4*ci+2];
    const int i3 = cell_vidx[4*ci+3];

    const float2 v0 = pos[i0], v1 = pos[i1],
                 v2 = pos[i2], v3 = pos[i3];

    const float2 bary = make_float2( (v0.x+v1.x+v2.x+v3.x)*0.25f,
                                     (v0.y+v1.y+v2.y+v3.y)*0.25f );

    // ---------- activation: simple 4‑vertex average ------------------
    float Ta_avg = 0.25f * ( Ta_vertex[i0] + Ta_vertex[i1]
                           + Ta_vertex[i2] + Ta_vertex[i3] );

    // ---------- spring rest lengths ---------------------------------
    const float rest_a = rest_passive[ci];
    const float rest_b = rest_a / (1.0f + c_f * Ta_avg);

    // ---------- local shortcuts -------------------------------------

    const int    vIdx[4] = { i0,i1,i2,i3 };
    const float2 vPos[4] = { v0,v1,v2,v3 };

    const float intersect_ratio = intersect_ratios[ci];

    // ---------- edge loop -------------------------------------------
    #pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        int a = k, b = (k+1)&3;

        float t = intersect_ratios[ci];               // 0…1
        float2 edge = make_float2(vPos[b].x - vPos[a].x,
                                vPos[b].y - vPos[a].y);
        float2 q    = make_float2(vPos[a].x + t*edge.x,
                                vPos[a].y + t*edge.y);

        float2 d    = make_float2(q.x - bary.x, q.y - bary.y);
        float  length  = hypotf(d.x, d.y);
        if(length < 1e-7f) continue;
        d.x /= length; d.y /= length;

        bool edge_horz  = !(k & 1);             // 0,2 horizontal; 1,3 vertical
        bool spring_act = edge_horz ^ active_is_horz[ci];

        float  restLen = spring_act ? rest_b : rest_a;

        float mag = ks_radial * (length - restLen);

        atomicAdd(&force[vIdx[a]].x, -d.x * mag);
        atomicAdd(&force[vIdx[a]].y, -d.y * mag);
        atomicAdd(&force[vIdx[b]].x, -d.x * mag);
        atomicAdd(&force[vIdx[b]].y, -d.y * mag);
    }
}

//--------------------------------------------------------------------
// Do a single Position‑Verlet step on the CPU.
//
//  d_pos_c   : device pointer to current positions   (xₙ)
//  d_pos_p   : device pointer to previous positions  (xₙ₋₁)
//  d_force   : device pointer to forces at time n    (fₙ = m aₙ)
//
//  N         : number of vertices
//  dt        : timestep (s)
//
// After the call       d_pos_c and d_pos_p are updated on the GPU.
//--------------------------------------------------------------------
void cpuVerletHostStep(float2 *d_pos_c,
                       float2 *d_pos_p,
                       const float2 *d_force,
                       int N,
                       float dt)
{
    if (dt <= 0.0f || N == 0) throw std::runtime_error("bad arguments");

    // 1.  Bring the three arrays to the host
    std::vector<float2> h_pos_c(N), h_pos_p(N), h_force(N);

    check(cudaMemcpy(h_pos_c.data(), d_pos_c, N*sizeof(float2),
                     cudaMemcpyDeviceToHost));
    check(cudaMemcpy(h_pos_p.data(), d_pos_p, N*sizeof(float2),
                     cudaMemcpyDeviceToHost));
    check(cudaMemcpy(h_force.data(), d_force, N*sizeof(float2),
                     cudaMemcpyDeviceToHost));

    // 2.  Integrate on the CPU
    const float dt2 = dt*dt;
    //#pragma omp parallel for       // optional: use OpenMP if available
    for (int i = 0; i < N; ++i)
    {
        float2 xn   = h_pos_c[i];
        float2 xn_1 = h_pos_p[i];
        float2 a    = h_force[i];          // mass = 1 → a = f

        float2 xn1;                        // x_{n+1}
        xn1.x = 2.0f*xn.x - xn_1.x + a.x*dt2;
        xn1.y = 2.0f*xn.y - xn_1.y + a.y*dt2;

        h_pos_p[i] = xn;                   // store old xₙ   → xₙ₋₁
        h_pos_c[i] = xn1;                  // store new xₙ₊₁ → xₙ
    }

    // 3.  Copy positions back to the device
    check(cudaMemcpy(d_pos_c, h_pos_c.data(), N*sizeof(float2),
                     cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_pos_p, h_pos_p.data(), N*sizeof(float2),
                     cudaMemcpyHostToDevice));
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


    // Calculate orthogonal forces
    if (C > 0)
    {
        int numCells = C;
        dim3 gridDim((numCells + blockSize - 1) / blockSize);
        calculate_orthogonal_forces<<<gridDim, blockSize>>>(d_pos_c, d_intersection_ratio, T_a, d_orthogonal_rest_lengths, d_cell_vidx, d_active_spring_is_horizontal, d_force, numCells, ks_radial, c_f);
        check(cudaGetLastError());
    }

    // Calculate structural forces
    dim3 gridDim((nx + 15) / 16, (ny + 15) / 16);
    dim3 blockDim(16, 16);
    calculate_structural_forces<<<gridDim, blockDim>>>(d_pos_c, d_vel, d_force, nx, ny, ks_edge, MechSim::rest_length_edge, damping);
    check(cudaGetLastError());

    // Calculate boundary forces
    calculate_boundary_forces<<<(numBoundary + blockSize - 1) / blockSize, blockSize>>>(d_pos_c, d_force, d_boundaryIdx, d_boundaryPositions, ks_boundary, numBoundary);
    check(cudaGetLastError());

    // Position Verlet integration
    //position_verlet<<<(N + blockSize - 1) / blockSize, blockSize>>>(d_pos_c, d_pos_p, d_force, dt, N);
    cpuVerletHostStep(d_pos_c, d_pos_p, d_force, N, dt);
    //check(cudaGetLastError());

    // Calculate velocities
    calculate_velocity_pv<<<(N + blockSize - 1) / blockSize, blockSize>>>(d_vel, d_pos_c, d_pos_p, dt, N);
    check(cudaGetLastError());

}