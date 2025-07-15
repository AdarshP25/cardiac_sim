#include "mech_sim.hpp"
#include <stdexcept>
#include <math.h>
#include <math_constants.h>
#include <iostream>

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

MechSim::MechSim(int nx_, int ny_, float *h_fiber_angles, float damping_, int padding_)
    : nx(nx_), ny(ny_), damping(damping_), padding(padding_)
{
    N = nx*ny; C = (nx-1)*(ny-1);
    if(N==0||C<0) throw std::runtime_error("bad grid dims");

    // --- device buffers ------------------------------------
    check(cudaMalloc(&d_global_pos, N*sizeof(float2)));   // reference grid X₀
    check(cudaMalloc(&d_pos_c     , N*sizeof(float2)));   // displacement uₙ
    check(cudaMalloc(&d_pos_p     , N*sizeof(float2)));   // displacement uₙ₋₁
    check(cudaMalloc(&d_vel       , N*sizeof(float2)));
    check(cudaMalloc(&d_force     , N*sizeof(float2)));

    check(cudaMalloc(&d_cell_vidx , C*4*sizeof(int)));
    check(cudaMalloc(&d_intersection_ratio,           C*sizeof(float)));
    check(cudaMalloc(&d_orthogonal_rest_lengths,       C*sizeof(float)));
    check(cudaMalloc(&d_active_spring_is_horizontal,   C*sizeof(char)));

    // --- build host reference grid -------------------------
    std::vector<float2> h_ref(N);
    for(int y=0;y<ny;++y)
        for(int x=0;x<nx;++x)
            h_ref[vid(x,y,nx)] = make_float2(x*rest_length_edge,
                                            y*rest_length_edge);

    check(cudaMemcpy(d_global_pos, h_ref.data(), N*sizeof(float2), cudaMemcpyHostToDevice));
    check(cudaMemset(d_pos_c, 0, N*sizeof(float2)));
    check(cudaMemset(d_pos_p, 0, N*sizeof(float2)));
    check(cudaMemset(d_vel  , 0, N*sizeof(float2)));
    check(cudaMemset(d_force, 0, N*sizeof(float2)));

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

            float orphi = h_fiber_angles[ci];
            float phi = orphi;
            if (phi > M_PI_4) {
                phi -= M_PI_2;
            }
            if (phi > M_PI_4) {
                phi -= M_PI_2;
            }

            float cos_phi = cosf(phi);
            float sin_phi = sinf(phi);

            h_rest[ci] = fabsf( (rest_length_edge*0.5f) / (cos_phi) );

            h_intersection_ratio[ci] = (0.5f * rest_length_edge + 0.5f * rest_length_edge * tanf(phi)) / rest_length_edge;

            if(orphi <= M_PI_4 || orphi >= 3 * M_PI_4) {
                h_asph[ci] = 1;
            }
            // std::cout << "Cell " << ci << ": "
            //      << "rest_length = " << h_rest[ci] << ", "
            //      << "intersection_ratio = " << h_intersection_ratio[ci] << ", "
            //      << "is_horizontal = " << (h_asph[ci] ? "true" : "false") << ", " << orphi << std::endl;
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
    if (d_pos_c) cudaFree(d_pos_c);
    if (d_pos_p) cudaFree(d_pos_p);
    if (d_vel) cudaFree(d_vel);
    if (d_force) cudaFree(d_force);
    if (d_cell_vidx) cudaFree(d_cell_vidx);
    if (d_global_pos) cudaFree(d_global_pos);
    if (d_intersection_ratio) cudaFree(d_intersection_ratio);
    if (d_orthogonal_rest_lengths) cudaFree(d_orthogonal_rest_lengths);
    if (d_active_spring_is_horizontal) cudaFree(d_active_spring_is_horizontal);
    if (d_boundaryIdx) cudaFree(d_boundaryIdx);
    if (d_boundaryPositions) cudaFree(d_boundaryPositions);
}

__device__ __forceinline__ float2 world(float2 u, const float2& X0)
{ return make_float2(X0.x + u.x, X0.y + u.y); }

__global__ void position_verlet(
    float2* u, float2* u_prev, const float2* force,
    float dt,int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x; if(i>=N) return;
    float2 cur = u[i];
    float2 prv = u_prev[i];
    float2 a   = force[i];
    float2 nxt;
    float dt2 = dt*dt;
    nxt.x = 2.f*cur.x - prv.x + a.x*dt2;
    nxt.y = 2.f*cur.y - prv.y + a.y*dt2;
    u_prev[i]=cur; u[i]=nxt;
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
    const float2* __restrict__ u,
    const float2* __restrict__ global_pos,
    const float2* __restrict__ vel,
    float2*       __restrict__ force,
    int nx,int ny, float ks,float rest,float damping)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix>=nx||iy>=ny) return;
    int idx = iy*nx+ix;

    float2 pa = world(u[idx], global_pos[idx]);
    float2 f  = make_float2(0.f,0.f);
    const int nbh[4][2]={{-1,0},{+1,0},{0,-1},{0,+1}};
    #pragma unroll
    for(int k=0;k<4;++k){
        int jx=ix+nbh[k][0], jy=iy+nbh[k][1];
        if(jx<0||jx>=nx||jy<0||jy>=ny) continue;
        int j = jy*nx+jx;
        float2 pb = world(u[j], global_pos[j]);
        float2 d  = make_float2(pb.x-pa.x, pb.y-pa.y);
        float len = hypotf(d.x,d.y);
        float mag = ks*(len-rest);
        f.x += mag*d.x/len;
        f.y += mag*d.y/len;
    }
    atomicAdd(&force[idx].x, f.x);
    atomicAdd(&force[idx].y, f.y);
    // damping
    float2 damp = make_float2(-damping*vel[idx].x, -damping*vel[idx].y);
    atomicAdd(&force[idx].x, damp.x);
    atomicAdd(&force[idx].y, damp.y);
}

__global__ void calculate_boundary_forces(
    const float2* __restrict__ u,              // displacement array (N)
    float2*       __restrict__ force,          // accumulated forces  (N)
    const int*    __restrict__ boundaryIdx,    // vertex indices      (numBoundary)
    const float2* __restrict__ boundaryPos,    // world‑space anchors (numBoundary)
    float  ks_boundary,
    int    numBoundary,
    const float2* __restrict__ global_pos)     // reference grid X₀  (N)
{
    int bi = blockIdx.x * blockDim.x + threadIdx.x;   // boundary‑list index
    if (bi >= numBoundary) return;

    int v = boundaryIdx[bi];                          // vertex index in full mesh

    // current world position  X = X₀ + u
    float2 p = world(u[v], global_pos[v]);

    // anchor for this boundary vertex
    float2 target = boundaryPos[bi];

    float fx = ks_boundary * (target.x - p.x);
    float fy = ks_boundary * (target.y - p.y);

    atomicAdd(&force[v].x, fx);   // <- atomic!
    atomicAdd(&force[v].y, fy);
}

__global__ void clamp_forces(float2 *force,
                             float   fmax,          // ≥ 0  (0 ⇒ no clamping)
                             int     N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || fmax <= 0.f) return;

    float2 f = force[i];
    float  m2 = f.x*f.x + f.y*f.y;        // squared magnitude
    float  fmax2 = fmax * fmax;

    if (m2 > fmax2)
    {
        float inv_len = rsqrtf(m2);       // 1 / ‖f‖
        f.x *= fmax * inv_len;
        f.y *= fmax * inv_len;
        force[i] = f;                     // write back
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
        float ks_radial, float c_f, float2 *__restrict__ global_pos, int padding)
{
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= C) return;

    // -------- vertex indices & positions -----------------------------
    const int i0 = cell_vidx[4*ci+0];
    const int i1 = cell_vidx[4*ci+1];
    const int i2 = cell_vidx[4*ci+2];
    const int i3 = cell_vidx[4*ci+3];

    const float2 v0 = world(pos[i0], global_pos[i0]), v1 = world(pos[i1], global_pos[i1]),
                 v2 = world(pos[i2], global_pos[i2]), v3 = world(pos[i3], global_pos[i3]);

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

        float force_mag_x = -d.x * mag;  // Force component in x-direction
        float force_mag_y = -d.y * mag;  // Force component in y-direction

        atomicAdd(&force[vIdx[a]].x, (1 - t) * force_mag_x);
        atomicAdd(&force[vIdx[a]].y, (1 - t) * force_mag_y);
        atomicAdd(&force[vIdx[b]].x, t * force_mag_x);
        atomicAdd(&force[vIdx[b]].y, t * force_mag_y);
    }
}


void MechSim::step(float dt,float ks_edge,float ks_radial,
                   float ks_boundary,float* T_a,float c_f, float fmax)
{
    if(dt<=0.f) throw std::runtime_error("dt<=0");
    int blk=256, nb=(N+blk-1)/blk;
    // clear force accumulator
    clear_forces<<<nb,blk>>>(d_force,N);

    // orthogonal forces
    if(C>0){
        dim3 g((C+blk-1)/blk);  // 1‑D grid
        calculate_orthogonal_forces<<<g,blk>>>(d_pos_c, d_intersection_ratio,
            /*Ta*/T_a, d_orthogonal_rest_lengths, d_cell_vidx,
            d_active_spring_is_horizontal, d_force, C,
            ks_radial, c_f, d_global_pos, padding);
    }

    // structural forces
    dim3 tb(16,16); dim3 gb((nx+15)/16,(ny+15)/16);
    calculate_structural_forces<<<gb,tb>>>(d_pos_c, d_global_pos, d_vel,
        d_force, nx,ny, ks_edge, rest_length_edge, damping);

    // boundary forces
    int nbndBlk = (numBoundary+blk-1)/blk;
    calculate_boundary_forces<<<nbndBlk,blk>>>(d_pos_c, d_force,
        d_boundaryIdx, d_boundaryPositions, ks_boundary, numBoundary,
        d_global_pos);

    //int nbClamp = (N + blk - 1) / blk;
    //clamp_forces<<<nbClamp, blk>>>(d_force, fmax, N);

    // integration step
    position_verlet<<<nb,blk>>>(d_pos_c, d_pos_p, d_force, dt, N);
    calculate_velocity_pv<<<nb,blk>>>(d_vel, d_pos_c, d_pos_p, dt, N);
}