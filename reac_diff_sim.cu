#include "reac_diff_sim.hpp"
#include <stdexcept>

// helper
static void check(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}

ReacDiffSim::ReacDiffSim(int nx_, int ny_)
    : nx(nx_), ny(ny_),
      d_u(nullptr), d_v(nullptr),
      d_u_new(nullptr), d_v_new(nullptr),
      d_lap_u(nullptr)
{
    size_t sz = size_t(nx) * ny * sizeof(float);
    check(cudaMalloc(&d_u, sz));
    check(cudaMalloc(&d_v, sz));
    check(cudaMalloc(&d_u_new, sz));
    check(cudaMalloc(&d_v_new, sz));
    check(cudaMalloc(&d_lap_u, sz));
}

ReacDiffSim::~ReacDiffSim()
{
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_u_new);
    cudaFree(d_v_new);
    cudaFree(d_lap_u);
}

// CUDA kernel for 5-point Laplacian w/ Neumann Boundary Conditions
__global__ void laplacianKernel(const float *__restrict__ u, float *__restrict__ lap_u, int nx, int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;

    
    int idx = iy * nx + ix;

    // mirror (Neumann) boundary: if you step outside, reflect
    int ixm = (ix > 0) ? ix - 1 : 1;
    int ixp = (ix < nx - 1) ? ix + 1 : nx - 2;
    int iym = (iy > 0) ? iy - 1 : 1;
    int iyp = (iy < ny - 1) ? iy + 1 : ny - 2;

    // 5 point Laplacian
    float left = u[iy * nx + ixm];
    float right = u[iy * nx + ixp];
    float bottom = u[iym * nx + ix];
    float top = u[iyp * nx + ix];

    lap_u[idx] = left + right + bottom + top - 4.f * u[idx];
}

// CUDA kernel for Aliev-Panfilov update
__global__ void updateKernel(float *u, float *v, float *u_new, float *v_new, float *lap_u, float D, float dt, float eps_0, float a, float k, float mu1, float mu2, int nx, int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= nx || iy >= ny)
        return;

    int idx = iy * nx + ix;

    float u_val = u[idx];
    float v_val = v[idx];
    float eps = eps_0 + ((mu1 * v_val) / (mu2 + u_val));
    u_new[idx] = u_val + ((-1 * k) * u_val * (u_val - a) * (u_val - 1) - (u_val * v_val)) * dt + D * lap_u[idx];
    v_new[idx] = v[idx] + eps * ((-1 * v_val) - (k * u_val * (u_val - a - 1))) * dt;
}

void ReacDiffSim::step(float D, float dt,
                       float eps0, float a, float k,
                       float mu1, float mu2)
{
    dim3 block(32, 8);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    laplacianKernel<<<grid, block>>>(d_u, d_lap_u, nx, ny);
    updateKernel<<<grid, block>>>(
        d_u, d_v, d_u_new, d_v_new,
        d_lap_u, D, dt, eps0, a, k, mu1, mu2,
        nx, ny);
    check(cudaDeviceSynchronize());

    std::swap(d_u, d_u_new);
    std::swap(d_v, d_v_new);
}
