#include "reac_diff_sim.hpp"
#include <stdexcept>
#include <random>

// helper
static void check(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}

ReacDiffSim::ReacDiffSim(int nx_, int ny_)
    : nx(nx_), ny(ny_),
      d_u(nullptr), d_v(nullptr), d_Ta(nullptr),
      d_u_new(nullptr), d_v_new(nullptr), d_Ta_new(nullptr),
      d_lap_u(nullptr)
{
    size_t sz = size_t(nx) * ny * sizeof(float);
    check(cudaMalloc(&d_u, sz));
    check(cudaMalloc(&d_v, sz));
    check(cudaMalloc(&d_Ta, sz));
    check(cudaMalloc(&d_u_new, sz));
    check(cudaMalloc(&d_v_new, sz));
    check(cudaMalloc(&d_Ta_new, sz));
    check(cudaMalloc(&d_lap_u, sz));
}

ReacDiffSim::~ReacDiffSim()
{
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_Ta);
    cudaFree(d_u_new);
    cudaFree(d_v_new);
    cudaFree(d_Ta_new);
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
__global__ void updateKernel(float *u, float *v, float *Ta, float *u_new, float *v_new, float *Ta_new, float *lap_u, float D, float dt, float eps_0, float a, float k, float mu1, float mu2, float k_T, int nx, int ny)
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
    float eps_T = (u[idx] < 0.05) ? 10.0 : 1.0;
    Ta_new[idx] = eps_T * (k_T * u[idx] - Ta[idx]) * dt + Ta[idx];

}

void ReacDiffSim::step(float D, float dt,
                       float eps0, float a, float k,
                       float mu1, float mu2, float k_T)
{
    dim3 block(32, 8);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    laplacianKernel<<<grid, block>>>(d_u, d_lap_u, nx, ny);
    updateKernel<<<grid, block>>>(
        d_u, d_v, d_Ta, d_u_new, d_v_new, d_Ta_new,
        d_lap_u, D, dt, eps0, a, k, mu1, mu2, k_T,
        nx, ny);
    check(cudaDeviceSynchronize());

    std::swap(d_u, d_u_new);
    std::swap(d_v, d_v_new);
    std::swap(d_Ta, d_Ta_new);
}


// ── GPU kernel: add `inc` inside a disc of radius r, centred at (cx,cy)
__global__
void add_disc(float *u, int nx, int ny,
              int cx, int cy, int r, float inc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N   = nx * ny;
    if (idx >= N) return;

    int  x  = idx % nx;
    int  y  = idx / nx;
    int dx  = x - cx;
    int dy  = y - cy;

    if (dx*dx + dy*dy <= r*r)
        u[idx] += inc;
}

void ReacDiffSim::randomFocal(float prob, std::mt19937 &rng)
{
    static std::uniform_real_distribution<float> unif01(0.0f, 1.0f);
    if (unif01(rng) >= prob) return;

    /* pick a safe centre  */
    int  r  = int(0.05f * std::min(nx, ny));
    if (r < 1) r = 1;                         // guard tiny grids
    int  cx = r + rand() % (nx - 2*r);
    int  cy = r + rand() % (ny - 2*r);

    // one 1‑D launch is enough for disc radii ≪ grid size
    int N = nx * ny;
    int block = 256;
    int grid  = (N + block - 1) / block;

    add_disc<<<grid, block>>>(d_u, nx, ny, cx, cy, r, 1.0f);
    check(cudaGetLastError());   // macro from your helper
}