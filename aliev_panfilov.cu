#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <random>


// Params
const int nx = 1002;
const int ny = 1002;
const int nt = 1000001;
const float dt = 0.001f;
const float eps_0 = 0.01f;
const float a = 0.1035f;
const float k = 8.0f;
const float D = 0.01f;
const float mu1 = 0.15f;
const float mu2 = 0.15f;

const int snapshot_interval = 1000;
const int num_snapshots = nt / snapshot_interval;

// CUDA kernel for 5-point Laplacian w/ Neumann Boundary Conditions
__global__ void laplacianKernel(float* u, float* lap_u, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // Shared memory for 32x8 tile + 1-cell halo
    __shared__ float u_shared[10][34]; // 10 rows (y: 8 + 1 + 1), 34 columns (x: 32 + 1 + 1)

    int idx = iy * nx + ix;

    // Load center
    if (ix < nx && iy < ny) {
        u_shared[threadIdx.y + 1][threadIdx.x + 1] = u[idx];
    }

    if (threadIdx.x == 0 && ix > 0 && iy < ny) {
        u_shared[threadIdx.y + 1][0] = u[iy * nx + (ix - 1)]; // Left
    } else if (threadIdx.x == 0 && ix == 0 && iy < ny) {
        u_shared[threadIdx.y + 1][0] = u[iy * nx + 1]; // Left boundary: u[-1,j] = u[1,j]
    }
    if (threadIdx.x == blockDim.x - 1 && ix < nx - 1 && iy < ny) {
        u_shared[threadIdx.y + 1][blockDim.x + 1] = u[iy * nx + (ix + 1)]; // vight
    } else if (threadIdx.x == blockDim.x - 1 && ix == nx - 1 && iy < ny) {
        u_shared[threadIdx.y + 1][blockDim.x + 1] = u[iy * nx + (nx - 2)]; // vight boundary: u[nx,j] = u[nx-2,j]
    }
    if (threadIdx.y == 0 && iy > 0 && ix < nx) {
        u_shared[0][threadIdx.x + 1] = u[(iy - 1) * nx + ix]; // Top
    } else if (threadIdx.y == 0 && iy == 0 && ix < nx) {
        u_shared[0][threadIdx.x + 1] = u[nx + ix]; // Bottom boundary: u[i,-1] = u[i,1]
    }
    if (threadIdx.y == blockDim.y - 1 && iy < ny - 1 && ix < nx) {
        u_shared[blockDim.y + 1][threadIdx.x + 1] = u[(iy + 1) * nx + ix]; // Bottom
    } else if (threadIdx.y == blockDim.y - 1 && iy == ny - 1 && ix < nx) {
        u_shared[blockDim.y + 1][threadIdx.x + 1] = u[(ny - 2) * nx + ix]; // Top boundary: u[i,ny] = u[i,ny-2]
    }

    __syncthreads();

    if (ix < nx && iy < ny) {
        float lap = 0.0f;

        // Interior points: standard 5-point stencil
        if (ix > 0 && ix < nx - 1 && iy > 0 && iy < ny - 1) {
            lap = u_shared[threadIdx.y + 1][threadIdx.x] +     // Left
                  u_shared[threadIdx.y + 1][threadIdx.x + 2] + // vight
                  u_shared[threadIdx.y][threadIdx.x + 1] +     // Top
                  u_shared[threadIdx.y + 2][threadIdx.x + 1] - // Bottom
                  4.0f * u_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
        }
        // Boundary points
        else {
            // Left boundary (i=0, 0 < j < ny-1)
            if (ix == 0 && iy > 0 && iy < ny - 1) {
                lap = 2.0f * u_shared[threadIdx.y + 1][threadIdx.x + 2] + // vight (2*u[1,j])
                      u_shared[threadIdx.y][threadIdx.x + 1] +           // Top
                      u_shared[threadIdx.y + 2][threadIdx.x + 1] -       // Bottom
                      4.0f * u_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
            }
            // vight boundary (i=nx-1, 0 < j < ny-1)
            else if (ix == nx - 1 && iy > 0 && iy < ny - 1) {
                lap = u_shared[threadIdx.y + 1][threadIdx.x] +           // Left
                      u_shared[threadIdx.y][threadIdx.x + 1] +           // Top
                      u_shared[threadIdx.y + 2][threadIdx.x + 1] -       // Bottom
                      4.0f * u_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
            }
            // Top boundary (0 < i < nx-1, j=ny-1)
            else if (iy == ny - 1 && ix > 0 && ix < nx - 1) {
                lap = u_shared[threadIdx.y + 1][threadIdx.x] +           // Left
                      u_shared[threadIdx.y + 1][threadIdx.x + 2] +       // vight
                      u_shared[threadIdx.y][threadIdx.x + 1] -           // Top
                      4.0f * u_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
            }
            // Bottom boundary (0 < i < nx-1, j=0)
            else if (iy == 0 && ix > 0 && ix < nx - 1) {
                lap = u_shared[threadIdx.y + 1][threadIdx.x] +           // Left
                      u_shared[threadIdx.y + 1][threadIdx.x + 2] +       // vight
                      u_shared[threadIdx.y + 2][threadIdx.x + 1] -       // Bottom
                      4.0f * u_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
            }
            // Corners
            else if (ix == 0 && iy == 0) { // Bottom-left
                lap = 2.0f * u_shared[threadIdx.y + 1][threadIdx.x + 2] + // vight
                      2.0f * u_shared[threadIdx.y + 2][threadIdx.x + 1] - // Bottom
                      4.0f * u_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
            }
            else if (ix == 0 && iy == ny - 1) { // Top-left
                lap = 2.0f * u_shared[threadIdx.y + 1][threadIdx.x + 2] + // vight
                      u_shared[threadIdx.y][threadIdx.x + 1] -           // Top
                      4.0f * u_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
            }
            else if (ix == nx - 1 && iy == 0) { // Bottom-right
                lap = u_shared[threadIdx.y + 1][threadIdx.x] +           // Left
                      2.0f * u_shared[threadIdx.y + 2][threadIdx.x + 1] - // Bottom
                      4.0f * u_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
            }
            else if (ix == nx - 1 && iy == ny - 1) { // Top-right
                lap = u_shared[threadIdx.y + 1][threadIdx.x] +           // Left
                      u_shared[threadIdx.y][threadIdx.x + 1] -           // Top
                      4.0f * u_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
            }
        }

        lap_u[idx] = lap;
    }
}

// CUDA kernel for FitzHugh-Nagumo update
__global__ void updateKernel(float* u, float* v, float* u_new, float* v_new, float* lap_u, float D, float dt, float eps_0, float a, float k, float mu1, float mu2, int nx, int ny, int t) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix >= nx || iy >= ny) return;
    
    int idx = iy * nx + ix;
    
    // Update u
    float u_val = u[idx];
    float v_val = v[idx];
    float eps = eps_0 + ((mu1 * v_val) / (mu2 + u_val));
    u_new[idx] = u_val + ((-1 * k) * u_val * (u_val - a) * (u_val - 1) - (u_val * v_val)) * dt + D * lap_u[idx];
    v_new[idx] = v[idx] + eps * ((-1 * v_val) - (k * u_val * (u_val - a - 1))) * dt;
}

int main() {
    try {        
        // Allocate mem
        size_t size = nx * ny * sizeof(float);
        float* h_u = new float[nx * ny];
        float* h_v = new float[nx * ny];
        float* d_u, *d_v, *d_u_new, *d_v_new, *d_lap_u, *d_temp;
        
        cudaMalloc(&d_u, size);
        cudaMalloc(&d_v, size);
        cudaMalloc(&d_u_new, size);
        cudaMalloc(&d_v_new, size);
        cudaMalloc(&d_lap_u, size);
        cudaMalloc(&d_temp, size);
        
        // Potential initial conditions

        // Initialize u with scattered 50x50 squares, v to 0
        std::memset(h_u, 0, size);
        std::memset(h_v, 0, size);

        
        for (int i = 500; i < 600; ++i) {
            for (int j = 450; j < 550; ++j) {
                h_u[i * nx + j] = 0.8f;
            }
        }

        for (int i = 500; i < 600; ++i) {
            for (int j = 550; j < 1000; ++j) {
                h_v[i * nx + j] = 1.0f;
            }
        }

        for (int i = 500; i < 1000; ++i) {
            for (int j = 0; j < 450; ++j) {
                h_v[i * nx + j] = 1.0f;
            }
        }

        for (int i = 0; i < 500; ++i) {
            for (int j = 0; j < 1000; ++j) {
                h_v[i * nx + j] = 1.0f;
            }
        }
        
        // Copy initial conditions to device
        cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, h_v, size, cudaMemcpyHostToDevice);
        
        // Set up CUDA grid and block dimensions
        dim3 blockDim(32, 8);
        dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);
        
        // Allocate host memory for snapshots
        float* h_frames = new float[num_snapshots * nx * ny];
        int snap_index = 0;
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        // Time-iter
        for (int t = 1; t < nt; ++t) {

            laplacianKernel<<<gridDim, blockDim>>>(d_u, d_lap_u, nx, ny);
            updateKernel<<<gridDim, blockDim>>>(d_u, d_v, d_u_new, d_v_new, d_lap_u, D, dt, eps_0, a, k, mu1, mu2, nx, ny, t);
            cudaDeviceSynchronize();

            std::swap(d_u, d_u_new);
            std::swap(d_v, d_v_new);
            
            // Save snapshot
            if (t % snapshot_interval == 0) {
                cudaMemcpy(&h_frames[snap_index * nx * ny], d_u, size, cudaMemcpyDeviceToHost);
                snap_index++;
                std::cout << "Time step: " << t << std::endl;
                
                std::ofstream out("data2/snapshot_" + std::to_string(t) + ".bin", std::ios::binary);
                out.write(reinterpret_cast<char*>(&h_frames[(snap_index - 1) * nx * ny]), size);
                out.close();
            }
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Kernel execution time: %f ms\n", milliseconds);
        
        // deallocate mem
        delete[] h_u;
        delete[] h_v;
        delete[] h_frames;
        cudaFree(d_u);
        cudaFree(d_v);
        cudaFree(d_u_new);
        cudaFree(d_v_new);
        cudaFree(d_lap_u);
        cudaFree(d_temp);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}