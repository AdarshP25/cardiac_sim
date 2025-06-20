#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <random>


// Params
const int nx = 1002;
const int ny = 1002;
const int nt = 100001;
const float dt = 0.001f;
const float eps = 0.08f;
const float a = 0.0001f;
const float b = 0.8f;
const float I_ext = 0.0f;
const float D = 0.1f;
const int snapshot_interval = 1000;
const int num_snapshots = nt / snapshot_interval;

// CUDA kernel for 5-point Laplacian
__global__ void laplacianKernel(float* V, float* lap_V, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // Shared memory for 32x8 tile + 1-cell halo
    __shared__ float V_shared[10][34]; // 10 rows (y: 8 + 1 + 1), 34 columns (x: 32 + 1 + 1)

    int idx = iy * nx + ix;

    // Load center
    if (ix < nx && iy < ny) {
        V_shared[threadIdx.y + 1][threadIdx.x + 1] = V[idx];
    }

    // Load halo, applying Neumann conditions
    if (threadIdx.x == 0 && ix > 0 && iy < ny) {
        V_shared[threadIdx.y + 1][0] = V[iy * nx + (ix - 1)]; // Left
    } else if (threadIdx.x == 0 && ix == 0 && iy < ny) {
        V_shared[threadIdx.y + 1][0] = V[iy * nx + 1]; // Left boundary: V[-1,j] = V[1,j]
    }
    if (threadIdx.x == blockDim.x - 1 && ix < nx - 1 && iy < ny) {
        V_shared[threadIdx.y + 1][blockDim.x + 1] = V[iy * nx + (ix + 1)]; // Right
    } else if (threadIdx.x == blockDim.x - 1 && ix == nx - 1 && iy < ny) {
        V_shared[threadIdx.y + 1][blockDim.x + 1] = V[iy * nx + (nx - 2)]; // Right boundary: V[nx,j] = V[nx-2,j]
    }
    if (threadIdx.y == 0 && iy > 0 && ix < nx) {
        V_shared[0][threadIdx.x + 1] = V[(iy - 1) * nx + ix]; // Top
    } else if (threadIdx.y == 0 && iy == 0 && ix < nx) {
        V_shared[0][threadIdx.x + 1] = V[nx + ix]; // Bottom boundary: V[i,-1] = V[i,1]
    }
    if (threadIdx.y == blockDim.y - 1 && iy < ny - 1 && ix < nx) {
        V_shared[blockDim.y + 1][threadIdx.x + 1] = V[(iy + 1) * nx + ix]; // Bottom
    } else if (threadIdx.y == blockDim.y - 1 && iy == ny - 1 && ix < nx) {
        V_shared[blockDim.y + 1][threadIdx.x + 1] = V[(ny - 2) * nx + ix]; // Top boundary: V[i,ny] = V[i,ny-2]
    }

    __syncthreads();

    if (ix < nx && iy < ny) {
        float lap = 0.0f;

        // Interior points: standard 5-point stencil
        if (ix > 0 && ix < nx - 1 && iy > 0 && iy < ny - 1) {
            lap = V_shared[threadIdx.y + 1][threadIdx.x] +     // Left
                  V_shared[threadIdx.y + 1][threadIdx.x + 2] + // Right
                  V_shared[threadIdx.y][threadIdx.x + 1] +     // Top
                  V_shared[threadIdx.y + 2][threadIdx.x + 1] - // Bottom
                  4.0f * V_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
        }
        // Boundary points
        else {
            // Left boundary (i=0, 0 < j < ny-1)
            if (ix == 0 && iy > 0 && iy < ny - 1) {
                lap = 2.0f * V_shared[threadIdx.y + 1][threadIdx.x + 2] + // Right (2*V[1,j])
                      V_shared[threadIdx.y][threadIdx.x + 1] +           // Top
                      V_shared[threadIdx.y + 2][threadIdx.x + 1] -       // Bottom
                      4.0f * V_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
            }
            // Right boundary (i=nx-1, 0 < j < ny-1)
            else if (ix == nx - 1 && iy > 0 && iy < ny - 1) {
                lap = V_shared[threadIdx.y + 1][threadIdx.x] +           // Left
                      V_shared[threadIdx.y][threadIdx.x + 1] +           // Top
                      V_shared[threadIdx.y + 2][threadIdx.x + 1] -       // Bottom
                      4.0f * V_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
            }
            // Top boundary (0 < i < nx-1, j=ny-1)
            else if (iy == ny - 1 && ix > 0 && ix < nx - 1) {
                lap = V_shared[threadIdx.y + 1][threadIdx.x] +           // Left
                      V_shared[threadIdx.y + 1][threadIdx.x + 2] +       // Right
                      V_shared[threadIdx.y][threadIdx.x + 1] -           // Top
                      4.0f * V_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
            }
            // Bottom boundary (0 < i < nx-1, j=0)
            else if (iy == 0 && ix > 0 && ix < nx - 1) {
                lap = V_shared[threadIdx.y + 1][threadIdx.x] +           // Left
                      V_shared[threadIdx.y + 1][threadIdx.x + 2] +       // Right
                      V_shared[threadIdx.y + 2][threadIdx.x + 1] -       // Bottom
                      4.0f * V_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
            }
            // Corners
            else if (ix == 0 && iy == 0) { // Bottom-left
                lap = 2.0f * V_shared[threadIdx.y + 1][threadIdx.x + 2] + // Right
                      2.0f * V_shared[threadIdx.y + 2][threadIdx.x + 1] - // Bottom
                      4.0f * V_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
            }
            else if (ix == 0 && iy == ny - 1) { // Top-left
                lap = 2.0f * V_shared[threadIdx.y + 1][threadIdx.x + 2] + // Right
                      V_shared[threadIdx.y][threadIdx.x + 1] -           // Top
                      4.0f * V_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
            }
            else if (ix == nx - 1 && iy == 0) { // Bottom-right
                lap = V_shared[threadIdx.y + 1][threadIdx.x] +           // Left
                      2.0f * V_shared[threadIdx.y + 2][threadIdx.x + 1] - // Bottom
                      4.0f * V_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
            }
            else if (ix == nx - 1 && iy == ny - 1) { // Top-right
                lap = V_shared[threadIdx.y + 1][threadIdx.x] +           // Left
                      V_shared[threadIdx.y][threadIdx.x + 1] -           // Top
                      4.0f * V_shared[threadIdx.y + 1][threadIdx.x + 1]; // Center
            }
        }

        lap_V[idx] = lap;
    }
}

// CUDA kernel for FitzHugh-Nagumo update
__global__ void updateKernel(float* V, float* R, float* V_new, float* R_new, float* lap_V,
                            float D, float dt, float eps, float a, float b, float I_ext,
                            int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix >= nx || iy >= ny) return;
    
    int idx = iy * nx + ix;
    
    // Update V: V_new = V + D * lap_V + (V - (V^3)/3 - R + I_ext) * dt
    float V_val = V[idx];
    float V_cubed = V_val * V_val * V_val;
    V_new[idx] = V_val + D * lap_V[idx] + ((V_val - V_cubed / 3.0f - R[idx] + I_ext)) * dt;
    
    // Update R: R_new = R + eps * (V + a - b * R) * dt
    R_new[idx] = R[idx] + eps * (V_val + a - b * R[idx]) * dt;
}

int main() {
    try {
 
        // Allocate mem
        size_t size = nx * ny * sizeof(float);
        float* h_V = new float[nx * ny];
        float* h_R = new float[nx * ny];
        float* d_V, *d_R, *d_V_new, *d_R_new, *d_lap_V, *d_temp;
        
        cudaMalloc(&d_V, size);
        cudaMalloc(&d_R, size);
        cudaMalloc(&d_V_new, size);
        cudaMalloc(&d_R_new, size);
        cudaMalloc(&d_lap_V, size);
        cudaMalloc(&d_temp, size);
        
        // Potential initial conditions

        // Initialize V with scattered 50x50 squares, R to 0
        std::memset(h_V, 0, size);
        std::memset(h_R, 0, size);

        // Random placement of 10 squares (from above)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 951);
        std::vector<std::pair<int, int>> squares;
        const int num_squares = 100;
        while (squares.size() < num_squares) {
            int i = dis(gen);
            int j = dis(gen);
            bool overlap = false;
            for (const auto& [si, sj] : squares) {
                int di = std::abs(i + 25 - (si + 25));
                int dj = std::abs(j + 25 - (sj + 25));
                if (di < 50 && dj < 50) {
                    overlap = true;
                    break;
                }
            }
            if (!overlap) {
                squares.emplace_back(i, j);
                for (int ii = i; ii < i + 50; ++ii) {
                    for (int jj = j; jj < j + 50; ++jj) {
                        h_V[ii * nx + jj] = 1.0f;
                    }
                }
            }
        }
        
        // Copy initial conditions to device
        cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_R, h_R, size, cudaMemcpyHostToDevice);
        
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

            laplacianKernel<<<gridDim, blockDim>>>(d_V, d_lap_V, nx, ny);
            updateKernel<<<gridDim, blockDim>>>(d_V, d_R, d_V_new, d_R_new, d_lap_V, D, dt, eps, a, b, I_ext, nx, ny);
            cudaDeviceSynchronize();

            std::swap(d_V, d_V_new);
            std::swap(d_R, d_R_new);
            
            // Save snapshot
            if (t % snapshot_interval == 0) {
                cudaMemcpy(&h_frames[snap_index * nx * ny], d_V, size, cudaMemcpyDeviceToHost);
                snap_index++;
                std::cout << "Time step: " << t << std::endl;
                
                std::ofstream out("data/snapshot_" + std::to_string(t) + ".bin", std::ios::binary);
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
        delete[] h_V;
        delete[] h_R;
        delete[] h_frames;
        cudaFree(d_V);
        cudaFree(d_R);
        cudaFree(d_V_new);
        cudaFree(d_R_new);
        cudaFree(d_lap_V);
        cudaFree(d_temp);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}