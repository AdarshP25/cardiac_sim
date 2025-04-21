#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>


// Params
const int nx = 1002;
const int ny = 1002;
const int nt = 1000001;
const float dt = 0.001f;
const float eps = 0.08f;
const float a = 0.0001f;
const float b = 0.8f;
const float I_ext = 0.0f;
const float D = 0.001f;
const int snapshot_interval = 1000;
const int num_snapshots = nt / snapshot_interval;

// CUDA kernel for 5-point Laplacian
__global__ void laplacianKernel(float* V, float* lap_V, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix >= nx || iy >= ny) return;
    
    int idx = iy * nx + ix;
    float lap = 0.0f;
    
    // 5-point stencil: [0, 1, 0], [1, -4, 1], [0, 1, 0]
    if (ix > 0)      lap += V[idx - 1];
    if (ix < nx - 1) lap += V[idx + 1];
    if (iy > 0)      lap += V[idx - nx];
    if (iy < ny - 1) lap += V[idx + nx];
    lap += -4.0f * V[idx];
    
    lap_V[idx] = lap;
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
    V_new[idx] = V_val + D * lap_V[idx] + (V_val - V_cubed / 3.0f - R[idx] + I_ext) * dt;
    
    // Update R: R_new = R + eps * (V + a - b * R) * dt
    R_new[idx] = R[idx] + eps * (V_val + a - b * R[idx]) * dt;
}

int main() {
    try {
        // Initialize cuBLAS
        cublasHandle_t cublasHandle;
        cublasCreate(&cublasHandle);
        
        // Allocate host and device memory
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
        std::memset(h_V, 0, size);
        std::memset(h_R, 0, size);
        for (int i = 51; i < 100; ++i) {
            for (int j = 51; j < 100; ++j) {
                h_V[i * nx + j] = 1.0f;
            }
            for (int j = 0; j < 51; ++j) {
                h_R[i * nx + j] = 1.0f;
                h_V[i * nx + j] = -1.0f;
            }
        }
        
        // Copy initial conditions to device
        cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_R, h_R, size, cudaMemcpyHostToDevice);
        
        // Set up CUDA grid and block dimensions
        dim3 blockDim(16, 16);
        dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);
        
        // Allocate host memory for snapshots
        float* h_frames = new float[num_snapshots * nx * ny];
        int snap_index = 0;
        
        // Time-stepping loop
        for (int t = 1; t < nt; ++t) {
            // Compute Laplacian
            laplacianKernel<<<gridDim, blockDim>>>(d_V, d_lap_V, nx, ny);
            cudaGetLastError();
            
            // Update V and R
            updateKernel<<<gridDim, blockDim>>>(d_V, d_R, d_V_new, d_R_new, d_lap_V,
                                               D, dt, eps, a, b, I_ext, nx, ny);
            cudaGetLastError();
            
            // Swap pointers for next iteration
            std::swap(d_V, d_V_new);
            std::swap(d_R, d_R_new);
            
            // Save snapshot every 1000 steps
            if (t % snapshot_interval == 0) {
                cudaMemcpy(&h_frames[snap_index * nx * ny], d_V, size, cudaMemcpyDeviceToHost);
                snap_index++;
                std::cout << "Time step: " << t << std::endl;
                
                // Write snapshot to binary file
                std::ofstream out("data/snapshot_" + std::to_string(t) + ".bin", std::ios::binary);
                out.write(reinterpret_cast<char*>(&h_frames[(snap_index - 1) * nx * ny]), size);
                out.close();
            }
            
            cudaDeviceSynchronize();
        }
        
        // Clean up
        delete[] h_V;
        delete[] h_R;
        delete[] h_frames;
        cudaFree(d_V);
        cudaFree(d_R);
        cudaFree(d_V_new);
        cudaFree(d_R_new);
        cudaFree(d_lap_V);
        cudaFree(d_temp);
        cublasDestroy(cublasHandle);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}