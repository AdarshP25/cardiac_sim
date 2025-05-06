#include "reac_diff_sim.hpp"
#include "mech_sim.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstring>
#include <thread>
#include <stdexcept>
#include <string>

static void check(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}


void initial_excitation(float *h_u, float *h_v, int nx, int ny)
{
    size_t N = static_cast<size_t>(nx) * ny;
    std::memset(h_u, 0, N * sizeof(float));
    std::memset(h_v, 0, N * sizeof(float));

    // Example excitation pattern (adjust as needed)
    int cx = nx / 2;
    int cy = ny / 2;
    for (int j = cy - 50; j < cy + 50; ++j) {
       for (int i = cx - 50; i < cx + 50; ++i) {
           if (i >= 0 && i < nx && j >=0 && j < ny) {
               size_t idx = static_cast<size_t>(j) * nx + i;
               h_u[idx] = 0.9f;
               h_v[idx] = 0.4f;
           }
       }
    }
}

int main()
{
    // Simulation parameters
    const int nx = 1002, ny = 1002, nt = 100001; // Grid size, number of timesteps
    const float dt = 0.001f;                    // Timestep size

    // Reaction-Diffusion parameters
    const float eps0 = 0.01f, a = 0.1035f, k = 8.0f;
    const float D = 0.01f, mu1 = 0.15f, mu2 = 0.15f;

    // Mechanics parameters
    const float ks_edge = 100.0f;  // Stiffness for axial springs
    const float ks_radial = 100.0f; // Stiffness for diagonal springs

    const float T0 = 50.0f;       // Maximum active tension (tune this - start comparable to ks_axial?)
    const float beta = 20.0f;    // Steepness of activation (tune this)
    const float ua = 0.15f;      // Activation threshold for u (tune this)
    const float active_force_scaling = 0.1f; // Scales tension to force (tune this)

    const int snapshot_interval = 1000;

    cudaStream_t ioStream;
    check(cudaStreamCreate(&ioStream));

    size_t N = static_cast<size_t>(nx) * ny;
    size_t bytes_rd = N * sizeof(float);
    size_t bytes_mech = N * sizeof(float2);

    // Pinned host memory buffers for asynchronous snapshot writing
    float *h_frame_rd[2]; 
    check(cudaHostAlloc(&h_frame_rd[0], bytes_rd, cudaHostAllocDefault));
    check(cudaHostAlloc(&h_frame_rd[1], bytes_rd, cudaHostAllocDefault));

    float2 *h_frame_mech[2];
    check(cudaHostAlloc(&h_frame_mech[0], bytes_mech, cudaHostAllocDefault));
    check(cudaHostAlloc(&h_frame_mech[1], bytes_mech, cudaHostAllocDefault));

    // RD Sim
    ReacDiffSim sim(nx, ny);
    float *h_u = new float[N];
    float *h_v = new float[N];
    initial_excitation(h_u, h_v, nx, ny);
    
    check(cudaMemcpy(sim.d_u, h_u, bytes_rd, cudaMemcpyHostToDevice));
    check(cudaMemcpy(sim.d_v, h_v, bytes_rd, cudaMemcpyHostToDevice));
    delete[] h_u;
    delete[] h_v;


    MechSim mechSim(nx, ny);

    // Time iter
    int buf_idx = 0;
    for (int t = 1; t <= nt; ++t)
    {
        
        sim.step(D, dt, eps0, a, k, mu1, mu2);

        
        mechSim.step(dt, ks_edge, ks_radial, sim.d_u, T0, beta, ua, active_force_scaling);

        // Snapshot saving logic
        if (t % snapshot_interval == 0)
        {
            buf_idx = 1 - buf_idx; // Toggle buffer index

            check(cudaMemcpyAsync(
                h_frame_rd[buf_idx],
                sim.d_u, // Source: device u field
                bytes_rd,
                cudaMemcpyDeviceToHost,
                ioStream));

            check(cudaMemcpyAsync(
                h_frame_mech[buf_idx],
                mechSim.d_pos_c,
                bytes_mech,
                cudaMemcpyDeviceToHost,
                ioStream));

            
            std::thread([t, buf_idx, bytes_rd, bytes_mech, h_frame_rd, h_frame_mech, ioStream]() mutable // Capture mutable ioStream
                {
                    check(cudaStreamSynchronize(ioStream)); // Wait for copies on ioStream to finish

                    
                    std::string filename_u = "data2/u_" + std::to_string(t) + ".bin";
                    std::ofstream outU(filename_u, std::ios::binary);
                    if(outU) {
                        outU.write(reinterpret_cast<char*>(h_frame_rd[buf_idx]), bytes_rd);
                    } else {
                         std::cerr << "Error opening file: " << filename_u << std::endl;
                    }

                    std::string filename_x = "data2/x_" + std::to_string(t) + ".bin";
                    std::ofstream outX(filename_x, std::ios::binary);
                     if(outX) {
                        outX.write(reinterpret_cast<char*>(h_frame_mech[buf_idx]), bytes_mech);
                    } else {
                        std::cerr << "Error opening file: " << filename_x << std::endl;
                    }
                }).detach();

            std::cout << "Time step: " << t << " (Snapshot scheduled)" << std::endl;
        }
    }

    // Cleanup
    check(cudaDeviceSynchronize());
    check(cudaStreamSynchronize(ioStream));


    cudaFreeHost(h_frame_rd[0]);
    cudaFreeHost(h_frame_rd[1]);
    cudaFreeHost(h_frame_mech[0]);
    cudaFreeHost(h_frame_mech[1]);

    check(cudaStreamDestroy(ioStream));

    std::cout << "Simulation finished." << std::endl;
    return 0;
}