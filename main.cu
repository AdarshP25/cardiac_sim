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
#include <toml.hpp>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <random>

// nvcc -o sim main.cu reac_diff_sim.cu mech_sim.cu -I.

static void check(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}

void dot_excitation(float *h_u, float *h_v, int nx, int ny)
{
    size_t N = static_cast<size_t>(nx) * ny;
    std::memset(h_u, 0, N * sizeof(float));
    std::memset(h_v, 0, N * sizeof(float));

    // Example excitation pattern
    int cx = nx / 2;
    int cy = ny / 2;
    for (int j = cy; j < cy + 1; ++j) {
       for (int i = cx; i < cx + 1; ++i) {
           if (i >= 0 && i < nx && j >=0 && j < ny) {
               size_t idx = static_cast<size_t>(j) * nx + i;
               h_u[idx] = 2.5f;
           }
       }
    }
}

void small_excitation(float *h_u, float *h_v, int nx, int ny)
{
    size_t N = static_cast<size_t>(nx) * ny;
    std::memset(h_u, 0, N * sizeof(float));
    std::memset(h_v, 0, N * sizeof(float));

    // Example excitation pattern
    int cx = nx / 2;
    int cy = ny / 2;
    for (int j = cy - 2; j < cy + 2; ++j) {
       for (int i = cx - 2; i < cx + 2; ++i) {
           if (i >= 0 && i < nx && j >=0 && j < ny) {
               size_t idx = static_cast<size_t>(j) * nx + i;
               h_u[idx] = 2.5f;
           }
       }
    }
}

void initial_excitation(float *h_u, float *h_v, int nx, int ny)
{
    size_t N = static_cast<size_t>(nx) * ny;
    std::memset(h_u, 0, N * sizeof(float));
    std::memset(h_v, 0, N * sizeof(float));

    // Example excitation pattern
    int cx = nx / 2;
    int cy = ny / 2;
    for (int j = 0; j < 40; ++j) {
       for (int i = 40; i < 50; ++i) {
           if (i >= 0 && i < nx && j >=0 && j < ny) {
               size_t idx = static_cast<size_t>(j) * nx + i;
               h_u[idx] = 2.5f;
           }
       }
    }
    for (int j = 0; j < 40; ++j) {
       for (int i = 50; i < nx; ++i) {
           if (i >= 0 && i < nx && j >=0 && j < ny) {
               size_t idx = static_cast<size_t>(j) * nx + i;
               h_v[idx] = 3.0f;
           }
       }
    }
}

void focal_wave(float *h_u, float *h_v, int nx, int ny) {
    size_t N = static_cast<size_t>(nx) * ny;
    std::memset(h_u, 0, N * sizeof(float));
    std::memset(h_v, 0, N * sizeof(float));

    // Example excitation pattern
    int cx = nx / 2;
    int cy = ny / 2;
    for (int j = cy - 10; j < cy + 10; ++j) {
       for (int i = cx - 10; i < cy + 10; ++i) {
           if (i >= 0 && i < nx && j >=0 && j < ny) {
               size_t idx = static_cast<size_t>(j) * nx + i;
               h_u[idx] = 2.5f;
           }
       }
    }
}

void blank_initial(float *h_u, float *h_v, int nx, int ny) {
    size_t N = static_cast<size_t>(nx) * ny;
    std::memset(h_u, 0, N * sizeof(float));
    std::memset(h_v, 0, N * sizeof(float));
}

float sampleVonMisesAxial(float psi, float kappa, std::mt19937 &rng)
{
    if (kappa < 1e-8) {
        std::uniform_real_distribution<float> unif(0.0, M_PI);
        return unif(rng);
    }

    const float mu = 2.0 * psi;
    const float kappa2 = 2.0 * kappa;

    std::uniform_real_distribution<float> unif(0.0, 1.0);

    const float a = 1.0 + std::sqrt(1.0 + 4.0 * kappa2 * kappa2);
    const float b = (a - std::sqrt(2.0 * a)) / (2.0 * kappa2);
    const float r = (1.0 + b * b) / (2.0 * b);

    float phi = 0.0;

    while(true) {
        const float u1 = unif(rng);
        const float z = std::cos(M_PI * u1);
        const float f = (1.0 + r * z) / (r + z);
        const float c = kappa2 * (r - f);

        const double u2 = unif(rng);
        if (u2 < c * (2.0 - c) || u2 <= c * std::exp(1.0 - c))
        {
            const double u3 = unif(rng);
            phi = mu + (u3 > 0.5 ?  std::acos(f)
                                 : -std::acos(f));
            break;
        }
    }

    phi = std::fmod(phi, 2.0 * M_PI);
    if (phi < 0.0) phi += 2.0 * M_PI;

    double theta = 0.5 * phi;
    if (theta >= M_PI) theta -= M_PI;

    return theta; 
}


inline void print_progress(int step, int total, int bar_width = 50)
{
    float frac   = float(step) / float(total);
    int   filled = int(frac * bar_width);

    std::cout << '\r' << '[';
    for (int i = 0; i < bar_width; ++i)
        std::cout << (i < filled ? '=' : ' ');
    std::cout << "] " << std::setw(3) << int(frac * 100.0f) << "%";
    std::cout.flush();
}


int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.toml>\n";
        return EXIT_FAILURE;
    }

    auto params = toml::parse(argv[1]);
    // Simulation parameters
    const int nx = toml::find<int>(params, "simulation", "nx"), ny = toml::find<int>(params, "simulation", "ny"), nt = toml::find<int>(params, "simulation", "nt"); // Grid size, number of timesteps
    const float dt = toml::find<float>(params, "simulation", "dt");                    // Timestep size
    const bool mechanics_on = toml::find<bool>(params, "simulation", "mechanics_on"); // Whether to run mechanics simulation
    const int mechanics_per_potential = toml::find<int>(params, "simulation", "mechanics_per_potential");
    const unsigned int seed = toml::find<int>(params, "simulation", "seed");
    const int simulation_id = toml::find<int>(params, "simulation", "simulation_id");

    // Reaction-Diffusion parameters
    const float eps0 =  toml::find<float>(params, "voltage", "eps0"), a =  toml::find<float>(params, "voltage", "a"), k =  toml::find<float>(params, "voltage", "k");
    const float D =  toml::find<float>(params, "voltage", "D"), mu1 =  toml::find<float>(params, "voltage", "mu1"), mu2 =  toml::find<float>(params, "voltage", "mu2");
    const float k_T =  toml::find<float>(params, "voltage", "k_T");

    // Mechanics parameters
    const float ks_edge = toml::find<float>(params, "mechanics", "ks_edge");  // Stiffness for axial springs
    const float ks_boundary = toml::find<float>(params, "mechanics", "ks_boundary");
    const float fiber_angle = toml::find<float>(params, "mechanics", "fiber_angle"); // radians
    const float fiber_correlation = toml::find<float>(params, "mechanics", "fiber_correlation"); // radians
    const float damping = toml::find<float>(params, "mechanics", "damping");
    const float ks_radial = toml::find<float>(params, "mechanics", "ks_radial");
    const int padding = toml::find<int>(params, "mechanics", "padding");
    const float fmax = toml::find<float>(params, "mechanics", "fmax");

    const float c_f = toml::find<float>(params, "mechanics", "c_f");

    const int snapshot_interval = toml::find<int>(params, "simulation", "snapshot_interval"); // Interval for saving snapshots

    std::seed_seq seq{ seed };
    std::array<std::uint32_t,2> subseeds;
    seq.generate(subseeds.begin(), subseeds.end());

    std::mt19937 focal_rng(subseeds[0]);
    std::mt19937 fiber_rng(subseeds[1]);

    // ---------- build the output folder name ----------
    std::filesystem::path base_dir = "data";
    std::filesystem::create_directories(base_dir);

    std::ostringstream folder_ss;
    folder_ss << simulation_id;

    
    std::filesystem::path out_path = base_dir / folder_ss.str();
    std::filesystem::create_directories(out_path);   // “data/…/” is now ready

    
    std::string out_dir = out_path.string();

    cudaStream_t ioStream;
    check(cudaStreamCreate(&ioStream));

    size_t N = static_cast<size_t>(nx) * ny;
    size_t bytes_rd = N * sizeof(float);
    size_t bytes_mech = N * sizeof(float2);

    // Pinned host memory buffers for asynchronous snapshot writing
    float *h_frame_u[2]; 
    check(cudaHostAlloc(&h_frame_u[0], bytes_rd, cudaHostAllocDefault));
    check(cudaHostAlloc(&h_frame_u[1], bytes_rd, cudaHostAllocDefault));

    float *h_frame_v[2]; 
    check(cudaHostAlloc(&h_frame_v[0], bytes_rd, cudaHostAllocDefault));
    check(cudaHostAlloc(&h_frame_v[1], bytes_rd, cudaHostAllocDefault));

    float2 *h_frame_mech[2];
    check(cudaHostAlloc(&h_frame_mech[0], bytes_mech, cudaHostAllocDefault));
    check(cudaHostAlloc(&h_frame_mech[1], bytes_mech, cudaHostAllocDefault));

    // RD Sim
    ReacDiffSim sim(nx, ny);
    float *h_u = new float[N];
    float *h_v = new float[N];
    float *h_Ta = new float[N];
    blank_initial(h_u, h_v, nx, ny);
    
    check(cudaMemcpy(sim.d_u, h_u, bytes_rd, cudaMemcpyHostToDevice));
    check(cudaMemcpy(sim.d_v, h_v, bytes_rd, cudaMemcpyHostToDevice));
    delete[] h_u;
    delete[] h_v;

    // ---- host: build fibre map for cells ----
    size_t C = static_cast<size_t>(nx-1)*(ny-1);
    std::vector<float> h_fiber(C);
    for (int j=0; j<ny-1; ++j)
        for (int i=0; i<nx-1; ++i) {
            size_t ci = j*(nx-1) + i;
            h_fiber[ci] = sampleVonMisesAxial(fiber_angle, fiber_correlation, fiber_rng);
            //h_fiber[ci] = fiber_angle;
    }

    std::filesystem::path fiber_file = "fibers.bin";
    std::ofstream fout(fiber_file, std::ios::binary);
    if (!fout) {
        throw std::runtime_error("Unable to open " + fiber_file.string());
    }
    fout.write(reinterpret_cast<char*>(h_fiber.data()),
               h_fiber.size() * sizeof(float));
    fout.close();

    MechSim mechSim(nx, ny, h_fiber.data(), damping, padding);


    
    // Time iter
    int buf_idx = 0;
    for (int t = 0; t <= nt; ++t)
    {
        if (t < nt / 2) sim.randomFocal(0.0005, focal_rng);
        sim.step(D, dt, eps0, a, k, mu1, mu2, k_T);

        if (mechanics_on){
            for (int t2 = 0; t2 < mechanics_per_potential; t2++) 
            {
                mechSim.step(dt / mechanics_per_potential, ks_edge, ks_radial, ks_boundary, sim.d_Ta, c_f, fmax);
            }
        }
        
        // Snapshot saving logic
        if (t % snapshot_interval == 0)
        {
            buf_idx = 1 - buf_idx; // Toggle buffer index


            // ---- async copy to host --------------------------------------
            check(cudaMemcpyAsync(h_frame_u[buf_idx],   // ← reuse old buffer
                                sim.d_u,
                                bytes_rd,              // N * sizeof(float)
                                cudaMemcpyDeviceToHost,
                                ioStream));

            check(cudaMemcpyAsync(
                h_frame_v[buf_idx],
                sim.d_v,
                bytes_rd,
                cudaMemcpyDeviceToHost,
                ioStream));
            
            check(cudaMemcpyAsync(
                h_frame_mech[buf_idx],
                mechSim.d_pos_c,
                bytes_mech,
                cudaMemcpyDeviceToHost,
                ioStream));

            check(cudaStreamSynchronize(ioStream));      // wait right here

            std::thread([t, buf_idx, bytes_rd, bytes_mech, h_frame_u, h_frame_v, h_frame_mech, ioStream, out_dir]() mutable // Capture mutable ioStream
                {
                    check(cudaStreamSynchronize(ioStream)); // Wait for copies on ioStream to finish

                    
                    std::string filename_u = out_dir + "/u_" + std::to_string(t) + ".bin";
                    std::ofstream outU(filename_u, std::ios::binary);
                    if(outU) {
                        outU.write(reinterpret_cast<char*>(h_frame_u[buf_idx]), bytes_rd);
                    } else {
                         std::cerr << "Error opening file: " << filename_u << std::endl;
                    }

                    std::string filename_v = out_dir + "/v_" + std::to_string(t) + ".bin";
                    std::ofstream outV(filename_v, std::ios::binary);
                    if(outV) {
                        outV.write(reinterpret_cast<char*>(h_frame_v[buf_idx]), bytes_rd);
                    } else {
                         std::cerr << "Error opening file: " << filename_v << std::endl;
                    }

                    std::string filename_x = out_dir + "/x_" + std::to_string(t) + ".bin";
                    std::ofstream outX(filename_x, std::ios::binary);
                     if(outX) {
                        outX.write(reinterpret_cast<char*>(h_frame_mech[buf_idx]), bytes_mech);
                    } else {
                        std::cerr << "Error opening file: " << filename_x << std::endl;
                    }
                }).detach();

            print_progress(t, nt);
        }
    }

    std::cout << std::endl;  

    // Cleanup
    check(cudaDeviceSynchronize());
    check(cudaStreamSynchronize(ioStream));


    cudaFreeHost(h_frame_u[0]);
    cudaFreeHost(h_frame_u[1]);
    cudaFreeHost(h_frame_v[0]);
    cudaFreeHost(h_frame_v[1]);
    cudaFreeHost(h_frame_mech[0]);
    cudaFreeHost(h_frame_mech[1]);

    check(cudaStreamDestroy(ioStream));

    std::cout << "Simulation finished." << std::endl;
    return 0;
}