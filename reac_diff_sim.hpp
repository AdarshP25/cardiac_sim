#pragma once
#include <cuda_runtime.h>
#include <random>

struct ReacDiffSim
{
    int nx, ny;
    float *d_u, *d_v, *d_Ta, *d_u_new, *d_v_new, *d_Ta_new, *d_lap_u;

    ReacDiffSim(int nx, int ny);
    ~ReacDiffSim();
    void step(float D, float dt, float eps0, float a, float k, float mu1, float mu2, float k_T);
    void randomFocal(float prob, std::mt19937 &rng);
};
