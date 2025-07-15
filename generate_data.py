import subprocess, sys
import os
import random, math

class Simulation:
    def __init__(self, simulation_id: int,
                 seed: int,
                 nx: int, 
                 ny: int, 
                 dt: float,
                 nt: int, 
                 mechanics_on: bool, 
                 mechanics_per_potential: int,
                 snapshot_interval: int):
        self.simulation_id = simulation_id
        self.seed = seed
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.nt = nt
        self.mechanics_on = mechanics_on
        self.mechanics_per_potential = mechanics_per_potential
        self.snapshot_interval = snapshot_interval
    
    def __str__(self):
        lines = [
            "[simulation]",
            f"simulation_id = {self.simulation_id}",
            f"seed = {self.seed}",
            f"nx = {self.nx}",
            f"ny = {self.ny}",
            f"dt = {self.dt}",
            f"nt = {self.nt}",
            f"mechanics_on = " + str(self.mechanics_on).lower(),
            f"mechanics_per_potential = {self.mechanics_per_potential}",
            f"snapshot_interval = {self.snapshot_interval}",
        ]
        return "\n".join(lines)

class Voltage:
    D_range = (0.0001, 0.01)
    def __init__(self, D: float,
                 eps0: float,
                 a: float,
                 k: float,
                 mu1: float,
                 mu2: float,
                 k_T: float):
        self.D = D
        self.eps0 = eps0
        self.a = a
        self.k = k
        self.mu1 = mu1
        self.mu2 = mu2
        self.k_T = k_T

    def __str__(self):
        lines = [
            "[voltage]",
            f"D = {self.D}",
            f"eps0 = {self.eps0}",
            f"a = {self.a}",
            f"k = {self.k}",
            f"mu1 = {self.mu1}",
            f"mu2 = {self.mu2}",
            f"k_T = {self.k_T}"
        ]
        return "\n".join(lines)

class Mechanics:
    def __init__(self, ks_edge: float,
                 ks_radial: float,
                 ks_boundary: float,
                 rest_edge_length: float,
                 fiber_angle: float,
                 fiber_correlation: float,
                 damping: float,
                 c_f: float,
                 padding: float,
                 fmax: float):
        self.ks_edge = ks_edge
        self.ks_radial = ks_radial
        self.ks_boundary = ks_boundary
        self.rest_edge_length = rest_edge_length
        self.fiber_angle = fiber_angle
        self.fiber_correlation = fiber_correlation
        self.damping = damping
        self.c_f = c_f
        self.padding = padding
        self.fmax = fmax

    def __str__(self):
        lines = [
            "[mechanics]",
            f"ks_edge = {self.ks_edge}",
            f"ks_radial = {self.ks_radial}",
            f"ks_boundary = {self.ks_boundary}",
            f"rest_edge_length = {self.rest_edge_length}",
            f"fiber_angle = {self.fiber_angle}",
            f"fiber_correlation = {self.fiber_correlation}",
            f"damping = {self.damping}",
            f"c_f = {self.c_f}",
            f"padding = {self.padding}",
            f"fmax = {self.fmax}"
        ]
        return "\n".join(lines)


class Config:
    def __init__(self, simulation: Simulation, 
                 voltage: Voltage,
                 mechanics: Mechanics):
        self.simulation = simulation
        self.voltage = voltage
        self.mechanics = mechanics

    def __str__(self):
        lines = [
            str(self.simulation),
            "\n",
            str(self.voltage),
            "\n",
            str(self.mechanics)
        ]
        return "\n".join(lines)
    
    def save_toml(self, dirname: str = "config"):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = self.simulation.simulation_id
        with open(f"{dirname}/{filename}.toml", "w") as f:
            f.write(str(self))

num_simulations = 500
random.seed(-1)

for i in range(num_simulations):
    print(f"Creating configuration file for simulation {i + 1}/{num_simulations}...")
    sim = Simulation(simulation_id = i,
                    seed = random.randint(0, 2**31 - 1),
                    nx = 256, 
                    ny = 256, 
                    dt = 0.01, 
                    nt = 100000, 
                    mechanics_on = True, 
                    mechanics_per_potential = 10,
                    snapshot_interval = 50)

    volt = Voltage(D = 0.03,
                eps0 = 0.01,
                a = 0.1035,
                k = 8.0,
                mu1 = 0.15,
                mu2 = 0.15,
                k_T = 3.0)

    mech = Mechanics(ks_edge = 20.0,
                    ks_radial = 30.0,
                    ks_boundary = 20.0,
                    rest_edge_length = 1.0,
                    fiber_angle = random.uniform(0.0, math.pi),
                    fiber_correlation = 5.0,
                    damping = 2.0,
                    c_f = 0.1,
                    padding = 2,
                    fmax = 0.1)

    config = Config(simulation=sim, voltage=volt, mechanics=mech)
    config.save_toml("config")
    print("Config file saved!")

    print(f"Running simulation {i + 1}/{num_simulations}...")

    config_file = f"config/{i}.toml"
    subprocess.run(['./sim', config_file], check=True)

    print(f"Simulation {i + 1}/{num_simulations} completed!\n")

