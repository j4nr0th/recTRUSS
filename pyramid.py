import pandas as pd
import unittest
import numpy as np
from scipy.optimize import fsolve
from main import main
from point import load_points_from_file
from material import load_materials_from_file
from profile import load_profiles_from_file




test_files = 'tests/structure1'
F = pd.read_csv(test_files + '.nat')['Fz'][0].astype(float)

points = load_points_from_file(test_files + '.pts')
E = load_materials_from_file('tests/sample.mat')[0].E
a, b, c, d, e = points
La = np.linalg.norm(((e.x - a.x), (e.y - a.y), (e.z - a.z)))
Lb = np.linalg.norm(((e.x - b.x), (e.y - b.y), (e.z - b.z)))
Lc = np.linalg.norm(((e.x - c.x), (e.y - c.y), (e.z - c.z)))
Ld = np.linalg.norm(((e.x - d.x), (e.y - d.y), (e.z - d.z)))
pf = load_profiles_from_file(test_files + '.pro')[0]
Aa = np.pi * (pf.r**2 - (pf.r-pf.t)**2)
Ab, Ac, Ad = Aa, Aa, Aa

# analytical_stress = analytical_sol / Aa
stresses = main(test_files, 0, 0, optimizing=False, gravity=False)

def find_E_solution(xe):
    Fa = (np.sqrt((xe[0] - a.x) ** 2 + (xe[1] - a.y) ** 2 + (xe[2] - a.z) ** 2) - La) * E * Aa / La
    Fb = (np.sqrt((xe[0] - b.x) ** 2 + (xe[1] - b.y) ** 2 + (xe[2] - b.z) ** 2) - Lb) * E * Ab / Lb
    Fc = (np.sqrt((xe[0] - c.x) ** 2 + (xe[1] - c.y) ** 2 + (xe[2] - c.z) ** 2) - Lc) * E * Ac / Lc
    Fd = (np.sqrt((xe[0] - d.x) ** 2 + (xe[1] - d.y) ** 2 + (xe[2] - d.z) ** 2) - Ld) * E * Ad / Ld

    return np.array([-Fa / La + 3 * Fb / Lb - 1 * Fc / Lc + 3 * Fd / Ld,
                -Fa / La - 1 * Fb / Lb + 3 * Fc / Lc + 3 * Fd / Ld,
                F - Fa / La - Fb / Lb - Fc / Lc - Fd / Ld])

def find_analytical_sol():
    xe = fsolve(find_E_solution, x0=np.array([3,1,1]))
    Fa = (np.sqrt((xe[0] - a.x) ** 2 + (xe[1] - a.y) ** 2 + (xe[2] - a.z) ** 2) - La) * E * Aa / La
    Fb = (np.sqrt((xe[0] - b.x) ** 2 + (xe[1] - b.y) ** 2 + (xe[2] - b.z) ** 2) - Lb) * E * Ab / Lb
    Fc = (np.sqrt((xe[0] - c.x) ** 2 + (xe[1] - c.y) ** 2 + (xe[2] - c.z) ** 2) - Lc) * E * Ac / Lc
    Fd = (np.sqrt((xe[0] - d.x) ** 2 + (xe[1] - d.y) ** 2 + (xe[2] - d.z) ** 2) - Ld) * E * Ad / Ld
    return Fa, Fb, Fc, Fd

# def test_simple_sym_pyramid():
#     forces = np.array(find_analytical_sol())
#     analytical_sol = forces / np.array([Aa, Ab, Ac, Ad])
#     # assert np.isclose(analytical_sol, stresses).all()

analytical_stress = find_analytical_sol() / np.array([Aa, Ab, Ac, Ad])


