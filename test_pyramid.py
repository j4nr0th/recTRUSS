import sys
import pandas as pd
import unittest
import numpy as np
from scipy.optimize import fsolve
from main import main
from point import load_points_from_file
from profile import load_profiles_from_file
from material import load_materials_from_file


class MyTestCase(unittest.TestCase):
    test_files = 'tests/structure1'
    F = float(pd.read_csv(test_files + '.nat')['Fz'])

    points = load_points_from_file(test_files + '.pts')
    E = load_materials_from_file('tests/sample.mat')[0].E
    a, b, c, d, e = points
    La = np.linalg.norm(((e.x - a.x), (e.y - a.y), (e.z - a.z)))
    Lb = np.linalg.norm(((e.x - b.x), (e.y - b.y), (e.z - b.z)))
    Lc = np.linalg.norm(((e.x - c.x), (e.y - c.y), (e.z - c.z)))
    Ld = np.linalg.norm(((e.x - d.x), (e.y - d.y), (e.z - d.z)))
    pf = load_profiles_from_file(test_files + '.pro')[0]
    Aa = np.pi * (pf.r ** 2 - (pf.r - pf.t) ** 2)
    Ab, Ac, Ad = Aa, Aa, Aa


    def find_E_solution(self, xe):
        Fa = (np.sqrt((xe[0] - self.a.x) ** 2 + (xe[1] - self.a.y) ** 2 + (xe[2] - self.a.z) ** 2) - self.La) * self.E * self.Aa / self.La
        Fb = (np.sqrt((xe[0] - self.b.x) ** 2 + (xe[1] - self.b.y) ** 2 + (xe[2] - self.b.z) ** 2) - self.Lb) * self.E * self.Ab / self.Lb
        Fc = (np.sqrt((xe[0] - self.c.x) ** 2 + (xe[1] - self.c.y) ** 2 + (xe[2] - self.c.z) ** 2) - self.Lc) * self.E * self.Ac / self.Lc
        Fd = (np.sqrt((xe[0] - self.d.x) ** 2 + (xe[1] - self.d.y) ** 2 + (xe[2] - self.d.z) ** 2) - self.Ld) * self.E * self.Ad / self.Ld

        return np.array([-Fa * 3 / self.La + Fb / self.Lb + Fc / self.Lc - 3 * Fd / self.Ld,
                -Fa / self.La - Fb / self.Lb + 3 * Fc / self.Lc + 3 * Fd / self.Ld,
                -self.F + Fa / self.La + Fb / self.Lb + Fc / self.Lc + Fd / self.Ld])

    def find_analytical_sol(self):
        xe = fsolve(self.find_E_solution, x0=np.array([3, 1, 1]), xtol=sys.float_info.epsilon * 100)
        assert np.isclose(self.find_E_solution(xe), np.array([0,0,0])).all()
        Fa = (np.sqrt((xe[0] - self.a.x) ** 2 + (xe[1] - self.a.y) ** 2 + (xe[2] - self.a.z) ** 2) - self.La) * self.E * self.Aa / self.La
        Fb = (np.sqrt((xe[0] - self.b.x) ** 2 + (xe[1] - self.b.y) ** 2 + (xe[2] - self.b.z) ** 2) - self.Lb) * self.E * self.Ab / self.Lb
        Fc = (np.sqrt((xe[0] - self.c.x) ** 2 + (xe[1] - self.c.y) ** 2 + (xe[2] - self.c.z) ** 2) - self.Lc) * self.E * self.Ac / self.Lc
        Fd = (np.sqrt((xe[0] - self.d.x) ** 2 + (xe[1] - self.d.y) ** 2 + (xe[2] - self.d.z) ** 2) - self.Ld) * self.E * self.Ad / self.Ld
        return Fa, Fb, Fc, Fd


    def test_pyramid(self):
        stresses = main(self.test_files, 0, 0, optimizing=False, printing=False, plotting=False, gravity=False)
        analytical_stress = np.array(self.find_analytical_sol()) / np.array([self.Aa, self.Ab, self.Ac, self.Ad])
        assert np.isclose(stresses, analytical_stress).all()