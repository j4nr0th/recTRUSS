from numpy import np


def min_area(yield_stress, safety_factor, force):
    min_area = force * safety_factor / yield_stress
    return min_area


def rod_mass(area, length, density):
    return area * length * density

