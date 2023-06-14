from point import Point
from profile import Profile
import numpy as np


def mmoi_structure(node1: Point, node2: Point, profile: Profile, mass: float, axis: tuple[float, ...]):
    dx, dy, dz = node2.x - node1.x, node2.y - node1.y, node2.z - node1.z
    length = np.linalg.norm([dx, dy, dz])
    distances_to_axis = np.array((node1.x + dx/2, node1.y + dy/2)) - np.array((axis[0], axis[1]))
    dist_axis = np.linalg.norm(distances_to_axis)
    mmoi = 0
    # I wish I could find a nice universal formula, I don't want to derive it myself :'(
    if np.abs(dx) < 0.0005 and abs(dy) < 0.0005:
        # Fully vertical members
        mmoi = mass * (profile.r ** 2 + (profile.r - profile.t) ** 2) / 2

    elif np.abs(dz) > 0.0005:
        # diagonals that go up vertically
        if np.abs(dx) > 0.0005:
            theta = np.arctan2(dz, dx)
        else:
            theta = np.arctan2(dz, dy)
        mmoi = mass * length ** 2 * np.sin(theta) ** 2 / 12
    elif np.abs(dz) < 0.0005:
        # fully horizontal members
        mmoi = mass * length ** 2 / 12

    mmoi += mass * dist_axis ** 2
    return mmoi


def mmoi_drivetrain(pts: list[Point], tot_dt_mass, drivetrain_count, column_count):
    max_y = max([point.y for point in pts])
    column_width = max_y / column_count
    rotors_per_column = drivetrain_count / column_count
    rotor_diameter = column_width / rotors_per_column
    drivetrain_locations = np.arange(rotor_diameter/2, max_y - rotor_diameter /2 + 1, rotor_diameter)
    mmoi = (tot_dt_mass / drivetrain_count) * np.sum(drivetrain_locations ** 2)
    return mmoi

