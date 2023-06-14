from full_structure_generation import generate_structure
import pandas as pd


def test_applied_forces():
    cell_file_name = "test_applied_forces/structure1"
    cell_rows, cell_columns = 2, 2
    total_thrust_rotors = 10E3
    total_generator_mass = 2E3
    total_rotor_mass = 1E3
    total_height, total_width, total_depth = 40, 40, 20
    constrained_points = ('A0000', 'B0000', 'A0100', 'B0100')
    additional_loads = [(0, 'B', 'x', -3E3),
                        (0,'A','z',-500)]

    generate_structure(cell_file_name, layers=cell_rows, columns=cell_columns,
                       total_rotor_thrust=total_thrust_rotors, total_gen_mass=total_generator_mass,
                       total_rotor_mass=total_rotor_mass, additional_loads=additional_loads,
                       height=total_height, width=total_width, depth=total_depth,
                       constrained_points=constrained_points)

    forces = pd.read_csv(cell_file_name + "_fullstruct.nat", index_col=0)
    forces = forces.astype(float)
    assert abs(forces.sum()).sum() - (total_thrust_rotors + total_generator_mass * 9.81 + total_rotor_mass * 9.81 + 3E3 + 500) <= 1E-10