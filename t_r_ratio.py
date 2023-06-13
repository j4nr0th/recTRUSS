from generate_profiles import generate_profiles
from full_structure_generation import generate_structure
from main import main
import numpy as np
import matplotlib.pyplot as plt


def get_t_r_curve(thickness_ratios):
    cell_file_name = "2_not_j_thickness_variation/structure1"
    file_loc = cell_file_name + "_fullstruct"
    mass_thickness_data = np.empty((len(thickness_ratios), 4))
    for i in range(len(thickness_ratios)):
        print(thickness_ratios[i])
        generate_profiles(0.15, 2, 20, thickness_ratios[i], file_loc + '.pro')

        # Dimensions
        total_height, total_width, total_depth = 280, 280, 33
        cell_rows, cell_columns = 6, 3

        # Rotor shit for natural frequency
        rotors_per_cell = 2
        rotor_diameter = total_width / 2 / cell_columns / rotors_per_cell
        TSR, v_rated = 4.5, 11.2
        rotor_frequency = (TSR * v_rated) / (rotor_diameter * np.pi)

        # Applied forces FOR HALF THE STRUCTURE!!!
        total_thrust_rotors = 8.53E6 / 2 / 2
        gen_count = rotors_per_cell * cell_columns
        total_generator_mass = 60E3 * 6
        # 30 tons rotors + 500 tons shafts
        total_rotor_mass = 300E3 / 9.81 + 136E3 * cell_columns
        torque = 0  # 3E6
        I_z_estimate = 150E9
        skipped_rows = 2
        total_HLD_downforce = 0.778E6 * (cell_rows - skipped_rows) * cell_columns
        total_HLD_thrust = total_HLD_downforce * 0.25

        constrained_points = ('A0000', 'B0000', 'A0003', 'B0003')

        HLD_rows = cell_rows - skipped_rows + 1
        additional_loads = [(i, 'B', 'x', -total_HLD_thrust / HLD_rows) for i in
                            range(skipped_rows, HLD_rows + skipped_rows)]
        additional_loads += [(i, 'B', 'z', -total_HLD_downforce / HLD_rows) for i in
                             range(skipped_rows, HLD_rows + skipped_rows)]

        generate_structure(cell_file_name, layers=cell_rows, columns=cell_columns,
                           total_rotor_thrust=total_thrust_rotors, total_gen_mass=total_generator_mass,
                           total_rotor_mass=total_rotor_mass, additional_loads=additional_loads,
                           height=total_height, width=total_width, depth=total_depth,
                           constrained_points=constrained_points)
        main(file_loc, drive_train_count=gen_count, gen_mass=total_generator_mass, rotor_mass=total_rotor_mass,
             optimizing=True)
        mass, freq = main(file_loc, drive_train_count=gen_count, gen_mass=total_generator_mass,
                          rotor_mass=total_rotor_mass, optimizing=False, printing=False, plotting=False)

        frequencies = sorted(freq)
        mass_thickness_data[i,:] = [thickness_ratios[i], mass, frequencies[0], frequencies[1]]

    return mass_thickness_data


if __name__ == '__main__':
    thickness_ratio_range = np.linspace(0.001, 0.1, 20)
    resulting_mass = get_t_r_curve(thickness_ratio_range)

    plt.plot(resulting_mass[:, 0], 2 * resulting_mass[:, 1]/1000)
    plt.plot([1/240, 1/240], [0, max(2*resulting_mass[:, 1]/1000)])
    plt.ylabel('Structural mass [tonnes]')
    plt.xlabel('Thickness to radius ratio [-]')
    plt.show()

    plt.plot(resulting_mass[:, 0], resulting_mass[:, 2], label='Natural frequency 1')
    plt.plot(resulting_mass[:, 0], resulting_mass[:, 3], label='Natural frequency 2')
    plt.plot([1/240, 1/240], [0, max(2*resulting_mass[:, 1]/1000)])

    plt.ylabel('Natural frequency [Hz]')
    plt.xlabel('Thickness to radius ratio [-]')
    plt.title("Lowest natural frequencies to thickness variation")
    plt.show()
