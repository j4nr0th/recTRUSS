from matplotlib import pyplot as plt
import numpy as np
from full_structure_generation import generate_structure, add_torque
import sigfig as sig
from generate_profiles import generate_profiles
from main import main

def get_depth_ratio_mass(depth_ratios, nat_freqs):
    cell_file_name = "2_not_j_depth_variation/structure1"
    file_loc = cell_file_name + "_fullstruct"
    depth_data = np.empty((len(depth_ratios), 2 + nat_freqs))
    for i in range(len(depth_ratios)):

        optimizing = True
        # Dimensions
        total_height, total_width = 280, 280
        total_depth = total_width * depth_ratios[i]
        cell_rows, cell_columns = 6, 3

        # Rotor shit for natural frequency
        rotors_per_cell = 2
        rotor_diameter = total_width / 2 / cell_columns / rotors_per_cell
        TSR, v_rated = 4.5, 11.2
        rotor_frequency = (TSR * v_rated) / (rotor_diameter * np.pi)
        print("Rotor frequency:", rotor_frequency)

        # Applied forces FOR HALF THE STRUCTURE!!!
        total_thrust_rotors = 8.53E6 / 2 / 2 * 11.2 ** 2 / 15 ** 2
        gen_count = rotors_per_cell * cell_columns
        total_generator_mass = 60E3 * 6
        # 30 tons rotors + 136 tons per shaft
        total_rotor_mass = 300E3 / 9.81 + 136E3 * cell_columns * rotors_per_cell
        # Torque 3MN due to electric motor
        # time_to_turn_90 = 60 * 60
        # angular_acc = 0.5 * np.pi / time_to_turn_90 ** 2
        torque = 0  # 3E6
        I_z_estimate = 150E9
        angular_acc = torque / I_z_estimate

        # storm conditions : 0 MN * 4 in downforce, 1.6E MN per cell thrust
        # HLD operational 0.778E6 * (cell_rows - skipped_rows) * cell_columns
        skipped_rows = 2
        total_HLD_downforce = 0.778E6 * (cell_rows - skipped_rows) * cell_columns
        total_HLD_thrust = total_HLD_downforce * 0.25

        constrained_points = ('A0000', 'B0000', 'A0004', 'B0004')
        # constrained_points = ('A0100', 'A0300', 'B0100', 'B0300')

        HLD_rows = cell_rows - skipped_rows + 1
        additional_loads = [(i, 'B', 'x', -total_HLD_thrust / HLD_rows) for i in
                            range(skipped_rows, HLD_rows + skipped_rows)]
        additional_loads += [(i, 'B', 'z', -total_HLD_downforce / HLD_rows) for i in
                             range(skipped_rows, HLD_rows + skipped_rows)]

        if optimizing:
            if angular_acc != 0:
                generate_structure(cell_file_name, layers=cell_rows, columns=cell_columns,
                                   total_rotor_thrust=total_thrust_rotors, total_gen_mass=total_generator_mass,
                                   total_rotor_mass=total_rotor_mass, additional_loads=additional_loads,
                                   height=total_height, width=total_width, depth=total_depth,
                                   constrained_points=constrained_points)

                it_limit = 20
                for j in range(it_limit):
                    print(j)
                    add_torque(filename_full=file_loc, ang_acc=angular_acc,
                               tot_gen_mass=total_generator_mass, tot_rotor_mass=total_rotor_mass,
                               axis=(total_depth / 2, 0, 0))
                    main(file_loc, drive_train_count=gen_count, gen_mass=total_generator_mass,
                         rotor_mass=total_rotor_mass,
                         optimizing=True, torque=True)

                mass, freq = main(file_loc, drive_train_count=gen_count, gen_mass=total_generator_mass,
                                  rotor_mass=total_rotor_mass, optimizing=False, torque=True)

            else:
                generate_structure(cell_file_name, layers=cell_rows, columns=cell_columns,
                                   total_rotor_thrust=total_thrust_rotors, total_gen_mass=total_generator_mass,
                                   total_rotor_mass=total_rotor_mass, additional_loads=additional_loads,
                                   height=total_height, width=total_width, depth=total_depth,
                                   constrained_points=constrained_points)
                main(file_loc, drive_train_count=gen_count, gen_mass=total_generator_mass, rotor_mass=total_rotor_mass,
                     optimizing=True)
                mass, freq = main(file_loc, drive_train_count=gen_count, gen_mass=total_generator_mass,
                                  rotor_mass=total_rotor_mass, optimizing=False, plotting=False, printing=False)

        else:
            if angular_acc != 0:
                mass, freq = main(file_loc, drive_train_count=gen_count, gen_mass=total_generator_mass,
                                  rotor_mass=total_rotor_mass, optimizing=False, torque=True)
            else:
                mass, freq = main(file_loc, drive_train_count=gen_count, gen_mass=total_generator_mass,
                                  rotor_mass=total_rotor_mass, optimizing=False)

        if optimizing:
            with open(file_loc + ".results", 'a') as f:
                f.write(
                    f"{cell_file_name},{round(mass, 0)},{cell_rows},{cell_columns * 2},{total_width},{total_height},{total_depth}"
                    f",{sig.round(freq[0], 3)},{sig.round(freq[1], 3)},{sig.round(freq[2], 3)}\n")

        print(f"Depth ratio: {round(depth_ratios[i],2)} - Total depth: {round(total_depth, 2)}m - Structure mass: {round(2*mass/1000,0)} tonnes")
        frequencies = sorted(freq)
        depth_data[i,:] = [depth_ratios[i], mass, *frequencies[0:nat_freqs]]

    return depth_data


if __name__ == '__main__':
    depth_ratio_range = np.linspace(0.01, 0.15, 30)
    nat_freq_nr = 4
    resulting_mass = get_depth_ratio_mass(depth_ratio_range, nat_freq_nr)

    plt.plot(resulting_mass[:, 0], 2 * resulting_mass[:, 1]/1000)
    plt.plot([1/(2*3*2*2), 1/(2*3*2*2)], [0, max(2*resulting_mass[:, 1]/1000)])
    plt.ylim(min(2*resulting_mass[:, 1]/1000) /1.5, max(2*resulting_mass[:, 1]/1000)*1.1)
    plt.ylabel('Structural mass [tonnes]')
    plt.xlabel('Depth to width ratio [-]')
    plt.title("Structural mass due to depth variation")
    plt.show()
    for i in range(3,resulting_mass.shape[1]):
        plt.plot(resulting_mass[:, 0], resulting_mass[:, i])

    plt.plot([1 / (2 * 3 * 2 * 2), 1 / (2 * 3 * 2 * 2)], [0, np.max(resulting_mass[:,3:])])
    plt.ylabel('Natural frequency [Hz]')
    plt.xlabel('Depth to width ratio [-]')
    plt.title("Lowest natural frequencies to depth variation")
    plt.show()