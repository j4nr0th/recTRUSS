from matplotlib import pyplot as plt
import copy
import numpy as np
from element import elements_assemble
from plotting import show_structure, show_deformed, show_forces, show_masses
from point import Point, load_points_from_file
from material import Material, load_materials_from_file
from profile import Profile, load_profiles_from_file
from connection import Connection, load_connections_from_file, write_connections_to_file
from bcs import BoundaryCondition, load_natural_bcs, load_numerical_bcs
from generate_profiles import smaller_profile, larger_profile
from full_structure_generation import generate_structure, add_torque
from mmoi_z import mmoi_structure, mmoi_drivetrain


def compute_global_to_local_transform(dx: float, dy: float, dz: float) -> np.ndarray:
    alpha = np.arctan2(dy, dx)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    beta = -(np.pi / 2 - np.arctan2(np.hypot(dx, dy), dz))
    cb = np.cos(beta)
    sb = np.sin(beta)
    T_z = np.array(
        [[ca, sa, 0],
         [-sa, ca, 0],
         [0, 0, 1]]
    )
    T_y = np.array(
        [[cb, 0, -sb],
         [0, 1, 0],
         [sb, 0, cb]]
    )
    T = (T_y @ T_z)
    d = np.array((dx, dy, dz))
    assert np.all(np.isclose(T @ (d / np.linalg.norm(d)), np.array([1, 0, 0])))
    return T


def compute_element_stiffness_matrix(T: np.ndarray, k: float) -> np.ndarray:
    K_element = np.array(
        [[+1, 0, 0, -1, 0, 0],
         [+0, 0, 0, +0, 0, 0],
         [+0, 0, 0, +0, 0, 0],
         [-1, 0, 0, +1, 0, 0],
         [+0, 0, 0, +0, 0, 0],
         [+0, 0, 0, +0, 0, 0]]
    )

    return k * (T.T @ K_element @ T)


def save_displacements_to_file(filename: str, nodes: list["Point"], u: np.ndarray, r: np.ndarray):
    u = u.flatten()
    r = r.flatten()
    u[np.isclose(u, 0)] = 0.0
    r[np.isclose(r, 0)] = 0.0
    with open(filename, "w") as f_out:
        f_out.write(f"point label,ux,uy,uz,Rx,Ry,Rz\n")
        for i, pt in enumerate(nodes):
            f_out.write(
                f"{pt.label},{u[3 * i + 0]},{u[3 * i + 1]},{u[3 * i + 2]},{r[3 * i + 0]},{r[3 * i + 1]},{r[3 * i + 2]}\n")


def main(file_loc, drive_train_count, tot_dt_mass, optimizing=True, plotting=True, printing=True, gravity=True, torque=False):

    if optimizing:
        plotting = False
        printing = False

    abs_plotting = False
    #   Load bare data from files

    node_list = load_points_from_file(file_loc + ".pts")
    material_list = load_materials_from_file(file_loc.split('/')[0] + '/sample.mat')

    profile_loc = file_loc + ".pro"
    connection_loc = file_loc + ".con"
    profile_list = load_profiles_from_file(profile_loc)
    if not torque:
        natural_bc_list = load_natural_bcs(file_loc + ".nat", node_list)
    else:
        natural_bc_list = load_natural_bcs(file_loc + "_torqued" + ".nat", node_list)
    numerical_bc_list = load_numerical_bcs(file_loc + ".num", node_list)
    x_vals = {point.x for point in node_list}
    structural_depth = abs(max(x_vals) - min(x_vals))

    running = True
    i=0
    it_limit = len(profile_list) * 3
    while running and i < it_limit:
        i+=1
        if not optimizing:
            running = False
        connection_list = load_connections_from_file(connection_loc)
        old_connection_list = copy.deepcopy(connection_list)
        #   Assemble elements from nodes, materials, profiles, and connections
        elements = elements_assemble(connection_list, material_list, profile_list, node_list)
        n = len(node_list)
        n_dof = 3 * n
        MMOIz = []
        if tot_dt_mass > 0 and total_rotor_mass > 0:
            MMOIz.append(mmoi_drivetrain(node_list, tot_dt_mass, drive_train_count))
        approx_mmoiz = 0

        if plotting:
            fig = show_structure(node_list, elements, numerical_bc_list, natural_bc_list)
            fig.suptitle("Problem setup")
            plt.show()

        #   Now assemble the global system
        K_g = np.zeros((n_dof, n_dof))
        M_g = np.zeros((n_dof, n_dof))
        f_g = np.zeros((n_dof, 1))
        u_g = np.zeros((n_dof, 1))
        free_dofs = np.ones(n_dof, dtype=bool)
        for i, e in enumerate(elements):
            #   Build transformation matrix
            n1 = node_list[e.node1]
            n2 = node_list[e.node2]
            m = material_list[e.material]
            p = profile_list[e.profile]
            dx = n2.x - n1.x
            dy = n2.y - n1.y
            dz = n2.z - n1.z

            d = np.array(((dx,), (dy,), (dz,)))
            L = np.linalg.norm(d)

            T_one = compute_global_to_local_transform(dx, dy, dz)
            assert np.all(np.isclose(np.array(((1,), (0,), (0,))), T_one @ (d / L)))
            assert np.isclose(np.linalg.det(T_one), 1)
            T = np.zeros((6, 6))
            T[0:3, 0:3] = T_one
            T[3:6, 3:6] = T_one
            assert np.isclose(np.linalg.det(T), 1)

            #   Build local stiffness matrix
            E = m.E
            A = (p.r ** 2 - (p.r - p.t) ** 2) * np.pi
            #   Transform it to global coordinate frame
            K_e = compute_element_stiffness_matrix(T, A * E / L)
            #   Insert it into global matrix
            indices = (3 * e.node1, 3 * e.node1 + 1, 3 * e.node1 + 2, 3 * e.node2, 3 * e.node2 + 1, 3 * e.node2 + 2)
            K_g[np.ix_(indices, indices)] += K_e
            mass = m.rho * A * L
            #   Add gravitational force
            if gravity:
                f_g[3 * e.node1 + 2] += -0.5 * m.rho * A * L * 9.81
                f_g[3 * e.node2 + 2] += -0.5 * m.rho * A * L * 9.81
            #   Add temperature
            # F_thermal = np.abs(n1.t - n2.t) * E * A * m.alpha / (2 * L)
            # f_g[3 * e.node2: 3 * e.node2 + 3] += d * F_thermal
            # f_g[3 * e.node1: 3 * e.node1 + 3] += -d * F_thermal

            M_g[indices, indices] += mass / 2
            # M_e = mass / 6 * np.array(
            #     [[2, 0, 0, 1, 0, 0],
            #      [0, 2, 0, 0, 1, 0],
            #      [0, 0, 2, 0, 0, 1],
            #      [1, 0, 0, 2, 0, 0],
            #      [0, 1, 0, 0, 2, 0],
            #      [0, 0, 1, 0, 0, 2]])
            # M_g[np.ix_(indices, indices)] += M_e
            MMOIz.append(mmoi_structure(n1, n2, p, mass, (structural_depth/2, 0., 0.)))
            approx_mmoiz += mass / 2 * np.linalg.norm(((n1.x - structural_depth / 2), (n1.y - structural_depth / 2))) ** 2
            approx_mmoiz += mass / 2 * np.linalg.norm(((n2.x - structural_depth / 2), (n2.y - structural_depth / 2))) ** 2
            #   Apply numerical BCs
        for i, bc in enumerate(numerical_bc_list):
            pt_index = bc.node
            node = node_list[bc.node]
            if bc.x is not None:
                free_dofs[3 * pt_index + 0] = 0
                u_g[3 * pt_index + 0] = bc.x - node.x
            if bc.y is not None:
                free_dofs[3 * pt_index + 1] = 0
                u_g[3 * pt_index + 1] = bc.y - node.y
            if bc.z is not None:
                free_dofs[3 * pt_index + 2] = 0
                u_g[3 * pt_index + 2] = bc.z - node.z

        #   Apply natural BCs
        for i, bc in enumerate(natural_bc_list):
            pt_index = bc.node
            if bc.x is not None:
                f_g[3 * pt_index + 0] += bc.x
            if bc.y is not None:
                f_g[3 * pt_index + 1] += bc.y
            if bc.z is not None:
                f_g[3 * pt_index + 2] += bc.z

        #   Reduce the problem to only free DoFs
        K_r = K_g[np.ix_(free_dofs, free_dofs)]
        u_r = u_g[free_dofs]
        f_r = f_g[free_dofs]
        M_r = M_g[np.ix_(free_dofs, free_dofs)]
        assert np.all(u_r == 0)
        K_r_inv = np.linalg.inv(K_r)
        # u_r = np.linalg.solve(K_r, f_r)
        u_r = K_r_inv @ f_r
        u_g[free_dofs] = u_r
        r_g = K_g @ u_g - f_g
        eigenvalues = np.linalg.eigvals(K_r_inv @ M_r)
        if printing:
            print("Structural mass is:", np.sum(M_g) / 3)

        #   Postprocessing
        for i, n in enumerate(node_list):
            u = u_g[3 * i: 3 * (i + 1)]
            ux = u[0]
            uy = u[1]
            uz = u[2]
            F = r_g[3 * i: 3 * (i + 1)]
            Fx = F[0]
            Fy = F[1]
            Fz = F[2]
            if printing:
                print(
                    f"Node \"{n.label}\" moved from ({n.x}, {n.y}, {n.z}) to ({n.x + ux}, {n.y + uy}, {n.z + uz}). Reaction"
                    f" force it feels is {np.hypot(np.hypot(Fx, Fy), Fz)} ({Fx}, {Fy}, {Fz})")
        force_array = np.zeros_like(elements, dtype=np.float64)

        for i, e in enumerate(elements):
            n1 = node_list[e.node1]
            n2 = node_list[e.node2]
            u1 = u_g[3 * e.node1: 3 * (e.node1 + 1)]
            u2 = u_g[3 * e.node2: 3 * (e.node2 + 1)]
            d = np.array([n2.x - n1.x, n2.y - n1.y, n2.z - n1.z], dtype=np.float64)
            L = np.linalg.norm(d)
            d /= L
            p = profile_list[e.profile]
            A = (p.r ** 2 - (p.r - p.t) ** 2) * np.pi
            m = material_list[e.material]
            K = m.E * A / L
            force_array[i] = F_e = K * np.dot((u2 - u1).flatten(), d)
            F_lim = 0
            safety_factor_mat = 1.3
            safety_factor_buck = 1.2
            safety_factor_force = 1.3
            if F_e > 0:
                F_lim = A * m.sigma_y / (safety_factor_mat * safety_factor_force)
            else:
                F_lim = (-(np.pi / (L * 2)) ** 2 * (m.E / safety_factor_mat) * np.pi * (
                            p.r ** 4 - (p.r - p.t) ** 4) / 4) / safety_factor_buck
                if np.abs(F_lim) > A * m.sigma_y:
                    F_lim = -A * m.sigma_y / (safety_factor_mat * safety_factor_force)

            if printing:
                print(f"Force {connection_list[i].label} is {F_e}, limit is {F_lim}")
                print(
                    f"Stress {connection_list[i].label} is, {F_e / A}, which is {np.abs(F_e / F_lim) * 100} % of allowed\n")
                if np.abs(F_e / F_lim) * 100 > 100:
                    print('Not within safety factor \n \n')

            if abs(F_e / F_lim) > 1:
                connection_list[i].profile = larger_profile(connection_list[i].profile, profile_loc)
                # print(f'changed {old_connection_list[i].profile} to {connection_list[i].profile}')
            if abs(F_e / F_lim) < 0.5:
                connection_list[i].profile = smaller_profile(connection_list[i].profile, profile_loc)
                # print(f'changed {old_connection_list[i].profile} to {connection_list[i].profile}')

            force_array[i] /= A

        if old_connection_list == connection_list:
            running = False

        write_connections_to_file(connection_loc, connection_list)

        freq = np.real_if_close(np.sqrt(1 / eigenvalues), tol=10000) / (2 * np.pi)
        save_displacements_to_file("sample.dis", node_list, u_g, r_g)
        if printing:
            print("Vibrational modes of the structure in Hz:", *freq)
            print("Max tensile stress:", force_array.max() / 1e6, "MPa")
            print("Max x displacement:", np.max(np.abs(u_g[0::3])), '\n',
                  "Max y displacement:", np.max(np.abs(u_g[1::3])), '\n',
                  "Max z displacement:", np.max(np.abs(u_g[2::3])))

            print("Structural (half) mass is:", np.sum(M_g) / 3)
            print("Mass moment of inertia:", 2 * np.sum(MMOIz)/10E9, '10E9 kgm2')
            # print("Approx MMOI:", approx_mmoiz/10E9, '10E9 kgm2')

        if plotting:
            fig = show_structure(node_list, elements, numerical_bc_list, natural_bc_list)
            show_deformed(fig.get_axes()[0], 100 * u_g, node_list, elements, line_style="dashed", rod_color="red")
            fig.suptitle("Deformed Structure")
            plt.show()

            fig = show_forces(node_list, elements, force_array / 1e6, abs_plot=abs_plotting)
            fig.suptitle("Structural stresses")
            plt.show()

            fig = show_masses(node_list, elements, material_list, profile_list)
            fig.suptitle("Mass distribution")
            plt.show()

    return force_array, 2 * np.sum(MMOIz)


if __name__ == '__main__':
    cell_file_name = "2_not_j/structure1"
    file_loc = cell_file_name + "_fullstruct"
    optimizing = True

    # WHEN ANY OF THESE ARE CHANGED, IT HAS TO BE REOPTIMIZED OR THE NEW FORCES WONT BE APPLIED
    total_height, total_width, total_depth = 270, 270, 35
    cell_rows, cell_columns = 6, 3
    # note these are currently not scaling with the height and width of the structure
    total_thrust_rotors = 8.53E6/2/2
    rotors_per_cell = 2
    gen_count = rotors_per_cell * cell_columns
    total_generator_mass = 13E3 * 12
    total_rotor_mass = 300E3 / 9.81
    time_to_turn_90 = 100
    angular_acc = 0.5 * np.pi / time_to_turn_90 ** 2
    yaw_torque = angular_acc * 100E9
    # TODO: CHECK THAT THE TORQUE IS ACTING IN THRUST DIRECTION
    # storm conditions : 0 MN * 4 in downforce, 1.6E MN per cell thrust
    # HLD operational 0.778E6 * (cell_rows - skipped_rows) * cell_columns
    skipped_rows = 2
    total_HLD_downforce = 0.778E6 * (cell_rows - skipped_rows) * cell_columns
    total_HLD_thrust = total_HLD_downforce * 0.25

    HLD_rows = cell_rows - skipped_rows + 1
    additional_loads = [(i, 'B', 'x', -total_HLD_thrust/HLD_rows) for i in range(skipped_rows, HLD_rows + skipped_rows)]
    additional_loads += [(i, 'B', 'z', -total_HLD_downforce/HLD_rows) for i in range(skipped_rows, HLD_rows + skipped_rows)]

    if optimizing:
        if yaw_torque != 0:
            generate_structure(cell_file_name, layers=cell_rows, columns=cell_columns,
                               total_rotor_thrust=total_thrust_rotors, total_gen_mass=total_generator_mass,
                               total_rotor_mass=total_rotor_mass, additional_loads=additional_loads,
                               height=total_height, width=total_width, depth=total_depth)
            running_yaw_opt = True
            j = 0
            it_limit = 100
            while running_yaw_opt and j < it_limit:
                j += 1
                print(j)
                add_torque(filename_full=file_loc, ang_acc=angular_acc,
                           tot_gen_mass=total_generator_mass, tot_rotor_mass=total_rotor_mass,
                           axis=(total_depth / 2, 0, 0))
                main(file_loc, drive_train_count=gen_count, tot_dt_mass=total_generator_mass + total_rotor_mass,
                     optimizing=True, torque=True)

            main(file_loc, drive_train_count=gen_count, tot_dt_mass=total_generator_mass + total_rotor_mass, optimizing=False, torque=True)

        else:
            generate_structure(cell_file_name, layers=cell_rows, columns=cell_columns,
                               total_rotor_thrust=total_thrust_rotors, total_gen_mass=total_generator_mass,
                               total_rotor_mass=total_rotor_mass, additional_loads=additional_loads,
                               height=total_height, width=total_width, depth=total_depth)
            main(file_loc, drive_train_count=gen_count, tot_dt_mass=total_generator_mass + total_rotor_mass, optimizing=True)
            main(file_loc, drive_train_count=gen_count, tot_dt_mass=total_generator_mass + total_rotor_mass, optimizing=False)

    else:
        if yaw_torque != 0:
            main(file_loc, drive_train_count=gen_count, tot_dt_mass=total_generator_mass + total_rotor_mass, optimizing=False, torque=True)
        else:
            main(file_loc, drive_train_count=gen_count, tot_dt_mass=total_generator_mass + total_rotor_mass, optimizing=False)

