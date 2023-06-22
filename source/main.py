import datetime
import os.path
from sys import argv, stdout
from tomllib import load

import numpy as np
from matplotlib import pyplot as plt

from bcs import BoundaryCondition, load_natural_bcs, load_numerical_bcs
from connection import Connection, load_connections_from_file
from element import elements_assemble
from material import Material, load_materials_from_file
from plotting import show_structure, show_deformed, show_forces
from point import Point, load_points_from_file
from profile import Profile, load_profiles_from_file


def compute_global_to_local_transform(dx: float, dy: float, dz: float) -> np.ndarray:
    alpha = np.arctan2(dy, dx);
    ca = np.cos(alpha);
    sa = np.sin(alpha);
    beta = -(np.pi / 2 - np.arctan2(np.hypot(dx, dy), dz));
    cb = np.cos(beta);
    sb = np.sin(beta);
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


def save_displacements_to_file(filename: str, points: list[Point], u: np.ndarray, r: np.ndarray):
    u = u.flatten()
    r = r.flatten()
    u[np.isclose(u, 0)] = 0.0
    r[np.isclose(r, 0)] = 0.0
    with open(filename, "w") as f_out:
        f_out.write(f"point label,ux,uy,uz,Rx,Ry,Rz\n")
        for i, pt in enumerate(points):
            f_out.write(
                f"{pt.label},{u[3 * i + 0]},{u[3 * i + 1]},{u[3 * i + 2]},{r[3 * i + 0]},{r[3 * i + 1]},{r[3 * i + 2]}\n")


if __name__ == '__main__':
    if len(argv) != 2:
        print("Usage:", argv[0], "input_file")
        exit(0)
    POINTS_FILENAME: str;
    MATERIAL_FILENAME: str;
    NATURAL_BC_FILENAME: str;
    NUMERICAL_BC_FILENAME: str;
    PROFILES_FILENAME: str;
    ELEMENT_FILENAME: str;
    SILENT_OPTION: bool;
    GRAVITY_VALUE: float;
    SHOW_UNSOLVED_OPTION: bool;
    SHOW_DEFORMED_OPTION: bool;
    DEFORMATION_SCALE_VALUE: float;
    SHOW_STRESSES_OPTION: bool;
    NODE_OUTPUT_FILENAME: str;
    ELEMENT_OUTPUT_FILENAME: str;
    GENERAL_OUTPUT_FILENAME: str;
    SAVE_FIG_OPTION: bool;
    FIG_DIRNAME: str;

    print(f"Reading \"{argv[1]}\"")
    ops = None;
    try:
        ops = load((f_in := open(argv[1], "rb")))
        f_in.close()
        del f_in
    except Exception as e:
        print(f"Could not parse options from \"{argv[1]}\", reason:", e)
        exit(1)

    print("Parsing options")
    try:
        POINTS_FILENAME = str(ops["points"])
        MATERIAL_FILENAME = str(ops["materials"])
        NATURAL_BC_FILENAME = str(ops["natural_bcs"])
        NUMERICAL_BC_FILENAME = str(ops["numerical_bcs"])
        PROFILES_FILENAME = str(ops["profiles"])
        ELEMENT_FILENAME = str(ops["elements"])
        SILENT_OPTION = bool(ops["silent"])
        GRAVITY_VALUE = float(ops["gravity"])
        SHOW_UNSOLVED_OPTION = bool(ops["show_unsolved"])
        SHOW_DEFORMED_OPTION = bool(ops["show_deformed"])
        DEFORMATION_SCALE_VALUE = float(ops["deformation_scale"])
        SHOW_STRESSES_OPTION = bool(ops["show_stresses"])
        NODE_OUTPUT_FILENAME = str(ops["node_output_destination"])
        ELEMENT_OUTPUT_FILENAME = str(ops["element_output_destination"])
        GENERAL_OUTPUT_FILENAME = str(ops["general_output_destination"])
        SAVE_FIG_OPTION = bool(ops["save_figs"])
        FIG_DIRNAME = str(ops["fig_folder"])

    except Exception as e:
        print(f"Not all options were able to be converted, reason:", e)
        exit(1)

    #   Load bare data from files
    if not SILENT_OPTION:
        print("Loading problem")
    node_list: list[Point];
    material_list: list[Material];
    profile_list: list[Profile];
    connection_list: list[Connection];
    natural_bc_list: list[BoundaryCondition];
    numerical_bc_list: list[BoundaryCondition];

    try:
        node_list = load_points_from_file(POINTS_FILENAME)
    except Exception as e:
        print(f"Failed loading points from \"{POINTS_FILENAME}\", reason:", e)
        exit(1)

    try:
        material_list = load_materials_from_file(MATERIAL_FILENAME)
    except Exception as e:
        print(f"Failed loading materials from \"{MATERIAL_FILENAME}\", reason:", e)
        exit(1)

    try:
        profile_list = load_profiles_from_file(PROFILES_FILENAME)
    except Exception as e:
        print(f"Failed loading profiles from \"{PROFILES_FILENAME}\", reason:", e)
        exit(1)

    try:
        connection_list = load_connections_from_file(ELEMENT_FILENAME)
    except Exception as e:
        print(f"Failed loading elements from \"{ELEMENT_FILENAME}\", reason:", e)
        exit(1)

    try:
        natural_bc_list = load_natural_bcs(NATURAL_BC_FILENAME, node_list)
    except Exception as e:
        print(f"Failed loading natural bcs from \"{NATURAL_BC_FILENAME}\", reason:", e)
        exit(1)

    try:
        numerical_bc_list = load_numerical_bcs(NUMERICAL_BC_FILENAME, node_list)
    except Exception as e:
        print(f"Failed loading numerical_bcs from \"{NUMERICAL_BC_FILENAME}\", reason:", e)
        exit(1)

    #   Assemble elements from nodes, materials, profiles, and connections
    try:
        elements = elements_assemble(connection_list, material_list, profile_list, node_list)
    except Exception as e:
        print(f"Failed assembling elements, reason:", e)
        exit(1)
    n = len(node_list)
    n_dof = 3 * n

    if SHOW_UNSOLVED_OPTION:
        fig = show_structure(node_list, elements, numerical_bc_list, natural_bc_list)
        fig.suptitle("Problem setup")
        if SAVE_FIG_OPTION:
            out_name = os.path.join(FIG_DIRNAME, "unsolved.pdf")
            try:
                fig.savefig(out_name)
            except Exception as e:
                print(f"Could not save unsolved figure to \"{out_name}\", reason:", e)
        plt.show()

    #   Now assemble the global system
    K_g = np.zeros((n_dof, n_dof))
    M_g = np.zeros((n_dof, n_dof))
    f_g = np.zeros((n_dof, 1))
    u_g = np.zeros((n_dof, 1))
    free_dofs = np.ones(n_dof, dtype=bool)
    for i, e in enumerate(elements):
        if not SILENT_OPTION:
            print("Processing element", i, end="\r")
        #   Build transformation matrix
        n1 = node_list[e.point1]
        n2 = node_list[e.point2]
        m = material_list[e.material]
        p = profile_list[e.profile]
        dx = n2.x - n1.x
        dy = n2.y - n1.y
        dz = n2.z - n1.z
        d = np.array(((dx,), (dy,), (dz,)))
        L = np.linalg.norm(d)
        T_one = compute_global_to_local_transform(dx, dy, dz)
        T = np.zeros((6, 6))
        T[0:3, 0:3] = T_one
        T[3:6, 3:6] = T_one
        #   Build local stiffness matrix
        E = m.E
        #   Transform it to global coordinate frame
        K_e = compute_element_stiffness_matrix(T, p.A * E / L)
        #   Insert it into global matrix
        indices = (3 * e.point1, 3 * e.point1 + 1, 3 * e.point1 + 2, 3 * e.point2, 3 * e.point2 + 1, 3 * e.point2 + 2)
        K_g[np.ix_(indices, indices)] += K_e
        mass = m.rho * p.A * L
        #   Add gravitational force
        f_g[3 * e.point1 + 2] += -0.5 * m.rho * p.A * L * GRAVITY_VALUE
        f_g[3 * e.point2 + 2] += -0.5 * m.rho * p.A * L * GRAVITY_VALUE
        M_g[indices, indices] += mass / 2

    if not SILENT_OPTION:
        print("Processed elements")

    #   Apply numerical BCs
    for i, bc in enumerate(numerical_bc_list):
        if not SILENT_OPTION:
            print("Processing numerical BC", i, end="\r")
        pt_index = bc.point
        point = node_list[bc.point]
        if bc.x is not None:
            free_dofs[3 * pt_index + 0] = 0
            u_g[3 * pt_index + 0] = bc.x - point.x
        if bc.y is not None:
            free_dofs[3 * pt_index + 1] = 0
            u_g[3 * pt_index + 1] = bc.y - point.y
        if bc.z is not None:
            free_dofs[3 * pt_index + 2] = 0
            u_g[3 * pt_index + 2] = bc.z - point.z

    if not SILENT_OPTION:
        print("Processed numerical BCs")

    #   Apply natural BCs
    for i, bc in enumerate(natural_bc_list):
        if not SILENT_OPTION:
            print("Processing natural BC", i, end="\r")
        pt_index = bc.point
        if bc.x is not None:
            f_g[3 * pt_index + 0] += bc.x
        if bc.y is not None:
            f_g[3 * pt_index + 1] += bc.y
        if bc.z is not None:
            f_g[3 * pt_index + 2] += bc.z
    if not SILENT_OPTION:
        print("Processed natural BCs")

    #   Reduce the problem to only free DoFs
    K_r = K_g[np.ix_(free_dofs, free_dofs)]
    u_r = u_g[free_dofs]
    f_r = f_g[free_dofs]
    M_r = M_g[np.ix_(free_dofs, free_dofs)]
    # assert np.all(u_r == 0)

    if not SILENT_OPTION:
        print(f"Solving reduced system ({np.count_nonzero(free_dofs)} x {np.count_nonzero(free_dofs)})")

    try:
        try:
            K_r_inv = np.linalg.inv(K_r)
        except Exception as e:
            raise RuntimeError(f"Statically under-constrained (raised: {e})")

        u_r = K_r_inv @ f_r
        u_g[free_dofs] = u_r
        r_g = K_g @ u_g - f_g
        eigenvalues = np.linalg.eigvals(K_r_inv @ M_r)
    except Exception as e:
        print("Could not solve problem, reason:", e)
        exit(1)

    mass = np.sum(M_g) / 3

    f_out_node = stdout
    if NODE_OUTPUT_FILENAME != "":
        try:
            f_out_node = open(NODE_OUTPUT_FILENAME, "w")
        except Exception as e:
            print(f"Failed opening \"{NODE_OUTPUT_FILENAME}\" to write output to, reason:", e)
            exit(1)

    f_out_element = stdout
    if ELEMENT_OUTPUT_FILENAME != "":
        try:
            f_out_element = open(ELEMENT_OUTPUT_FILENAME, "w")
        except Exception as e:
            print(f"Failed opening \"{ELEMENT_OUTPUT_FILENAME}\" to write output to, reason:", e)
            exit(1)

    f_out_general = stdout
    if GENERAL_OUTPUT_FILENAME != "":
        try:
            f_out_general = open(GENERAL_OUTPUT_FILENAME, "w")
        except Exception as e:
            print(f"Failed opening \"{GENERAL_OUTPUT_FILENAME}\" to write output to, reason:", e)
            exit(1)

    f_out_node.write(f"point label,mass,ux,uy,uz,Rx,Ry,Rz\n")
    f_out_element.write(f"element label,point1,point2,F,sigma,sigma_y,sigma_b,sigma_lim,percent_allowed\n")

    #   Postprocessing
    for i, n in enumerate(node_list):
        u = u_g[3 * i: 3 * (i + 1)]
        ux = u[0][0]
        uy = u[1][0]
        uz = u[2][0]
        F = r_g[3 * i: 3 * (i + 1)]
        Fx = F[0][0]
        Fy = F[1][0]
        Fz = F[2][0]
        f_out_node.write(f"{n.label},{M_g[3 * i, 3 * i]},{ux},{uy},{uz},{Fx},{Fy},{Fz}\n")
    f_out_node.close()

    stress_array = np.zeros_like(elements, dtype=np.float64)
    for i, e in enumerate(elements):
        n1 = node_list[e.point1]
        n2 = node_list[e.point2]
        u1 = u_g[3 * e.point1: 3 * (e.point1 + 1)]
        u2 = u_g[3 * e.point2: 3 * (e.point2 + 1)]
        d = np.array([n2.x - n1.x, n2.y - n1.y, n2.z - n1.z], dtype=np.float64)
        L = np.linalg.norm(d)
        d /= L
        p = profile_list[e.profile]
        m = material_list[e.material]
        K = m.E * p.A / L
        stress_array[i] = F_e = K * np.dot((u2 - u1).flatten(), d)
        F_lim = 0
        sigma_b = -(np.pi / L) ** 2 * m.E * np.pi * p.I / 4 / p.A
        if F_e > 0:
            F_lim = p.A * m.sigma_y
        else:
            F_lim = sigma_b * p.A
            if np.abs(F_lim) > p.A * m.sigma_y:
                F_lim = -p.A * m.sigma_y
        # print(f"Force {connection_list[i].label} is {F_e}, limit is {F_lim}")
        # print(f"Stress {connection_list[i].label} is, {F_e / A}, which is {np.abs(F_e / F_lim) * 100} % of allowed\n")
        # label,point1,point2,F,sigma,sigma_y,sigma_b,percent_allowed
        f_out_element.write(f"{connection_list[i].label},{node_list[e.point1].label},{node_list[e.point2].label},{F_e},{F_e/p.A},{m.sigma_y},{sigma_b},{F_lim / p.A},{np.abs(F_e / F_lim) * 100}\n")
        stress_array[i] /= p.A
    f_out_element.close()
    freq = np.real_if_close(np.sqrt(1 / eigenvalues), tol=10000) / (2 * np.pi)
    # save_displacements_to_file("sample.dis", node_list, u_g, r_g)

    f_out_general.write(f"Simulation ran on: {datetime.datetime.now()}\n")
    f_out_general.write(f"Mass: {mass:g}\n")
    f_out_general.write(f"Max tensile stress: {np.max(stress_array):e}\n")
    f_out_general.write(f"Vibrational modes of the structure ({len(freq)} in total):\n")
    for i, f in enumerate(np.sort(freq)):
        f_out_general.write(f"\t{f}\n")
    f_out_general.close()

    if SHOW_DEFORMED_OPTION:
        fig = show_structure(node_list, elements, numerical_bc_list, natural_bc_list)
        show_deformed(fig.get_axes()[0], DEFORMATION_SCALE_VALUE * u_g, node_list, elements, line_style="dashed", rod_color="red")
        fig.suptitle("Deformed Structure")
        if SAVE_FIG_OPTION:
            out_name = os.path.join(FIG_DIRNAME, "deformed.pdf")
            try:
                fig.savefig(out_name)
            except Exception as e:
                print(f"Could not save unsolved figure to \"{out_name}\", reason:", e)
        plt.show()

    if SHOW_STRESSES_OPTION:
        fig = show_forces(node_list, elements, stress_array)
        fig.suptitle("Structural stresses")
        if SAVE_FIG_OPTION:
            out_name = os.path.join(FIG_DIRNAME, "stresses.pdf")
            try:
                fig.savefig(out_name)
            except Exception as e:
                print(f"Could not save unsolved figure to \"{out_name}\", reason:", e)
        plt.show()
