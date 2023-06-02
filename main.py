from matplotlib import pyplot as plt

from element import elements_assemble
from plotting import show_structure, show_deformed, show_forces
from point import Point, load_points_from_file
from material import Material, load_materials_from_file
from profile import Profile, load_profiles_from_file
from connection import Connection, load_connections_from_file
from bcs import BoundaryCondition, load_natural_bcs, load_numerical_bcs

import numpy as np


def compute_global_to_local_transform(dx: float, dy: float, dz: float) -> np.ndarray:
    alpha = np.arctan2(dy, dx);
    ca = np.cos(alpha);
    sa = np.sin(alpha);
    beta = -(np.pi/2 - np.arctan2(np.hypot(dx, dy), dz));
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


def save_displacements_to_file(filename: str, nodes: list["Point"], u: np.ndarray, r: np.ndarray):
    u = u.flatten()
    r = r.flatten()
    u[np.isclose(u, 0)] = 0.0
    r[np.isclose(r, 0)] = 0.0
    with open(filename, "w") as f_out:
        f_out.write(f"point label,ux,uy,uz,Rx,Ry,Rz\n")
        for i, pt in enumerate(nodes):
            f_out.write(f"{pt.label},{u[3*i+0]},{u[3*i+1]},{u[3*i+2]},{r[3*i+0]},{r[3*i+1]},{r[3*i+2]}\n")


if __name__ == '__main__':
    #   Load bare data from files
    node_list = load_points_from_file("full_structure/structure1_fullstruct.pts")
    material_list = load_materials_from_file("full_structure/sample.mat")
    profile_list = load_profiles_from_file("full_structure/sample.pro")
    connection_list = load_connections_from_file("full_structure/structure1_fullstruct.con")
    natural_bc_list = load_natural_bcs("full_structure/structure1_fullstruct.nat", node_list)
    numerical_bc_list = load_numerical_bcs("full_structure/structure1_fullstruct.num", node_list)

    # node_list = load_points_from_file("iter2/sample.pts")
    # material_list = load_materials_from_file("iter2/sample.mat")
    # profile_list = load_profiles_from_file("iter2/sample.pro")
    # connection_list = load_connections_from_file("iter2/sample.con")
    # natural_bc_list = load_natural_bcs("iter2/sample.nat", node_list)
    # numerical_bc_list = load_numerical_bcs("iter2/sample.num", node_list)

    #   Assemble elements from nodes, materials, profiles, and connections
    elements = elements_assemble(connection_list, material_list, profile_list, node_list)
    n = len(node_list)
    n_dof = 3 * n

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
        assert np.all(np.isclose(np.array(((1,), (0,), (0,))), T_one @ (d/L)))
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
        f_g[3 * e.node1 + 2] += -0.5 * m.rho * A * L * 9.81
        f_g[3 * e.node2 + 2] += -0.5 * m.rho * A * L * 9.81
        #   Add temperature
        # F_thermal = np.abs(n1.t - n2.t) * E * A * m.alpha / (2 * L)
        # f_g[3 * e.node2: 3 * e.node2 + 3] += d * F_thermal
        # f_g[3 * e.node1: 3 * e.node1 + 3] += -d * F_thermal
        M_g[indices, indices] += mass/2
        M_e = mass / 6 * np.array(
            [[2, 0, 0, 1, 0, 0],
             [0, 2, 0, 0, 1, 0],
             [0, 0, 2, 0, 0, 1],
             [1, 0, 0, 2, 0, 0],
             [0, 1, 0, 0, 2, 0],
             [0, 0, 1, 0, 0, 2]])
        M_g[np.ix_(indices, indices)] += M_e

    #   Apply numerical BCs
    for i, bc in enumerate(numerical_bc_list):
        pt_index = bc.node
        node = node_list[bc.node]
        if bc.x is not None:
            free_dofs[3*pt_index + 0] = 0
            u_g[3*pt_index + 0] = bc.x - node.x
        if bc.y is not None:
            free_dofs[3*pt_index + 1] = 0
            u_g[3*pt_index + 1] = bc.y - node.y
        if bc.z is not None:
            free_dofs[3*pt_index + 2] = 0
            u_g[3*pt_index + 2] = bc.z - node.z

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
    print("Structural mass is:", np.sum(M_g))

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
        print(f"Node \"{n.label}\" moved from ({n.x}, {n.y}, {n.z}) to ({n.x + ux}, {n.y + uy}, {n.z + uz}). Reaction"
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
        if F_e > 0:
            F_lim = A * m.sigma_y
        else:
            F_lim = -(np.pi / (L * 2)) ** 2 * m.E * np.pi * (p.r ** 4 - (p.r - p.t) ** 4) / 4
            if np.abs(F_lim) > A * m.sigma_y:
                F_lim = -A * m.sigma_y
        print(f"Force {connection_list[i].label} is {F_e}, limit is {F_lim}")
        print(f"Stress {connection_list[i].label} is, {F_e / A}, which is {np.abs(F_e / F_lim) * 100} % of allowed\n")
        force_array[i] /= A

    freq = np.real_if_close(np.sqrt(1 / eigenvalues), tol=10000) / (2 * np.pi)
    save_displacements_to_file("sample.dis", node_list, u_g, r_g)

    print("Vibrational modes of the structure in Hz:", *freq)
    print("Max tensile stress:", force_array.max()/1e6, "MPa")
    fig = show_structure(node_list, elements, numerical_bc_list, natural_bc_list)
    show_deformed(fig.get_axes()[0], 100 * u_g, node_list, elements, line_style="dashed", rod_color="red")
    fig.suptitle("Deformed Structure")
    plt.show()

    fig = show_forces(node_list, elements, force_array/1e6)
    fig.suptitle("Structural stresses")
    plt.show()
