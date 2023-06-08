import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from profile import Profile
from bcs import BoundaryCondition
from element import Element
from material import Material
from point import Point


def show_structure(pts: list[Point], elements: list[Element], num: list[BoundaryCondition],
                   nat: list[BoundaryCondition], **kwargs) -> plt.Figure:
    #   Set colors and sizes
    line_style = None
    if "line_style" in kwargs:
        line_style = kwargs["line_style"]
    f_len = 0.5
    if "f_len" in kwargs:
        f_len = kwargs["f_len"]
    rod_color = "black"
    if "rod_color" in kwargs:
        rod_color = kwargs["rod_color"]
    dof3_color = "blue"
    if "dof3_color" in kwargs:
        dof3_color = kwargs["dof3_color"]
    dof2_color = "yellow"
    if "dof2_color" in kwargs:
        dof2_color = kwargs["dof2_color"]
    dof1_color = "purple"
    if "dof1_color" in kwargs:
        dof1_color = kwargs["dof1_color"]
    dof0_color = "red"
    if "dof0_color" in kwargs:
        dof0_color = kwargs["dof0_color"]
    COLORS = [dof0_color, dof1_color, dof2_color, dof3_color]

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot(111, projection="3d")

    x_list = np.zeros(len(pts))
    y_list = np.zeros(len(pts))
    z_list = np.zeros(len(pts))
    dof_list = 3 * np.ones(len(pts), dtype=int)
    #   properly set up the coordinates
    for i, p in enumerate(pts):
        x_list[i] = p.x
        y_list[i] = p.y
        z_list[i] = p.z
        ax.text(p.x + 0.1, p.y + 0.1, p.z + 0.1, p.label, size=8, zorder=1, color="k")

    #   count Degrees of Freedom
    for i, bc, in enumerate(num):
        if bc.x is not None:
            dof_list[bc.node] -= 1;
        if bc.y is not None:
            dof_list[bc.node] -= 1;
        if bc.z is not None:
            dof_list[bc.node] -= 1;

    colors = [dof3_color] * len(pts)

    #   make the color array
    for i, d in enumerate(dof_list):
        colors[i] = COLORS[d]

    #   draw the points
    ax.scatter(x_list, y_list, zs=z_list, color=colors)

    max_len: float = 0
    #   now process the elements
    for i, e in enumerate(elements):
        n1 = pts[e.node1]
        n2 = pts[e.node2]
        length = np.hypot(np.hypot(n1.x - n2.x, n1.y - n2.y), n1.z - n2.z)
        if length > max_len:
            max_len = float(length)
        ax.plot(xs=[n1.x, n2.x], ys=[n1.y, n2.y], zs=[n1.z, n2.z], color=rod_color, linestyle=line_style)

    n_nat_bc = len(nat)
    if n_nat_bc > 0:
        lengths = np.zeros(n_nat_bc)
        directions = np.zeros((n_nat_bc, 3))
        x_list = np.zeros(n_nat_bc)
        y_list = np.zeros(n_nat_bc)
        z_list = np.zeros(n_nat_bc)
        for i, bc in enumerate(nat):
            # find the direction of the force and its magnitude
            F = np.array([bc.x, bc.y, bc.z])
            m = np.linalg.norm(F)
            lengths[i] = m
            directions[i, :] = F / m
            n = pts[bc.node]
            x_list[i] = n.x;
            y_list[i] = n.y;
            z_list[i] = n.z;

        lengths *= f_len / np.max(lengths) * max_len
        # for i, bc, in enumerate(nat):
        #    n = pts[bc.node]
        #    v = lengths[i] * directions[i]
        #    arrow = Arrow3D(n.x, n.y, n.z, v[0], v[1], v[2])
        #    ax.add_artist(arrow)
        directions[:, 0] *= lengths.flatten()
        directions[:, 1] *= lengths.flatten()
        directions[:, 2] *= lengths.flatten()
        ax.quiver(x_list, y_list, z_list, directions[:, 0], directions[:, 1], directions[:, 2])

    ax.set_aspect("equal")
    return fig


def show_deformed(ax: plt.Axes, u: np.ndarray, pts: list[Point], elements: list[Element], **kwargs) -> None:
    #   Set colors and sizes
    line_style = None
    if "line_style" in kwargs:
        line_style = kwargs["line_style"]
    if "f_len" in kwargs:
        f_len = kwargs["f_len"]
    rod_color = "black"
    if "rod_color" in kwargs:
        rod_color = kwargs["rod_color"]

    max_len = 0
    #   now process the elements
    for i, e in enumerate(elements):
        n1 = pts[e.node1]
        n2 = pts[e.node2]
        x1 = n1.x + u[3 * e.node1 + 0]
        y1 = n1.y + u[3 * e.node1 + 1]
        z1 = n1.z + u[3 * e.node1 + 2]
        x2 = n2.x + u[3 * e.node2 + 0]
        y2 = n2.y + u[3 * e.node2 + 1]
        z2 = n2.z + u[3 * e.node2 + 2]
        length = np.hypot(np.hypot(x1 - x2, y1 - y2), z1 - z2)
        if length > max_len:
            max_len = length
        ax.plot(xs=[x1, x2], ys=[y1, y2], zs=[z1, z2], color=rod_color, linestyle=line_style)
    ax.set_aspect("equal")


def show_forces(pts: list[Point], elements: list[Element], force_arr: np.ndarray, abs_plot=False, **kwargs) -> plt.Figure:
    #   Set colors and sizes
    line_style = None
    if "line_style" in kwargs:
        line_style = kwargs["line_style"]
    colormap_name = "jet"
    if "colormap_name" in kwargs:
        colormap_name = kwargs["colormap_name"]
    node_color = "black"
    if "node_color" in kwargs:
        node_color = kwargs["node_color"]

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot(111, projection="3d")
    if abs_plot:
        forces = np.abs(force_arr)
    else:
        forces = force_arr.copy()

    colormap: matplotlib.colors.Colormap = cm.get_cmap(colormap_name)
    max_F_mag = np.max(np.abs(forces))
    if abs_plot:
        color_scalar_map = plt.cm.ScalarMappable(plt.Normalize(0, max_F_mag), colormap)
    else:
        color_scalar_map = plt.cm.ScalarMappable(plt.Normalize(-max_F_mag, max_F_mag), colormap)
    x_list = np.zeros(len(pts))
    y_list = np.zeros(len(pts))
    z_list = np.zeros(len(pts))
    #   properly set up the coordinates
    for i, p in enumerate(pts):
        x_list[i] = p.x
        y_list[i] = p.y
        z_list[i] = p.z

    #   draw the points
    ax.scatter(x_list, y_list, zs=z_list, color=node_color)
    #   now process the elements
    lines = []
    for i, e in enumerate(elements):
        n1 = pts[e.node1]
        n2 = pts[e.node2]
        color = color_scalar_map.to_rgba(forces[i])
        line = ax.plot(xs=[n1.x, n2.x], ys=[n1.y, n2.y], zs=[n1.z, n2.z], color=color, linestyle=line_style)
        lines.append(*line)

    plt.colorbar(color_scalar_map)
    ax.set_aspect("equal")

    return fig


def show_masses(pts: list[Point], elements: list[Element], material_list: list[Material],
                profile_list: list[Profile], **kwargs) -> plt.Figure:
    #   Set colors and sizes
    line_style = None
    if "line_style" in kwargs:
        line_style = kwargs["line_style"]
    colormap_name = "jet"
    if "colormap_name" in kwargs:
        colormap_name = kwargs["colormap_name"]
    node_color = "black"
    if "node_color" in kwargs:
        node_color = kwargs["node_color"]

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot(111, projection="3d")

    masses = [material_list[element.material].rho *
              np.pi * (profile_list[element.profile].r ** 2 - (profile_list[element.profile].r - profile_list[element.profile].t) ** 2)
              for element in elements]

    colormap: matplotlib.colors.Colormap = cm.get_cmap(colormap_name)
    max_mass = np.max(np.abs(masses))

    color_scalar_map = plt.cm.ScalarMappable(plt.Normalize(0, max_mass), colormap)

    x_list = np.zeros(len(pts))
    y_list = np.zeros(len(pts))
    z_list = np.zeros(len(pts))
    #   properly set up the coordinates
    for i, p in enumerate(pts):
        x_list[i] = p.x
        y_list[i] = p.y
        z_list[i] = p.z

    #   draw the points
    ax.scatter(x_list, y_list, zs=z_list, color=node_color)
    #   now process the elements
    lines = []
    for i, e in enumerate(elements):
        n1 = pts[e.node1]
        n2 = pts[e.node2]
        color = color_scalar_map.to_rgba(masses[i])
        line = ax.plot(xs=[n1.x, n2.x], ys=[n1.y, n2.y], zs=[n1.z, n2.z], color=color, linestyle=line_style)
        lines.append(*line)

    plt.colorbar(color_scalar_map)
    ax.set_aspect("equal")

    return fig
