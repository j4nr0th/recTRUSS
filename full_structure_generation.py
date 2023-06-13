import pandas as pd

from element import elements_assemble
from plotting import show_structure, show_deformed, show_forces
from point import Point, load_points_from_file, write_points_to_file
from material import Material, load_materials_from_file
from profile import Profile, load_profiles_from_file
from connection import Connection, load_connections_from_file, write_connections_to_file
from bcs import BoundaryCondition, load_natural_bcs, load_numerical_bcs, extend_natural_bcs, extend_numerical_bcs
from matplotlib import pyplot as plt
import numpy as np


def min_max_height(nodes):
    z_vals = [node.z for node in nodes]
    return min(z_vals), max(z_vals)


def left_right_tips(nodes):
    y_vals = [node.y for node in nodes]
    return min(y_vals), max(y_vals)


def up_layer_number(label: str, layer : int = 1):
    new_layer = str(int(label[3:]) + layer)
    if len(new_layer) < 2:
        new_layer = '0' + new_layer
    new_label = label[:3] + new_layer
    return new_label


def up_column_number(label: str, column : int = 1):
    new_column = str(int(label[1:3]) + column)
    if len(new_column) < 2:
        new_column = '0' + new_column
    new_label = label[0] + new_column + label[3:]
    return new_label


def copy_nodes(nodes: list[Point], connections: list[Connection],
               layers: int, columns: int):

    # Make vertical columns
    min_height, max_height = min_max_height(nodes)
    height_offset = max_height - min_height
    non_floor_nodes = [node for node in nodes if node.z > min_height]

    column_nodes = nodes.copy()
    column_connections = connections.copy()

    for i in range(1, layers):

        # Create new points
        for node in non_floor_nodes:
            newlabel = up_layer_number(node.label, i)
            column_nodes.append(Point(label=newlabel, x=node.x, y=node.y, z=node.z+height_offset*i, t=node.t))

        # Connect new points
        for connection in connections:
            if connection.node1 in non_floor_nodes or connection.node2 in non_floor_nodes:
                newnode1, newnode2 = up_layer_number(connection.node1, i), up_layer_number(connection.node2, i)
                newconnection = Connection(label=newnode1+newnode2, node1=newnode1, node2=newnode2,
                                            material=connection.material, profile=connection.profile)

                column_connections.append(newconnection)

    # Copy pasta the columns horizontally
    left_tip, right_tip = left_right_tips(nodes)
    width = abs(right_tip - left_tip)
    non_left_tip_nodes = [node for node in column_nodes if node.y > left_tip]
    new_nodes = column_nodes.copy()
    new_connections = column_connections.copy()

    for j in range(1, columns):
        for node in non_left_tip_nodes:
            newlabel = up_column_number(node.label, j)
            new_nodes.append(Point(label=newlabel, x=node.x, y=node.y+width*j, z=node.z, t=node.t))

        for connection in column_connections:
            if connection.node1 in non_left_tip_nodes or connection.node2 in non_left_tip_nodes:
                newnode1, newnode2 = up_column_number(connection.node1, j), up_column_number(connection.node2, j)
                newconnection = Connection(label=newnode1+newnode2, node1=newnode1, node2=newnode2,
                                            material=connection.material, profile=connection.profile)

                new_connections.append(newconnection)

    return new_nodes, new_connections


def generate_numeric_bcs(filename_cell: str, nodes: list[Point], supported_vertices: list[str]):
    # Generate the BCs that should be copied to each cell
    filename_full_struct = extend_numerical_bcs(filename_cell, nodes)

    # Generate symmetry condition at the left (y=0) edge of the structure
    sym_plane_nodes = [node for node in nodes if abs(node.y) < 10E-10]
    bcs = pd.read_csv(filename_full_struct, index_col=0)
    for node in sym_plane_nodes:
        if node.label not in bcs.index:
            bcs.loc[node.label] = [None, 0, None]
        else:
            if type(bcs.loc[node.label]['y']) != float:
                bcs.loc[node.label]['y'] = 0

    # Fully constrain the support vertices
    for nodelabel in supported_vertices:
        node = nodes[nodes.index(nodelabel)]
        bcs.loc[node.label] = [node.x, node.y, node.z]

    bcs.to_csv(filename_full_struct)


def add_rotor_thrust(filename_full: str, total_thrust: float):
    """
    Forgive me for committing copy paste badness im lazy
    """
    rotor_points = get_gen_rotor_nodes(filename_full)[1]
    rotor_thrust = total_thrust / len(rotor_points)
    bcs = pd.read_csv(filename_full + '.nat', index_col=0)
    for pt in rotor_points:
        if pt in bcs.index:
            bcs.loc[pt]['Fx'] -= rotor_thrust
        else:
            bcs.loc[pt] = [-rotor_thrust,0,0]
    bcs.to_csv(filename_full + '.nat')


def get_gen_rotor_nodes(filename_full: str):
    pts = pd.read_csv(filename_full + ".pts", index_col=0).index
    # gens at bottom nodes - rotors at front non bottom nodes
    gen_locs = {point for point in pts if int(point[3:]) == 0}
    rotor_locs = {point for point in pts if int(point[3:]) != 0 and point[0] == 'A'}
    return gen_locs, rotor_locs


def add_drivetrain_load(filename_full: str, generator_mass, rotor_mass):
    gen_points, rotor_points = get_gen_rotor_nodes(filename_full)
    gen_load = generator_mass * 9.81 / len(gen_points)
    rotor_load = rotor_mass * 9.81 / len(rotor_points)

    pts = pd.read_csv(filename_full + ".pts", index_col=0).index
    bcs = pd.read_csv(filename_full + '.nat', index_col=0)
    for point in pts:
        if point in gen_points:
            if point in bcs.index:
                bcs.loc[point]['Fz'] -= gen_load
            else:
                bcs.loc[point] = [0,0,-gen_load]

        if point in rotor_points:
            if point in bcs.index:
                bcs.loc[point]['Fz'] -= rotor_load
            else:
                bcs.loc[point] = [0,0,-rotor_load]
    bcs.to_csv(filename_full + '.nat')


def add_load_to_row(filename_full: str, total_force, rownr: int, node_indicator: str = '', axis: str = 'x'):
    ax = axis.lower().strip()
    assert ax in ['x', 'y', 'z']
    pts = pd.read_csv(filename_full + ".pts", index_col=0).index
    node_types = {pt[0] for pt in pts}
    assert node_indicator in node_types or node_indicator == ''
    if node_indicator == '':
        rowpts = [pt for pt in pts if int(pt[3:]) == rownr]
    else:
        rowpts = [pt for pt in pts if int(pt[3:]) == rownr and pt[0] == node_indicator]
    load_per_node = total_force / len(rowpts)
    bcs = pd.read_csv(filename_full + '.nat', index_col=0)

    for point in rowpts:
        if point in bcs.index:
            bcs.loc[point][f'F{ax}'] += load_per_node
        else:
            if ax == 'x':
                bcs.loc[point] = [load_per_node, 0, 0]
            if ax == 'y':
                bcs.loc[point] = [0, load_per_node, 0]
            if ax == 'z':
                bcs.loc[point] = [0, 0, load_per_node]

    bcs.to_csv(filename_full + '.nat')


def add_torque(filename_full: str, ang_acc: float, tot_gen_mass: float, tot_rotor_mass: float, axis: tuple[float,...]):
    pts = pd.read_csv(filename_full + ".pts", index_col=0)
    pts.loc[:, 'mass'] = 0.
    floor_nodes = [pt for pt in pts.index if int(pt[3:]) == 0]
    rotor_mount_nodes = [pt for pt in pts.index if int(pt[3:]) != 0 and pt[0]== 'A']
    bcs = pd.read_csv(filename_full + ".nat", index_col=0)
    profiles = pd.read_csv(filename_full + ".pro", index_col=0)
    connections = pd.read_csv(filename_full + ".con", index_col=0)
    materials = pd.read_csv(filename_full.split("/")[0] + "/sample.mat", index_col=0)

    con_profiles = profiles.loc[connections["profile label"]]

    point1, point2 = pts.loc[connections["point label 1"]], pts.loc[connections["point label 2"]]
    # should fix this so I dont have to convert to arrays cuz ugly
    con_length = ((np.array(point1['x']) - np.array(point2['x']))**2 +
                  (np.array(point1['y']) - np.array(point2['y']))**2 +
                  (np.array(point1['z']) - np.array(point2['z']))**2) ** 0.5

    con_density = materials.loc[connections["material label"], "density"]
    con_mass = np.pi * (np.array(con_profiles["radius"]) ** 2 - (np.array(con_profiles["radius"]) - np.array(con_profiles["thickness"])) ** 2) \
               * con_length * np.array(con_density)

    connections['mass'] = con_mass.T
    # I suk at pandas and give up
    for index, connection in connections.loc[:, ['point label 1', 'point label 2', 'mass']].iterrows():
        pts.loc[connection['point label 1'], 'mass'] += connection["mass"] / 2
        pts.loc[connection['point label 2'], 'mass'] += connection["mass"] / 2

    gen_mass_node = tot_gen_mass / len(floor_nodes)
    for point in floor_nodes:
        pts.loc[point, 'mass'] += gen_mass_node

    rotor_mass_node = tot_rotor_mass / len(rotor_mount_nodes)
    for point in rotor_mount_nodes:
        pts.loc[point, 'mass'] += rotor_mass_node

    pt_distance = list(np.array((pts['x']-axis[0],pts['y']-axis[1], [0]*len(pts))).T)
    pts = pts.assign(distance=pt_distance)
    alpha = [0, 0, ang_acc]
    pts.loc[:, 'acc'] = [[0,0,0]] * len(pts)
    # pts.loc[:, 'eq_T_force'] = [[0,0,0]] * len(pts)
    for index, point in pts.iterrows():
        pts.at[index, 'acc'] = np.cross(alpha, point['distance'])

    pts.loc[:, 'eq_T_force'] = pts.loc[:, 'mass'] * pts.loc[:, 'acc']

    for index, pt in pts.iterrows():
        if index in bcs.index:
            bcs.loc[index,["Fx", "Fy", "Fz"]] += pt['eq_T_force']
        else:
            bcs.loc[index,["Fx", "Fy", "Fz"]] = pt['eq_T_force']

    bcs.to_csv(filename_full + "_torqued" + ".nat")


def scale_cell(filename_cell: str, tot_width, tot_height, tot_depth, cell_columns, cell_rows):
    pts = pd.read_csv(filename_cell + ".pts", index_col=0)
    width_single_cell = tot_width / cell_columns / 2
    height_single_cell = tot_height / cell_rows
    pts.loc[pts['x'] != 0, 'x'] = tot_depth
    pts.loc[pts['y'] != 0, 'y'] = width_single_cell
    pts.loc[pts['z'] != 0, 'z'] = height_single_cell
    pts.to_csv(filename_cell + ".pts")


def generate_structure(cell_filename: str, layers: int, columns: int, total_gen_mass: float, total_rotor_thrust: float,
                       total_rotor_mass: float, constrained_points=('A0000', 'B0000'),
                       additional_loads: list[tuple[int, str, str, float]] = (),
                       extend_forces_from_cell=False, height=280, width=270, depth=35, cell_scaling=True):

    """
    NAMING SCHEME CELL:
    Every point must be named XYYZZ, where X is any character indicating the unique point in the cell, 
    YY is a number indicating the column number of any duplicate point, 
    and ZZ is a number indicating the row number of any duplicate point. 
    
    Any point between two cells (both left-right and bottom-top) should have the same X character as name.
    So if you have a single cube, the bottom square would be (top view):
    
    A0000 ---- A0100
     |           |
     |           |
    B0000 ---- B0100
    
    The front square would then be (front view):
    B0001 ---- B0101
     |           |
     |           |
    B0000 ---- B0100
    
    If you dont adhere to this the script will shit itself.
    
    """

    # Number of requested layers and columns (of half the structure width) of cells
    if cell_scaling:
        scale_cell(cell_filename, tot_width=width, tot_height=height,
                   tot_depth=depth, cell_rows=layers, cell_columns=columns)

    full_structure_name = cell_filename + '_fullstruct'

    node_list = load_points_from_file(cell_filename + ".pts")
    connection_list = load_connections_from_file(cell_filename + ".con")
    newnodes, newconnections = copy_nodes(node_list, connection_list, layers=layers, columns=columns)
    write_points_to_file(full_structure_name + ".pts", newnodes)
    write_connections_to_file(full_structure_name + ".con", newconnections)

    new_bc_nat_loc = full_structure_name + '.nat'
    with open(new_bc_nat_loc, 'w') as f:
        f.write('point label,Fx,Fy,Fz\n')

    if extend_forces_from_cell:
        extend_natural_bcs(cell_filename + ".nat", newnodes)

    add_rotor_thrust(full_structure_name, total_thrust=total_rotor_thrust)
    add_drivetrain_load(full_structure_name, total_gen_mass, total_rotor_mass)

    for row, node_indicator, axis, load in additional_loads:
        add_load_to_row(full_structure_name, total_force=load, rownr=row, axis=axis, node_indicator=node_indicator)

    generate_numeric_bcs(cell_filename + ".num", newnodes, constrained_points)
