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


def add_drivetrain_load(filename_full: str, generator_mass, rotor_mass):
    pts = pd.read_csv(filename_full + ".pts", index_col=0).index
    floor_points = {point for point in pts if point[-1] == '0'}
    non_floor_points = set(pts) - floor_points
    gen_load = generator_mass * 9.81 / len(floor_points)
    rotor_load = rotor_mass * 9.81 / len(non_floor_points)

    bcs = pd.read_csv(filename_full + '.nat', index_col=0)
    for point in pts:
        if point in floor_points:
            if point in bcs.index:
                bcs.loc[point]['Fz'] -= gen_load
            else:
                bcs.loc[point] = [0,0,-gen_load]

        if point in non_floor_points:
            if point in bcs.index:
                bcs.loc[point]['Fz'] -= rotor_load
            else:
                bcs.loc[point] = [0,0,-rotor_load]
    bcs.to_csv(filename_full + '.nat')


def add_load_to_row(filename_full: str, total_force, rownr: int, axis: str='x'):
    ax = axis.lower().strip()
    assert ax in ['x', 'y', 'z']
    pts = pd.read_csv(filename_full + ".pts", index_col=0).index
    rowpts = [pt for pt in pts if int(pt[3:]) == rownr]
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


def generate_structure(cell_filename: str, rows, cols, total_gen_mass,
                       total_rotor_mass, constrained_points=('A0000', 'B0000'),
                       additional_loads: list[tuple[int, str, float]] = ()):

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
    layers = rows
    columns = cols

    full_structure_name = cell_filename + '_fullstruct'

    node_list = load_points_from_file(cell_filename + ".pts")

    connection_list = load_connections_from_file(cell_filename + ".con")
    newnodes, newconnections = copy_nodes(node_list, connection_list, layers=layers, columns=columns)
    extend_natural_bcs(cell_filename + ".nat", newnodes)
    add_drivetrain_load(full_structure_name, total_gen_mass, total_rotor_mass)

    for row, axis, load in additional_loads:
        add_load_to_row(full_structure_name, total_force=load, rownr=row, axis=axis)

    generate_numeric_bcs(cell_filename + ".num", newnodes, constrained_points)
    write_points_to_file(full_structure_name + ".pts", newnodes)
    write_connections_to_file(full_structure_name + ".con", newconnections)
