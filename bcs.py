from dataclasses import dataclass

import numpy as np
import pandas as pd

from point import Point


@dataclass
class BoundaryCondition:
    node: int;
    x: float;
    y: float;
    z: float;


def load_natural_bcs(filename: str, pts: list[Point]) -> list[BoundaryCondition]:
    out_list: list
    data_in = pd.read_csv(filename, skipinitialspace=True)
    data_in = data_in.replace({np.nan: 0})
    entry_count: int = len(data_in)
    out_list = [0] * entry_count
    for i in range(entry_count):
        point_indices = [x for x in filter(lambda p: p.label == data_in["point label"][i], pts)]
        assert len(point_indices) == 1
        out_list[i] = BoundaryCondition(pts.index(point_indices[0]), data_in["Fx"][i], data_in["Fy"][i], data_in["Fz"][i])
    return out_list


def load_numerical_bcs(filename: str, pts: list[Point]) -> list[BoundaryCondition]:
    out_list: list
    data_in = pd.read_csv(filename, skipinitialspace=True)
    data_in = data_in.replace({np.nan: None})
    entry_count: int = len(data_in)
    out_list = [0] * entry_count
    for i in range(entry_count):
        point_indices = [x for x in filter(lambda p: p.label == data_in["point label"][i], pts)]
        assert len(point_indices) == 1
        out_list[i] = BoundaryCondition(pts.index(point_indices[0]), data_in["x"][i], data_in["y"][i], data_in["z"][i])
    return out_list


def extend_natural_bcs(filename: str, pts: list[Point]):
    """
    For copying boundary conditions when generating a full structure out of a single cell.
    """

    old_data = pd.read_csv(filename, index_col=0)
    new_data = old_data.copy()
    for pt in pts:
        if pt.label in old_data.index:
            continue
        parent_label = pt.label[0]
        parent_bc = old_data[old_data.index.str.contains(parent_label)]
        if len(parent_bc) < 1:
            continue

        new_data.loc[pt.label] = parent_bc.iloc[0]

    name, suffix = filename.split('.')
    new_bc_loc = name + '_fullstruct' + '.' + suffix
    new_data.to_csv(new_bc_loc)


def extend_numerical_bcs(filename: str, pts: list[Point]):
    """
    When using this only enter the bcs in the file for nodes that need to be constrained for every copy.
    Boundary conditions that are not copied cell by cell should be added manually afterwards.
    """

    old_bcs = pd.read_csv(filename, index_col=0)
    new_bcs = old_bcs.copy()
    bc_types = list({idx[0] for idx in old_bcs.index})
    for bc in bc_types:
        parent_conditions = old_bcs[old_bcs.index.str.contains(bc)].iloc[0]
        similar_points = [pt for pt in pts if pt.label[0] == bc and pt.label not in old_bcs.index]
        for pt in similar_points:
            new_bcs.loc[pt.label] = parent_conditions
            if type(parent_conditions['x']) == float:
                new_bcs.loc[pt.label]['x'] = pt.x
            if type(parent_conditions['y']) == float:
                new_bcs.loc[pt.label]['y'] = pt.y
            if type(parent_conditions['z']) == float:
                new_bcs.loc[pt.label]['z'] = pt.z

    name, suffix = filename.split('.')
    new_bc_loc = name + '_fullstruct' + '.' + suffix
    new_bcs.to_csv(new_bc_loc)
    return new_bc_loc
