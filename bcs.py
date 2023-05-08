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
