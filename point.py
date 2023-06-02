from dataclasses import dataclass
import numpy as np

import pandas as pd


@dataclass
class Point:
    label: str;
    x: float;
    y: float;
    z: float;
    t: float;

    def __eq__(self, other):
        if type(other) == Point:
            return self.label == other.label
        if type(other) == str:
            return self.label == other
        raise NotImplementedError


def load_points_from_file(filename: str) -> list[Point]:
    out_list: list
    data_in = pd.read_csv(filename, skipinitialspace=True, sep=",")
    data_in = data_in.replace({np.nan: 0})
    entry_count: int = len(data_in)
    out_list = [0] * entry_count
    for i in range(entry_count):
        out_list[i] = Point(data_in["point label"][i], data_in["x"][i], data_in["y"][i], data_in["z"][i], data_in["T"][i])
    return out_list


def write_points_to_file(filename: str, points: list[Point]):
    with open(filename, 'w') as f:
        f.write('point label, x, y, z, T\n')
        for point in points:
            f.write(f"{point.label},{str(point.x)},{str(point.y)},{str(point.z)},{str(point.t)}\n")