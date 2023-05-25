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


def load_points_from_file(filename: str) -> list[Point]:
    out_list: list
    data_in = pd.read_csv(filename, skipinitialspace=True, sep=",")
    data_in = data_in.replace({np.nan: 0})
    entry_count: int = len(data_in)
    out_list = [0] * entry_count
    for i in range(entry_count):
        out_list[i] = Point(data_in["point label"][i], data_in["x"][i], data_in["y"][i], data_in["z"][i], data_in["T"][i])
    return out_list
