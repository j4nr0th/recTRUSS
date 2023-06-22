from dataclasses import dataclass

import pandas as pd


@dataclass
class Connection:
    label: str;
    point1: str;
    point2: str;
    material: str;
    profile: str


def load_connections_from_file(filename: str) -> list[Connection]:
    out_list: list
    data_in = pd.read_csv(filename, skipinitialspace=True)
    entry_count: int = len(data_in)
    out_list = [0] * entry_count
    for i in range(entry_count):
        out_list[i] = Connection(str(data_in["connection label"][i]), str(data_in["point label 1"][i]),
                                 str(data_in["point label 2"][i]), str(data_in["material label"][i]), str(data_in["profile label"][i]))
    return out_list
