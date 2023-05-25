from dataclasses import dataclass

import pandas as pd


@dataclass
class Material:
    label: str
    E: float
    rho: float
    sigma_y: float
    alpha: float


def load_materials_from_file(filename: str) -> list[Material]:
    out_list: list
    data_in = pd.read_csv(filename, skipinitialspace=True)
    entry_count: int = len(data_in)
    out_list = [0] * entry_count
    for i in range(entry_count):
        out_list[i] = Material(data_in["material label"][i], data_in["elastic modulus"][i],
                               data_in["density"][i], data_in["strength"][i], data_in["alpha"][i])
    return out_list
