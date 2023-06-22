from dataclasses import dataclass

import pandas as pd


@dataclass
class Profile:
    label: str;
    A: float;
    I: float;

def load_profiles_from_file(filename: str) -> list[Profile]:
    out_list: list
    data_in = pd.read_csv(filename, skipinitialspace=True)
    entry_count: int = len(data_in)
    out_list = [0] * entry_count
    for i in range(entry_count):
        out_list[i] = Profile(data_in["profile label"][i], data_in["A"][i], data_in["I"][i])
    return out_list
