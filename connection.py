import pandas as pd
from dataclasses import dataclass


@dataclass
class Connection:
    label: str;
    node1: str;
    node2: str;
    material: str;
    profile: str

    def __eq__(self,other):
        return self.label == other.label and self.node1 == other.node1 and self.node2 == other.node2 and self.material == other.material and self.profile == other.profile


def load_connections_from_file(filename: str) -> list[Connection]:
    out_list: list
    data_in = pd.read_csv(filename, skipinitialspace=True)
    entry_count: int = len(data_in)
    out_list = [0] * entry_count
    for i in range(entry_count):
        out_list[i] = Connection(str(data_in["connection label"][i]), str(data_in["point label 1"][i]),
                                 str(data_in["point label 2"][i]), str(data_in["material label"][i]), str(data_in["profile label"][i]))
    return out_list


def write_connections_to_file(filename: str, connections: list[Connection]):
    with open(filename, 'w') as f:
        f.write('connection label,material label,profile label,point label 1,point label 2\n')
        for con in connections:
            f.write(','.join([con.label, con.material, con.profile, con.node1, con.node2])+'\n')