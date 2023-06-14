import pandas as pd


def add_structure_drag(full_filename: str, nodes: list[str], drag_element: float):
    loads = pd.read_csv(full_filename + ".nat", index_col=0)

    for node in nodes:
        if node in loads.index:
            loads.loc[node, 'Fx'] -= drag_element / len(nodes)
        else:
            loads.loc[node, 'Fx'] = - drag_element / len(nodes)

    loads.to_csv(full_filename + ".nat")
