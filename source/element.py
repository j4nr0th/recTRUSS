from dataclasses import dataclass

from connection import Connection
from material import Material
from point import Point
from profile import Profile


@dataclass
class Element:
    point1: int
    point2: int
    material: int
    profile: int


def elements_assemble(connections: list[Connection], materials: list[Material], profiles: list[Profile], pts: list[Point]) -> list[Element]:
    n = len(connections)
    element_list = [None] * n
    for i, c in enumerate(connections):
        p1 = [x for x in filter(lambda p: p.label == c.point1, pts)]
        if len(p1) != 1:
            if len(p1) == 0:
                raise RuntimeError(f"Point with label \"{c.point1}\" has not been defined")
            else:
                raise RuntimeError(f"{len(p1)} different points had the same label\"{c.point1}\"")
        i_p1 = pts.index(p1[0])

        p2 = [x for x in filter(lambda p: p.label == c.point2, pts)]
        if len(p2) != 1:
            if len(p2) == 0:
                raise RuntimeError(f"Point with label \"{c.point2}\" has not been defined")
            else:
                raise RuntimeError(f"{len(p2)} different points had the same label\"{c.point2}\"")
        i_p2 = pts.index(p2[0])

        mat = [x for x in filter(lambda m: m.label == c.material, materials)]
        if len(mat) != 1:
            if len(mat) == 0:
                raise RuntimeError(f"Material with label \"{c.material}\" has not been defined")
            else:
                raise RuntimeError(f"{len(mat)} different materials had the same label\"{c.material}\"")
        i_mat = materials.index(mat[0])

        pro = [x for x in filter(lambda p: p.label == c.profile, profiles)]
        if len(pro) != 1:
            if len(pro) == 0:
                raise RuntimeError(f"Material with label \"{c.profile}\" has not been defined")
            else:
                raise RuntimeError(f"{len(pro)} different materials had the same label\"{c.profile}\"")
        i_pro = profiles.index(pro[0])

        e = Element(i_p1, i_p2, i_mat, i_pro)
        element_list[i] = e;

    return element_list

