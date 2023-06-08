import numpy as np

from connection import Connection
from material import Material
from point import Point
from profile import Profile
from dataclasses import dataclass


@dataclass
class Element:
    node1: int
    node2: int
    material: int
    profile: int


def elements_assemble(connections: list[Connection], materials: list[Material], profiles: list[Profile], pts: list[Point]) -> list[Element]:
    n = len(connections)
    element_list = [None] * n
    for i in range(n):
        c = connections[i]
        p1 = [x for x in filter(lambda p: p.label == c.node1, pts)]
        assert len(p1) == 1
        i_p1 = pts.index(p1[0])

        p2 = [x for x in filter(lambda p: p.label == c.node2, pts)]
        assert len(p2) == 1
        i_p2 = pts.index(p2[0])

        mat = [x for x in filter(lambda m: m.label == c.material, materials)]
        assert len(mat) == 1;
        i_mat = materials.index(mat[0])

        pro = [x for x in filter(lambda p: p.label == c.profile, profiles)]
        assert len(pro) == 1;
        i_pro = profiles.index(pro[0])

        e = Element(i_p1, i_p2, i_mat, i_pro)
        element_list[i] = e

    return element_list

