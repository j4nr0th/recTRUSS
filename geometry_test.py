import unittest
from main import compute_global_to_local_transform, compute_element_stiffness_matrix
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_transformation_matrix(self):
        a = (2 * np.random.rand() - 1) * np.pi
        b = np.pi / 2 * np.random.rand()
#        d = np.array([[np.sin(b) * np.cos(a)], [np.sin(b) * np.sin(a)], [np.cos(b)]])
        d = np.array([np.sin(b) * np.cos(a), np.sin(b) * np.sin(a), np.cos(b)])
        print("Direction vector is", np.eye(3) @ d, d.shape)
        T = compute_global_to_local_transform(d[0], d[1], d[2])
        print("Transformation matrix is", T)
        print("Transformed vector is", T @ d)
        self.assertTrue(np.all(np.isclose(T@d, np.array([1, 0, 0]))))
        self.assertTrue(np.all(np.isclose(T.T@np.array([1, 0, 0]), d)))

    def test_stiffness_matrix(self):
        a = (2 * np.random.rand() - 1) * np.pi
        b = np.pi / 2 * np.random.rand()
        k = 1000 * np.random.rand()
        d = np.array([np.sin(b) * np.cos(a), np.sin(b) * np.sin(a), np.cos(b)])
        dd = np.array([0, d[2], -d[1]])
        self.assertAlmostEqual(np.dot(dd, d), 0.0)
        u1 = np.random.rand() * d
        u2 = -np.random.rand() * d
        print("Direction vector is", d)
        T_one = compute_global_to_local_transform(d[0], d[1], d[2])
        T = np.zeros((6, 6))
        T[0:3, 0:3] = T_one
        T[3:6, 3:6] = T_one
        K = compute_element_stiffness_matrix(T, k)
        expected = np.abs(np.dot(u1 - u2, d)) * k
        print("Expected force", expected)
        computed = np.linalg.norm((K @ np.concatenate((u1, u2)))[0:3])
        print("Computed force", computed)
        assert np.all(np.isclose(expected, computed))
        

if __name__ == '__main__':
    unittest.main()
