import sys
import unittest

import numpy as np

sys.path.append("/home/matthias/Documents/inpainting_ss2024_Pijarowski_Matthias/cpp")

import example


defined = 0


def euclidean_distance(point1: tuple[int, int], point2: tuple[int, int]) -> float:
    p1_y, p1_x = point1
    p2_y, p2_x = point2

    return np.sqrt((p2_y - p1_y) ** 2 + (p2_x - p1_x) ** 2)



def find_nearest_neighbors_circle(mask: np.ndarray, origin: tuple[int, int], k: int) -> list[tuple[int, int]]:
    origin_y, origin_x = origin
    height, width = mask.shape

    d_tl, d_bl, d_tr, d_br = (euclidean_distance(origin, (0, 0)), 
                              euclidean_distance(origin, (height, 0)), 
                              euclidean_distance(origin, (0, width)), 
                              euclidean_distance(origin, (height, width)))

    thetas = np.linspace(0, 2, 360, endpoint=False) * np.pi
    neighbors = set()

    for radius in range(1, round(max([d_tl, d_bl, d_tr, d_br]))):
        Y = np.round(radius * np.cos(thetas) + origin_y).astype(int)
        X = np.round(radius * np.sin(thetas) + origin_x).astype(int)
        for y, x in zip(Y, X):
            if 0 <= y < height and 0 <= x < width and mask[y, x] == defined:
                neighbors.add((y, x))
                if len(neighbors) == k:
                    return list(neighbors)

    return list(neighbors)


class TestKNN(unittest.TestCase):
    def test_circular(self):
        origin = (15, 15)
        image = np.full((31, 31), 255)
        image[0, 0] = 0
        image[0, 30] = 0
        image[30, 0] = 0
        image[30, 30] = 0
        image[14, 15] = 128
        neighbors = find_nearest_neighbors_circle(image, origin, 4)
        gt = [(0, 0), (30, 0), (0, 30), (30, 30)]

        self.assertCountEqual(neighbors, gt)

    def test_native(self):
        origin = (15, 15)
        image = np.full((31, 31), 255)
        image[0, 0] = 0
        image[0, 30] = 0
        image[30, 0] = 0
        image[30, 30] = 0
        image[15, 15] = 111
        neighbors = example.nn_circular(image, origin, 4)
        gt = [(0, 0), (30, 0), (0, 30), (30, 30)]

        self.assertCountEqual(neighbors, gt)

unittest.main(argv=[""], verbosity=2)


# a = np.ones((31, 31))
# a = a.flatten()
# a[480] = 127
# a = a.reshape(31, 31)
# print(np.argwhere(a != 1))
