import numpy as np
import time
import sys

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
            if mask[y, x] == defined and 0 <= y < height and 0 <= x < width:
                neighbors.add((y, x))
                if len(neighbors) == k:
                    return list(neighbors)

    return list(neighbors)


def find_nearest_neighbors_circle_2(mask: np.ndarray, origin: tuple[int, int], k: int) -> list[tuple[int, int]]:
    origin_y, origin_x = origin
    height, width = mask.shape

    d_tl, d_bl, d_tr, d_br = (euclidean_distance(origin, (0, 0)), 
                              euclidean_distance(origin, (height, 0)), 
                              euclidean_distance(origin, (0, width)), 
                              euclidean_distance(origin, (height, width)))

    thetas = np.linspace(0, 2, 360, endpoint=False) * np.pi
    neighbors = []

    for radius in range(1, round(max([d_tl, d_bl, d_tr, d_br]))):
        Y = np.round(radius * np.cos(thetas) + origin_y).astype(np.int32)
        X = np.round(radius * np.sin(thetas) + origin_x).astype(np.int32)
        neighbors += example.get_neighbors(mask, Y, X, k - len(neighbors))
        if len(neighbors) == k:
            break

    return neighbors


np.random.seed(0)
image_size = 4000
origin = (image_size // 2, image_size // 2)
test_image = np.random.rand(image_size ** 2).reshape((image_size, image_size))
test_image = np.where(test_image < 0.01, 0, 255)

t1 = time.time()
for _ in range(1000):
    find_nearest_neighbors_circle(mask=test_image, origin=origin, k=4)
print(time.time() - t1)

t1 = time.time()
for _ in range(1000):
    find_nearest_neighbors_circle_2(mask=test_image, origin=origin, k=4)
print(time.time() - t1)

# t1 = time.time()
# for _ in range(1000):
#     example.nn_circular(test_image, origin, 4)
# print(time.time() - t1)

# t1 = time.time()
# for _ in range(1000):
#     example.nn_quadratic(test_image, origin, 4)
# print(time.time() - t1)