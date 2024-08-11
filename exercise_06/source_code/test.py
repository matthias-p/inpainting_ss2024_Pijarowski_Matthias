import numpy as np

import inpainting_functions

def sparse_l2(image: np.ndarray, origin: tuple, candidate: tuple, neighbors: list[tuple]) -> float:
    origin_y, origin_x = origin
    candidate_y, candidate_x = candidate
    distance = 0
    for neighbor_y, neighbor_x in neighbors:
        distance += np.sum(np.power(image[origin_y + neighbor_y, origin_x + neighbor_x] - image[candidate_y + neighbor_y, candidate_x + neighbor_x], 2))

    return np.sqrt(distance)

image = np.zeros((30, 30, 3), dtype=np.uint8)
image[0:3, 0:3] = 0
image[25: 28, 25: 28] = 1

neighbors = []
for i in range(-1, 2):
    for j in range(-1, 2):
        neighbors.append((i, j))

origin = (1, 1)
candidate = (26, 26)

print(sparse_l2(image, origin, candidate, neighbors))
print(inpainting_functions.sparse_l2(image, origin, candidate, neighbors))
