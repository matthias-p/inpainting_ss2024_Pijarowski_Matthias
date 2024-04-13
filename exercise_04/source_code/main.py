#pylint: skip-file

import cv2
import numpy as np


PATCH_SIZE = 7
HALF_PATCH_SIZE = PATCH_SIZE // 2

FILLFRONT_ITER = 0


def check_neighbours(mask: np.ndarray, y: int, x: int):
    return (mask[y - 1, x]) == 0 or (mask[y + 1, x] == 0) or (mask[y, x - 1] == 0) or (mask[y, x + 1] == 0)


def compute_confidences(confidences: np.ndarray, fillfront: list) -> list:
    fillfront_confidences = []
    for y, x in fillfront:
        fillfront_confidences.append(np.mean(confidences[y - HALF_PATCH_SIZE: y + HALF_PATCH_SIZE + 1, x - HALF_PATCH_SIZE: x + HALF_PATCH_SIZE + 1]))
    return fillfront_confidences


def select_target_coords(confidences: np.ndarray, fillfront: list) -> tuple[int, int]:
    fillfront_confidences = compute_confidences(confidences, fillfront)
    return fillfront[np.argmax(fillfront_confidences)]


def extract_fillfront(mask: np.ndarray) -> list:
    global FILLFRONT_ITER
    height, width = mask.shape
    fillfront = []
    fillfront_img = np.zeros(mask.shape)
    for y in range(HALF_PATCH_SIZE, height - HALF_PATCH_SIZE - 1):
        for x in range(HALF_PATCH_SIZE, width - HALF_PATCH_SIZE - 1):
            if mask[y, x] == 255:
                if check_neighbours(mask, y, x):
                    fillfront.append((y, x))
                    fillfront_img[y, x] = 255
    # cv2.imwrite(f"/tmp/img/fillfront{FILLFRONT_ITER}.png", fillfront_img)
    FILLFRONT_ITER += 1
    return fillfront


def calculate_cost(source: np.ndarray, target: np.ndarray) -> float:
    height, width = source.shape[0], source.shape[1]
    sum_ = 0
    for y in range(height):
        for x in range(width):
            if (target[y, x] == 255).all():
                continue
            sum_ += np.sum(np.power(source[y, x] - target[y, x], 2))
    return np.sqrt(sum_)


def find_best_patch(image: np.ndarray, mask: np.ndarray, target_coordinates: tuple[int, int]) -> tuple[int, int]:
    height, width = mask.shape
    source_region_coordinates = []
    for y in range(HALF_PATCH_SIZE, height - HALF_PATCH_SIZE - 1):
        for x in range(HALF_PATCH_SIZE, width - HALF_PATCH_SIZE - 1):
            if (mask[y - HALF_PATCH_SIZE: y + HALF_PATCH_SIZE + 1, x - HALF_PATCH_SIZE: x + HALF_PATCH_SIZE + 1] == 0).all():
                source_region_coordinates.append((y, x))

    y_target, x_target = target_coordinates
    target_patch = image[y_target - HALF_PATCH_SIZE: y_target + HALF_PATCH_SIZE + 1, x_target - HALF_PATCH_SIZE: x_target + HALF_PATCH_SIZE + 1]

    costs = []
    for y_source, x_source in source_region_coordinates:
        source_patch = image[y_source - HALF_PATCH_SIZE: y_source + HALF_PATCH_SIZE + 1, x_source - HALF_PATCH_SIZE: x_source + HALF_PATCH_SIZE + 1]
        costs.append(calculate_cost(source_patch, target_patch))
    
    return source_region_coordinates[np.argmin(costs)]


def perform_inpainting_step(image: np.ndarray, mask: np.ndarray, confidences: np.ndarray, source_coords: tuple[int, int], target_coords: tuple[int, int]):
    source_y, source_x = source_coords
    target_y, target_x = target_coords

    cv2.imwrite("/tmp/img/img_before.png", image)
    cv2.imwrite("/tmp/img/mask_before.png", mask)
    cv2.imwrite("/tmp/img/conf_before.png", confidences * 255)

    image[target_y - HALF_PATCH_SIZE: target_y + HALF_PATCH_SIZE + 1, target_x - HALF_PATCH_SIZE: target_x + HALF_PATCH_SIZE + 1] = image[source_y - HALF_PATCH_SIZE: source_y + HALF_PATCH_SIZE + 1, source_x - HALF_PATCH_SIZE: source_x + HALF_PATCH_SIZE + 1]
    mask[target_y - HALF_PATCH_SIZE: target_y + HALF_PATCH_SIZE + 1, target_x - HALF_PATCH_SIZE: target_x + HALF_PATCH_SIZE + 1] = 0
    conf_slice = confidences[target_y - HALF_PATCH_SIZE: target_y + HALF_PATCH_SIZE + 1, target_x - HALF_PATCH_SIZE: target_x + HALF_PATCH_SIZE + 1]
    conf_slice[:, :] = np.where(conf_slice == 0, np.mean(conf_slice), conf_slice)
    # confidences[target_y - HALF_PATCH_SIZE: target_y + HALF_PATCH_SIZE + 1, target_x - HALF_PATCH_SIZE: target_x + HALF_PATCH_SIZE + 1] = other
    # cv2.imwrite("/tmp/img/img_after.png", image)
    # cv2.imwrite("/tmp/img/mask_after.png", mask)
    cv2.imwrite("/tmp/img/conf_after.png", confidences * 255)


def inpaint_image(image: np.ndarray, mask: np.ndarray, confidences: np.ndarray):
    i = 0
    while True:
        fillfront = extract_fillfront(mask)
        if not fillfront:
            break
        target_coords = select_target_coords(confidences, fillfront)
        source_coords = find_best_patch(image, mask, target_coords)
        perform_inpainting_step(image, mask, confidences, source_coords, target_coords)

        cv2.imwrite(f"/tmp/img/{i}.png", image)
        i += 1
        # break


def main():
    image = cv2.imread("/home/m/Documents/inpainting_ss2024_Pijarowski_Matthias/exercise_04/source_code/image_01.jpg").astype(np.float32)
    mask = cv2.imread("/home/m/Documents/inpainting_ss2024_Pijarowski_Matthias/exercise_04/source_code/mask_01.png", cv2.IMREAD_GRAYSCALE)
    confidences = mask.copy()
    confidences = np.where(confidences == 255, 0, 1).astype(np.float32)

    image[mask == 255] = 255
    cv2.imwrite("temp.png", image)

    inpaint_image(image, mask, confidences)
    

if __name__ == "__main__":
    main()
