#pylint: skip-file

import cv2
import numpy as np


PATCH_SIZE = 7
HALF_PATCH_SIZE = PATCH_SIZE // 2


def check_neighbours(mask: np.ndarray, y: int, x: int):
    return (mask[y - 1, x]) == 0 or (mask[y + 1, x] == 0) or (mask[y, x - 1] == 0) or (mask[y, x + 1] == 0)


def extract_fillfront(mask: np.ndarray) -> list:
    height, width = mask.shape
    fillfront = []
    for y in range(HALF_PATCH_SIZE, height - HALF_PATCH_SIZE):
        for x in range(HALF_PATCH_SIZE, width - HALF_PATCH_SIZE):
            if mask[y, x] == 255:
                if check_neighbours(mask, y, x):
                    fillfront.append((y, x))
    return fillfront


def compute_confidences(confidences: np.ndarray, fillfront: list) -> list:
    fillfront_confidences = []
    for y, x in fillfront:
        fillfront_confidences.append(np.mean(confidences[y - HALF_PATCH_SIZE: y + HALF_PATCH_SIZE + 1, x - HALF_PATCH_SIZE: x + HALF_PATCH_SIZE + 1]))
    return fillfront_confidences


def select_target_coords(confidences: np.ndarray, fillfront: list) -> tuple[int, int]:
    fillfront_confidences = compute_confidences(confidences, fillfront)
    return fillfront[np.argmax(fillfront_confidences)]


def calculate_cost(source: np.ndarray, target: np.ndarray, target_mask: np.ndarray) -> float:
    # np.nditer könnte effizient sein, maske muss sein - woher sonst wissen ob pixel nicht vielleicht wirklich 255,255,255
    source_copy = source.copy()
    target_copy = target.copy()

    source_copy[target_mask == 255] = 0
    target_copy[target_mask == 255] = 0

    return np.sqrt(np.sum(np.power(source_copy - target_copy, 2)))

    # height, width = source.shape[0], source.shape[1]
    # sum_ = 0
    # for y in range(height):
    #     for x in range(width):
    #         if (target_mask[y, x] == 255):
    #             continue
    #         sum_ += np.sum(np.power(source[y, x] - target[y, x], 2))
    # return np.sqrt(sum_)


SOURCE_REGION_COORDINATES = []


def extract_source_region(mask: np.ndarray) -> list:
    global SOURCE_REGION_COORDINATES

    if not SOURCE_REGION_COORDINATES:
        height, width = mask.shape
        for y in range(HALF_PATCH_SIZE, height - HALF_PATCH_SIZE):
            for x in range(HALF_PATCH_SIZE, width - HALF_PATCH_SIZE):
                if (mask[y - HALF_PATCH_SIZE: y + HALF_PATCH_SIZE + 1, x - HALF_PATCH_SIZE: x + HALF_PATCH_SIZE + 1] == 0).all():
                    SOURCE_REGION_COORDINATES.append((y, x))
    return SOURCE_REGION_COORDINATES


def find_best_patch(image: np.ndarray, mask: np.ndarray, target_coordinates: tuple[int, int]) -> tuple[int, int]:
    height, width = mask.shape
    source_region_coordinates = extract_source_region(mask)

    y_target, x_target = target_coordinates
    target_patch = image[y_target - HALF_PATCH_SIZE: y_target + HALF_PATCH_SIZE + 1, x_target - HALF_PATCH_SIZE: x_target + HALF_PATCH_SIZE + 1]
    target_patch_mask = mask[y_target - HALF_PATCH_SIZE: y_target + HALF_PATCH_SIZE + 1, x_target - HALF_PATCH_SIZE: x_target + HALF_PATCH_SIZE + 1]

    costs = []
    for y_source, x_source in source_region_coordinates:
        source_patch = image[y_source - HALF_PATCH_SIZE: y_source + HALF_PATCH_SIZE + 1, x_source - HALF_PATCH_SIZE: x_source + HALF_PATCH_SIZE + 1]
        costs.append(calculate_cost(source_patch, target_patch, target_patch_mask))
    
    return source_region_coordinates[np.argmin(costs)]


def perform_inpainting_step(image: np.ndarray, mask: np.ndarray, confidences: np.ndarray, source_coords: tuple[int, int], target_coords: tuple[int, int]):
    source_y, source_x = source_coords
    target_y, target_x = target_coords

    # cv2.imwrite("/tmp/img/img_before.png", image)
    # cv2.imwrite("/tmp/img/mask_before.png", mask)
    # cv2.imwrite("/tmp/img/conf_before.png", confidences * 255)

    image[target_y - HALF_PATCH_SIZE: target_y + HALF_PATCH_SIZE + 1, target_x - HALF_PATCH_SIZE: target_x + HALF_PATCH_SIZE + 1] = image[source_y - HALF_PATCH_SIZE: source_y + HALF_PATCH_SIZE + 1, source_x - HALF_PATCH_SIZE: source_x + HALF_PATCH_SIZE + 1]
    mask[target_y - HALF_PATCH_SIZE: target_y + HALF_PATCH_SIZE + 1, target_x - HALF_PATCH_SIZE: target_x + HALF_PATCH_SIZE + 1] = 0
    conf_slice = confidences[target_y - HALF_PATCH_SIZE: target_y + HALF_PATCH_SIZE + 1, target_x - HALF_PATCH_SIZE: target_x + HALF_PATCH_SIZE + 1]
    conf_slice[:, :] = np.where(conf_slice == 0, np.mean(conf_slice), conf_slice)
    
    # cv2.imwrite("/tmp/img/img_after.png", image)
    # cv2.imwrite("/tmp/img/mask_after.png", mask)
    # cv2.imwrite("/tmp/img/conf_after.png", confidences * 255)


def inpaint_image(image: np.ndarray, mask: np.ndarray, confidences: np.ndarray):
    i = 0
    while True:
        fillfront = extract_fillfront(mask)
        if not fillfront:
            break
        target_coords = select_target_coords(confidences, fillfront)
        source_coords = find_best_patch(image, mask, target_coords)
        perform_inpainting_step(image, mask, confidences, source_coords, target_coords)

        cv2.imwrite(f"./inpainted/{i}.png", image)
        i += 1
        # break


def main():
    image = cv2.imread("./exercise_04/source_code/image_01.jpg").astype(np.float32)
    mask = cv2.imread("./exercise_04/source_code/mask_01.png", cv2.IMREAD_GRAYSCALE)
    confidences = mask.copy()
    confidences = np.where(confidences == 255, 0, 1).astype(np.float32)

    image[mask == 255] = 255

    inpaint_image(image, mask, confidences)
    

if __name__ == "__main__":
    main()
