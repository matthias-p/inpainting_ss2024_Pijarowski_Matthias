import cv2


def main():
    image = cv2.imread("image_01.jpg")
    mask = cv2.imread("mask_01.png", cv2.IMREAD_GRAYSCALE)


if __name__ == "__main__":
    main()
