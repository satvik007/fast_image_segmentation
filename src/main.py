import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from gridSegmentation import image_segmentation_3

img_dir = '../images/'


def main():
    image_names = os.listdir(img_dir)
    images = []
    for i, image_name in enumerate(image_names):
        image_path = os.path.join(img_dir, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    plt.figure(figsize=(12, 12))
    for i, img in enumerate(images):
        segments, result_image = image_segmentation_3(img)
        plt.subplot(2, 2, 2 * i + 1)
        plt.imshow(img)
        plt.subplot(2, 2, 2 * i + 2)
        plt.imshow(result_image)

    plt.show()


if __name__ == "__main__":
    main()