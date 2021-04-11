import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

img_dir = '../images/'


def main():
    image_names = os.listdir(img_dir)
    images = []
    plt.figure(figsize=(12, 6))
    for i, image_name in enumerate(image_names):
        image_path = os.path.join(img_dir, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        plt.subplot(1, 2, i + 1)
        plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()