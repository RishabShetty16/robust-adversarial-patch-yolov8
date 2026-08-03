"""
test_saliency.py

Standalone test for saliency defense.

Author:
    Rishab Shetty
"""

import cv2

from defense.saliency_defense import SaliencyDefense


def main():

    image_path = "data/sample/person.jpg"

    image = cv2.imread(image_path)

    if image is None:

        print("Image not found:")
        print(image_path)

        return

    defense = SaliencyDefense()

    saliency = defense.compute_saliency(image)

    heatmap = defense.heatmap(image)

    overlay = defense.overlay(image)

    cv2.imshow(
        "Original",
        image,
    )

    cv2.imshow(
        "Saliency",
        saliency,
    )

    cv2.imshow(
        "Heatmap",
        heatmap,
    )

    cv2.imshow(
        "Overlay",
        overlay,
    )

    print("=" * 60)
    print("Press any key to close...")
    print("=" * 60)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()