"""
bbox_utils.py

Utilities for extracting person bounding boxes.

Author:
    Rishab Shetty
"""

PERSON_CLASS = 0


def extract_person_boxes(result):
    """
    Extract all person bounding boxes from a YOLO result.

    Returns
    -------
    List of boxes in xyxy format.
    """

    person_boxes = []

    for box in result.boxes:

        class_id = int(box.cls.item())

        if class_id != PERSON_CLASS:
            continue

        coords = box.xyxy.squeeze().tolist()

        person_boxes.append(coords)

    return person_boxes


def largest_person_box(person_boxes):
    """
    Return the largest person bounding box.

    Parameters
    ----------
    person_boxes : list

    Returns
    -------
    list | None
    """

    if len(person_boxes) == 0:
        return None

    largest = None
    largest_area = -1

    for box in person_boxes:

        x1, y1, x2, y2 = box

        area = (x2 - x1) * (y2 - y1)

        if area > largest_area:

            largest_area = area
            largest = box

    return largest


if __name__ == "__main__":

    print("=" * 60)
    print("Bounding Box Utilities")
    print("=" * 60)
    print("Module Loaded Successfully")
    print("=" * 60)