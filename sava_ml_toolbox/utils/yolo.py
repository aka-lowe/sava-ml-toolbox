from typing import Dict, List, Optional

import cv2
import numpy as np


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """
    Perform non-maximum suppression (NMS) on bounding boxes.

    This function takes as input the bounding boxes, their associated scores,
    and an Intersection over Union (IoU) threshold. It returns a list of indices
    representing the boxes that should be kept based on the NMS algorithm.

    Args:
    - boxes (np.ndarray): Array of bounding boxes.
    - scores (np.ndarray): Array of scores associated with each box.
    - iou_threshold (float): IoU threshold for the NMS algorithm.

    Returns:
    - List[int]: List of indices of the boxes to keep.
    """
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute the Intersection over Union (IoU) between a box and multiple other boxes.

    This function takes as input a single box and an array of boxes, and computes the IoU
    between the single box and each box in the array. The IoU is a measure of the overlap
    between two bounding boxes.

    Args:
    - box (np.ndarray): Array representing a single box.
    - boxes (np.ndarray): Array of boxes.

    Returns:
    - np.ndarray: Array of IoU values.
    """
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Convert bounding box format from (x, y, w, h) to (x1, y1, x2, y2).

    This function takes as input an array representing bounding boxes in the format
    (x, y, w, h), where (x, y) is the center of the box, and w and h are the width and
    height of the box, respectively. It returns an array of the same shape, but with the
    boxes converted to the format (x1, y1, x2, y2), where (x1, y1) is the top-left corner
    of the box, and (x2, y2) is the bottom-right corner of the box.

    Args:
    - x (np.ndarray): Array of bounding boxes in the format (x, y, w, h).

    Returns:
    - np.ndarray: Array of bounding boxes in the format (x1, y1, x2, y2).
    """
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid function.

    This function takes as input an array and applies the sigmoid function to every element.
    The sigmoid function is defined as 1 / (1 + exp(-x)), and it maps any real value into
    the range (0, 1).

    Args:
    - x (np.ndarray): Input array.

    Returns:
    - np.ndarray: Output array with the sigmoid function applied element-wise.
    """
    return 1 / (1 + np.exp(-x))


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: List[int],
    detect_classes: Dict[str, str],
    mask_alpha: float = 0.3,
    mask_maps: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on an image for detected objects.

    Args:
    - image (np.ndarray): The image on which to draw.
    - boxes (np.ndarray): The bounding boxes for detected objects.
    - scores (np.ndarray): The confidence scores for each detection.
    - class_ids (List[int]): The class IDs for each detection.
    - mask_alpha (float, optional): The transparency level for masks. Defaults to 0.3.
    - mask_maps (np.ndarray, optional): The mask maps for each detection. Defaults to None.

    Returns:
    - np.ndarray: The image with the drawn detections.
    """
    # Create a list of colors for each class where each color is a tuple of 3 integer values
    colors = np.random.default_rng(3).uniform(0, 255, size=(len(detect_classes), 3))

    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    mask_img = draw_masks(image, boxes, colors, class_ids, mask_alpha, mask_maps)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        x1, y1, x2, y2 = np.array(box).astype(int)

        # Draw rectangle
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

        label = detect_classes[str(class_id)]
        caption = f"{label} {int(score * 100)}%"
        (tw, th), _ = cv2.getTextSize(
            text=caption,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=size,
            thickness=text_thickness,
        )
        th = int(th * 1.2)

        cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)

        cv2.putText(
            mask_img,
            caption,
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return mask_img


def draw_masks(
    image: np.ndarray,
    boxes: np.ndarray,
    colors: np.ndarray,
    class_ids: List[int],
    mask_alpha: float = 0.3,
    mask_maps: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw masks on an image for detected objects.

    Args:
    - image (np.ndarray): The image on which to draw.
    - boxes (np.ndarray): The bounding boxes for detected objects.
    - class_ids (List[int]): The class IDs for each detection.
    - mask_alpha (float, optional): The transparency level for masks. Defaults to 0.3.
    - mask_maps (np.ndarray, optional): The mask maps for each detection. Defaults to None.

    Returns:
    - np.ndarray: The image with the drawn masks.
    """
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        color = colors[class_id]

        x1, y1, x2, y2 = np.array(box).astype(int)

        # Draw fill mask image
        if mask_maps is None:
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
        else:
            crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
            crop_mask_img = mask_img[y1:y2, x1:x2]
            crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
            mask_img[y1:y2, x1:x2] = crop_mask_img

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


def draw_comparison(
    img1: np.ndarray,
    img2: np.ndarray,
    name1: str,
    name2: str,
    fontsize: float = 2.6,
    text_thickness: int = 3,
) -> np.ndarray:
    """
    Draw comparison between two images with their names.

    Args:
    - img1 (np.ndarray): The first image.
    - img2 (np.ndarray): The second image.
    - name1 (str): The name of the first image.
    - name2 (str): The name of the second image.
    - fontsize (float, optional): The font size for the text. Defaults to 2.6.
    - text_thickness (int, optional): The thickness of the text. Defaults to 3.

    Returns:
    - np.ndarray: The combined image with the drawn names.
    """
    (tw, th), _ = cv2.getTextSize(
        text=name1,
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=fontsize,
        thickness=text_thickness,
    )
    x1 = img1.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(
        img1,
        (x1 - offset * 2, y1 + offset),
        (x1 + tw + offset * 2, y1 - th - offset),
        (0, 115, 255),
        -1,
    )
    cv2.putText(
        img1,
        name1,
        (x1, y1),
        cv2.FONT_HERSHEY_DUPLEX,
        fontsize,
        (255, 255, 255),
        text_thickness,
    )

    (tw, th), _ = cv2.getTextSize(
        text=name2,
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=fontsize,
        thickness=text_thickness,
    )
    x1 = img2.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(
        img2,
        (x1 - offset * 2, y1 + offset),
        (x1 + tw + offset * 2, y1 - th - offset),
        (94, 23, 235),
        -1,
    )

    cv2.putText(
        img2,
        name2,
        (x1, y1),
        cv2.FONT_HERSHEY_DUPLEX,
        fontsize,
        (255, 255, 255),
        text_thickness,
    )

    combined_img = cv2.hconcat([img1, img2])
    if combined_img.shape[1] > 3840:
        combined_img = cv2.resize(combined_img, (3840, 2160))

    return combined_img
