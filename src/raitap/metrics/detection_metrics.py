from dataclasses import dataclass

import torch


def calculate_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate pairwise intersection-over-union (IoU) between two sets of bounding boxes.
    :param boxes1: [N,4]
    :param boxes2: [M,4]
    :return: [N,M] IoU values
    """
    # Input validation: boxes must be 2D like [N,4] (last dim is xyxy)
    if boxes1.ndim != 2 or boxes1.shape[-1] != 4:
        raise ValueError(f"boxes1 must be [N,4], got {tuple(boxes1.shape)}")
    if boxes2.ndim != 2 or boxes2.shape[-1] != 4:
        raise ValueError(f"boxes2 must be [M,4], got {tuple(boxes2.shape)}")

    # Convert to float32 as calculations involve division and integer division would lead to incorrect results
    # float32 is standard for geometric calculations
    boxes1 = boxes1.to(torch.float32)
    boxes2 = boxes2.to(torch.float32)

    # calculate areas
    # boxes1[:, 2] = all x_max values
    # boxes1[:, 0] = all x_min values
    # boxes1[:, 3] = all y_max values
    # boxes1[:, 1] = all y_min values
    # .clamp(min=0) -> remove negative values -> 0
    # -> calculate width and height of each box
    # area1 = [N] one area for each box
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(
        min=0
    )  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(
        min=0
    )  # [M]

    # calculate intersections
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    intersec_width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    intersec_area = intersec_width_height[:, :, 0] * intersec_width_height[:, :, 1]  # [N,M]

    # calculate union
    union_area = area1[:, None] + area2[None, :] - intersec_area

    # Calculate IoU with safe division
    iou = torch.where(union_area > 0, intersec_area / union_area, torch.zeros_like(intersec_area))
    return iou


def calculate_iou(bbox1, bbox2):

    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection_area = inter_w * inter_h

    area_bbox1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    area_bbox2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])

    union_area = area_bbox1 + area_bbox2 - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0

    return iou


# Define two bounding boxes

true_box = [50, 50, 150, 150]

region_proposal_box = [100, 100, 200, 200]

# Calculate IoU

iou = calculate_iou(true_box, region_proposal_box)

# Print the result

print("Intersection over Union (IoU):", iou)
