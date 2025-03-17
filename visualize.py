import random
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon

# All functions in this file are copied or adapted from https://github.com/tdmms/tdmms_DL/blob/master/mrcnn/visualize.py

def adjust_mask_size(mask, target_shape):
    """Ensures the mask matches the target shape by adding at most one row/column of zeros."""
    mask_h, mask_w = mask.shape
    target_h, target_w = target_shape
    
    dh, dw = target_h - mask_h, target_w - mask_w

    if (dh, dw) == (0, 0):
        return mask  # Already correct size

    if dh in [0, 1] and dw in [0, 1]:  
        padded_mask = np.zeros((target_h, target_w), dtype=mask.dtype)
        padded_mask[:mask_h, :mask_w] = mask
        return padded_mask
    else:
        raise ValueError(f"Mask shape mismatch: needed {dh} rows, {dw} columns. Too large to fix automatically.")

def display_instances(image, boxes, masks, classes,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_mask_polygon=True, show_bbox=True, 
                      colors=None, text_color='w', captions=None, show_caption=True, save_fig_path=None,
                      min_score=None, xlim=[0,0], ylim=[0,0], textsize = 11):

    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    show_mask_polygon (Ahmed Gad): Show the mask polygon or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    show_caption (Ahmed Gad): Whether to show the caption or not
    save_fig_path (Ahmed Gad): Path to save the figure
    min_score (Ahmed Gad): The minimum score of the objects to display.
    """

    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        print(boxes.shape,masks.shape, len(classes))
        assert boxes.shape[0] == masks.shape[-1] == len(classes)

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        if min_score is None:
            pass
        elif scores is None:
            pass
        elif scores[i] < min_score:
            continue

        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, #linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        if show_caption:
            # Label
            if not captions:
                score = scores[i] if scores is not None else None
                label = classes[i]
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]
            ax.text(x1+30, y1 + 30, caption,
                    color=color, size=textsize, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        mask = adjust_mask_size(mask, image.shape[:2])

        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        if show_mask_polygon:
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    if xlim != [0,0]:
        ax.set_xlim(*xlim)
    if ylim != [0,0]:
        ax.set_ylim(*ylim)
    ax.imshow(masked_image.astype(np.uint8))
    if not (save_fig_path is None):
        plt.savefig(save_fig_path, bbox_inches="tight")
    if auto_show:
        plt.show()

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

