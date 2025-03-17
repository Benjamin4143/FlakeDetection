from pycocotools import mask as maskUtils
import numpy as np
import skimage
import cv2
import os
import base64
from distutils.version import LooseVersion
import random
import warnings
import scipy
import json

# most functions below were copied or adapted from:
# https://github.com/tdmms/tdmms_DL/blob/master/tdmcoco.py
# or 
# https://github.com/matterport/Mask_RCNN/tree/master/mrcnn
# as specified in the function descriptions

def annToMask(ann, height, width):
    """ from tdmcoco.py
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m

def annToRLE(ann, height, width):
    """ from tdmcoco.py
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann['segmentation']
    return rle

def create_unique_folder(base_path):
    """
    Create a unique folder by appending numbers if the folder already exists.
    """
    folder_path = base_path
    counter = 1
    while os.path.exists(folder_path):
        folder_name = f"{base_path}_{counter}"
        counter += 1
    os.makedirs(folder_path)
    return folder_path

def load_params(param_file):
    params = {}
    with open(param_file, "r") as file:
        for line in file:
            key, value = line.strip().split(": ", maxsplit=1)
            params[key] = value
    return params

def process_for_json(data):
    """
    not all the data types in the output of build_coco_results 
    can be saved in a json, which is what this file takes care of
    """
    for entry in data:
        # Convert numpy float to native Python float
        if isinstance(entry['score'], np.float32):
            entry['score'] = float(entry['score'])

        # Convert bytes to base64 encoded string
        if isinstance(entry['segmentation']['counts'], bytes):
            entry['segmentation']['counts'] = base64.b64encode(entry['segmentation']['counts']).decode('utf-8')

        # Convert numpy int32 to native Python int in the "size" list
        if isinstance(entry['bbox'], list):
            entry['bbox']= [int(i) if isinstance(i, np.int32) else i for i in entry['bbox']]

    return data

def extract_bboxes(mask):
    """Function copied from mrcnn.utils
    
    Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """ Function copied from mrcnn.utils
    Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop

def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """ Function copied from mrcnn.utils
    A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)

def resize_mask(mask, scale, padding, crop=None):

    """ copied from mrcnn/utils.py
    Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

def inverse_resize_mask(resized_mask, scale, padding, crop=None):
    """Reverses the resizing operation of a mask using the given scale, padding, and crop.
    
    resized_mask: The resized mask to be reverted.
    scale: The scaling factor used during resizing.
    padding: Padding added during resizing, in the form [(top, bottom), (left, right), (0, 0)].
    crop: Crop applied during resizing, in the form (y, x, h, w). If None, no cropping was applied.
    shape: The expected final shape of the output mask (height, width).
    
    Returns:
    Original mask (approximately reconstructed), ensuring it matches the specified shape.
    """
    if crop is not None:
        raise ValueError("expect crop to be None")
    
    # Reverse padding by removing the added borders
    top, bottom = padding[0]
    left, right = padding[1]
    mask = resized_mask[top:resized_mask.shape[0] - bottom, 
                        left:resized_mask.shape[1] - right]
    
    # Reverse scaling
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original_mask = scipy.ndimage.zoom(mask, zoom=[1/scale, 1/scale, 1], order=0)
    
    return original_mask


def load_image(im_path):
    """adaptation from load_image_gt() from mrcnn/model.py"""


    #load image
    im = skimage.io.imread(im_path)
    original_shape = im.shape

    image, window, scale, padding, crop = resize_image(
        im,
        min_dim=800,
        min_scale=0,
        max_dim=1024,
        mode="square")

    image_meta = {'original_shape': original_shape, 'image_shape': image.shape, 
                  'scale': scale, 'padding': padding, 'crop': crop}

    return image, image_meta

def get_name_by_id(data, target_id):
    """Returns the name corresponding to the given ID from a list of dictionaries."""
    for item in data:
        if item["id"] == target_id:
            return item["name"]
    return None  # Return None if ID is not found

def build_RLE_results(image_name, r, classes_dict):
    """Arrange results to match COCO specs in http://cocodataset.org/#format
    such that it can be used as input for coco.loadRes()

    input r : matterport/mrcnn format inference results
    output : Results that is ready for coco.loadRes. With masks in RLE 

    """

    rois = r["rois"]
    class_ids=r["class_ids"]
    scores = r["scores"]
    masks = r["masks"].astype(np.uint8)

    # If no results, return an empty list
    if rois is None:
        return []

    RLE_results = []
    # Loop through detections
    for i in range(rois.shape[0]):
        class_id = class_ids[i]
        score = scores[i]
        bbox = np.around(rois[i], 1)
        mask = masks[:, :, i]

        class_name = get_name_by_id(classes_dict, class_id)

        result = {
            "image_name": image_name,
            "category": class_name,
            "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
            "score": score,
            "segmentation": maskUtils.encode(np.asfortranarray(mask))
        }

        RLE_results.append(result)
    return RLE_results

def json_to_RLE_results(file_path):
    """
    read in json inference results file and do the inverse 
    datatype change from process_for_json()

    input: inference results json file path
    output: results in the output format of build_RLE_results with masks in RLE encoding
    
    """
    with open(file_path, "r") as json_file:
        RLE_results = json.load(json_file)

    # Loop through each COCO-formatted result
    for result in RLE_results:
        segmentation = result["segmentation"]
        segmentation['counts'] = base64.b64decode(segmentation['counts'])
    
    return RLE_results
    
def parse_RLE_results(RLE_results):
    """
    Inverse of build_RLE_results. Takes the 
    input: results in coco format with masks in RLE encoding (output of build_RLE_results)
    output: results in matterport/mrcnn format
    """
    rois = []
    classes = []
    scores = []
    masks = []

    # Loop through each COCO-formatted result
    for result in RLE_results:
        # Extract fields
        category = result["category"]
        bbox = result["bbox"]
        score = result["score"]
        segmentation = result["segmentation"]

        # Decode the segmentation mask
        mask = maskUtils.decode(segmentation)

        # Reconstruct ROI
        y1, x1 = bbox[1], bbox[0]
        y2, x2 = bbox[1] + bbox[3], bbox[0] + bbox[2]
        roi = [y1, x1, y2, x2]

        # Append to respective lists
        rois.append(roi)
        classes.append(category)
        scores.append(score)
        masks.append(mask)

    # Convert lists to the appropriate format
    rois = np.array(rois)
    classes = np.array(classes)
    scores = np.array(scores)
    masks = np.stack(masks, axis=-1) if masks else np.zeros((0, 0, 0), dtype=np.uint8)

    # Return the reconstructed dictionary
    return {
        "rois": rois,
        "classes": classes,
        "scores": scores,
        "masks": masks,
    }

# Function to get the ground truth for a given image from a specified COCO JSON path
def get_ground_truth(coco_json_path, image_name):
    # Load the COCO annotations JSON file
    with open(coco_json_path) as f:
        coco_data = json.load(f)

    # Get the image info and annotations from the COCO data
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    # Create a mapping from class ID to class name
    class_names = {category['id']: category['name'] for category in categories}

    # Find the image by filename
    image_info = next((img for img in images if img['file_name'] == image_name), None)
    if not image_info:
        raise ValueError(f"Image {image_name} not found in the annotations")

    image_id = image_info['id']
    image_width = image_info['width']
    image_height = image_info['height']
    
    # Get all annotations for this image
    image_annotations = [anno for anno in annotations if anno['image_id'] == image_id]
    
    # Initialize lists to store ground truth values
    rois = []
    masks = []
    classes = []
    scores = []

    # Process each annotation for the image
    for anno in image_annotations:
        # Get bounding box (r['rois']) [x, y, width, height]
        x, y, w, h = anno['bbox']
        rois.append([y, x, y + h, x + w])  # Convert to [top, left, bottom, right] format
        
        # Get the segmentation mask (r['masks']) as a binary mask
        mask = annToMask(anno, image_height, image_width)
        
        masks.append(mask)
        
        # Get class name (r['classes'])
        class_name = class_names[anno['category_id']]
        classes.append(class_name)
        
        # Get score (r['scores']) - Ground truth doesn't have scores, so we set them to 1 (as it's 100% confident)
        scores.append(1.0)
    mask_array = np.stack(masks, axis=2).astype(np.bool)

    # Return the ground truth dictionary
    return {
        'rois': np.array(rois),
        'masks': mask_array,
        'classes': classes,
        'scores': np.array(scores)
    }

class ConfusionMatrix:
    """This code is from https://github.com/Jaluus/2DMatGMM/blob/main/Utils/confusionmatrix.py
    
    Constructs a confusion matrix for a multi-class classification problems.

    Does not support multi-label, multi-class problems.

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.

    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, ignore_label):
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix

        The shape of the confusion matrix is K x K, where K is the number
        of classes.

        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.

        """
        # _, predicted = predicted.max(1)

        # predicted = predicted.view(-1)
        # target = target.view(-1)

        # If target and/or predicted are tensors, convert them to numpy arrays
        ind = ~np.isin(target, self.ignore_label)
        predicted, target = predicted[ind], target[ind]

        assert (
            predicted.shape[0] == target.shape[0]
        ), "number of targets and predicted outputs do not match"

        if np.ndim(predicted) != 1:
            assert (
                predicted.shape[1] == self.num_classes
            ), "number of predictions does not match size of confusion matrix"
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.num_classes) and (
                predicted.min() >= 0
            ), "predicted values are not between 0 and k-1"

        if np.ndim(target) != 1:
            assert (
                target.shape[1] == self.num_classes
            ), "Onehot target does not match size of confusion matrix"
            assert (target >= 0).all() and (
                target <= 1
            ).all(), "in one-hot encoding, target values should be 0 or 1"
            assert (target.sum(1) == 1).all(), "multi-label setting is not supported"
            target = np.argmax(target, 1)
        else:
            assert (target.max() < self.num_classes) and (
                target.min() >= 0
            ), "target values are not between 0 and k-1"

        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.conf += conf

    def value(self, normalized=False):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        return self.conf