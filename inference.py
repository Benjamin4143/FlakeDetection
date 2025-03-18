import os
import sys
import torch
import numpy as np
import argparse
import time
import json
from datetime import datetime
import cv2
import base64
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
# Root directory of the project
ROOT_DIR = os.path.abspath("./")

sys.path.append(ROOT_DIR)  # To find local version of the library

import utils

# Initialize the parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("-m", "--model", type=str)
parser.add_argument("-p", "--checkpoint", type=str, default='model_final')
parser.add_argument("-d", "--data_path", type=str)
parser.add_argument("-i", "--image_ids", type=int, nargs='+', default=[])
parser.add_argument("-n", "--image_names", type=str, nargs='+', default=[])
parser.add_argument("-o", "--output_name", type=str, default='')

# Parse the arguments
args = parser.parse_args()


date_string = datetime.now().strftime("%d%m%y_%H%M%S")

IMAGE_DIR = os.path.join(ROOT_DIR, "data", args.data_path, "images")
OUTPUT_DIR = os.path.join(ROOT_DIR, "inferences", args.output_name)
MODEL_DIR = os.path.join(ROOT_DIR, "models", args.model)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
print(f'\n\n inference directory \n {args.output_name} \n\n')

# Write all arguments to a .txt file inside the folder
args_file_path = os.path.join(OUTPUT_DIR, "params.txt")
with open(args_file_path, "w") as file:
    file.write("modeltype: detectron\n")
    for arg, value in vars(args).items():
        if arg not in ['name', 'image_list']:
            file.write(f"{arg}: {value}\n")


# Get model config and then make predictor
cfg = get_cfg()
cfg.merge_from_file(f"{MODEL_DIR}/config.yaml")
cfg.MODEL.WEIGHTS = os.path.join(MODEL_DIR, f"{args.checkpoint}.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

with open(f"{MODEL_DIR}/categories.json", "r") as file:
    category_list = json.load(file)


valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")

if args.image_ids == []:
    image_names = args.image_names
elif args.image_names == []:
    indices = np.arange(args.image_ids[0], args.image_ids[1])
    image_names = np.array([os.path.splitext(f)[0] for f in sorted(os.listdir(IMAGE_DIR)) if f.lower().endswith(valid_extensions)])[indices]


inference_start_time = time.time()

#for each image: load image, do inference, save results in .json file
for file_name in os.listdir(IMAGE_DIR):
    image_name = os.path.splitext(file_name)[0]
    if image_name in image_names:
        image, image_meta = utils.load_image(os.path.join(IMAGE_DIR, file_name))     # Run detection
        outputs = predictor(image)

        # outputs --> result dict
        r = {}
        r["class_ids"] = outputs["instances"].pred_classes.detach().cpu().numpy()+1
        r["masks"] = utils.inverse_resize_mask(np.transpose(outputs["instances"].pred_masks.detach().cpu().numpy(), (1,2,0)),
                                                scale=image_meta["scale"], padding=image_meta["padding"], crop=image_meta["crop"])
        r["rois"] = utils.extract_bboxes(r["masks"])
        r["scores"] = outputs["instances"].scores.detach().cpu().numpy()

        image_results = utils.build_RLE_results(image_name, r, category_list)
        json_ready_results = utils.process_for_json(image_results)

        # Save to JSON file
        output_file = f"{OUTPUT_DIR}/{image_name}.json"
        with open(output_file, "w") as json_file:
            json.dump(json_ready_results, json_file, indent=4) 

time_per_image = (time.time()-inference_start_time)/len(image_names)
with open(args_file_path, "a") as file:
    file.write(f"time_per_image: {time_per_image}")