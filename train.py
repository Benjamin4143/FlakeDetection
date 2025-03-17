# import some common libraries
import os, sys, json
from datetime import datetime
import logging
import torch


from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_writers

from detectron2.modeling import build_model
from detectron2.utils.events import EventStorage
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper   # the default mapper
from detectron2.utils.file_io import PathManager

print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Number of GPUs available: {torch.cuda.device_count()}")
print(f"Current GPU device: {torch.cuda.current_device()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

logger = logging.getLogger("detectron2")

def validate(model, val_loader, val_size):
    """
    Compute validation loss for a Detectron2 model.

    Args:
        model: The Detectron2 model.
        val_loader: The validation data loader.

    Returns:
        float: The average validation loss.
    """ 
    loss_weights = {
        "loss_rpn_cls": 1.,
        "loss_rpn_loc": 1.,
        "loss_cls": 0.6,
        "loss_box_reg": 1.,
        "loss_mask": 1.
        }

    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():  # No gradient computation for validation
        for batch, iteration in zip(val_loader, range(val_size)):
            # Detectron2 models expect the batch to be a list of dictionaries
            loss_dict = model(batch)
            losses = sum(loss_weights[key] * value for key, value in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            total_loss += sum(loss_weights[key] * value for key, value in loss_dict_reduced.items())
            num_samples += len(batch)  # Number of samples in the batch

    # Calculate average loss
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    return avg_loss

# code adapted from https://github.com/facebookresearch/detectron2/blob/main/tools/plain_train_net.py

def do_train(cfg, model, train_loader, val_loader, 
             N_iter, lr, resume=False, start_iter=0, phase=0):
    """
    Perform training of a model using the configuration and options provided.

    Args:
        cfg (CfgNode): A Detectron2 configuration object containing all the 
            necessary parameters for training, including data loaders, model 
            architecture, optimizer, and training schedule.
        model (torch.nn.Module): The model to be trained. It should already be
            constructed and possibly initialized with pre-trained weights.
        resume (bool, optional): If True, resumes training from the last checkpoint. 
            Otherwise, training starts from scratch or with weights provided in the 
            configuration file. Default is False. Unlike in plain_train_net.py, optimizer is restarted. 

    Returns:
        None

    """
    model.train()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0001)
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR)
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
    max_iter = start_iter + N_iter

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    val_loss_file = os.path.join(cfg.OUTPUT_DIR, "val_loss.txt")

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.


    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(train_loader, range(start_iter, max_iter)):
        ########      ######
            image = data[0]["image"]  # The image tensor is usually under the "image" key

            # The image tensor's shape is typically [C, H, W], where C is the number of channels, H is height, and W is width
            height, width = image.shape[1], image.shape[2]
            storage.iter = iteration
            loss_dict = model(data)
            losses = sum(loss_weights[key] * value for key, value in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss_weights[key] * value for key, value in loss_dict_reduced.items())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
            ):
                val_loss = validate(model, val_loader, args.val_size)
                with open(val_loss_file, mode="a") as file:
                    file.write(f"{val_loss}\n")

                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

            if phase != 0 and iteration == (start_iter+max_iter)//2:
                checkpointer.save(f"mid_phase{phase}")


    if phase != 0:
        checkpointer.save(f"after_phase{phase}")



if __name__ == '__main__': 

    import argparse

    print(f"Is CUDA available? {torch.cuda.is_available()}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU device: {torch.cuda.current_device()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN')
    parser.add_argument('-t', '--train_path')
    parser.add_argument('-v', '--val_path', default="")
    parser.add_argument('-f', '--start_from', default="COCO")
    parser.add_argument('-l', '--lr', type=float, default=0.001)
    parser.add_argument('-i', '--its_per_epoch', type=int, default=500)
    parser.add_argument('-b', '--batch_size', type=int, default=2)
    parser.add_argument('-a', '--augmentations', type=int, default=1)
    parser.add_argument('-s', '--val_size', type=int, default=100)
    parser.add_argument("-n", "--model_name", type=str)


    args = parser.parse_args()
    # Root directory of the project
    ROOT_DIR = os.path.abspath("./")

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the libraryimport tdmcoco

    # register the dataset as "my_data"
    anno_path = os.path.join(ROOT_DIR, "data", args.train_path, "annotations.json")
    image_dir = os.path.join(ROOT_DIR, "data", args.train_path, "images")
    register_coco_instances("train_set", {}, anno_path, image_dir)

    if args.val_path != "":
        anno_path = os.path.join(ROOT_DIR, "data", args.val_path, "annotations.json")
        image_dir = os.path.join(ROOT_DIR, "data", args.val_path, "images")
        register_coco_instances("val_set", {}, anno_path, image_dir)

    #count number of categories
    with open(anno_path, 'r') as f:
        categories = json.load(f).get("categories", [])
        N_classes = len(categories)


    # put model configuration, weights and train dataset in cfg
    cfg = get_cfg()
    if args.start_from == "COCO":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    else:
        cfg.merge_from_file("{}/models/{}/config.yaml".format(ROOT_DIR, args.start_from))
        cfg.MODEL.WEIGHTS = "{}/models/{}/model_final.pth".format(ROOT_DIR, args.start_from)



    cfg.DATASETS.TRAIN = ("train_set",)
    if args.val_path != "":
        cfg.DATASETS.TEST = ("val_set",)
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.OUTPUT_DIR = f'models/{args.model_name}'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = N_classes
    cfg.TEST.EVAL_PERIOD = args.its_per_epoch
    cfg.DATALOADER.NUM_WORKERS = 2
    loss_weights = {
        "loss_rpn_cls": 1.,
        "loss_rpn_loc": 1.,
        "loss_cls": 0.6,
        "loss_box_reg": 1.,
        "loss_mask": 1.
    }
    cfg.SOLVER.CHECKPOINT_PERIOD = args.its_per_epoch*120

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)  # `exist_ok=True` avoids error if the directory already exists
    print(f'\n\n model directory \n {cfg.OUTPUT_DIR} \n\n')

    # Write all arguments to the file
    with open(os.path.join(cfg.OUTPUT_DIR, "args.txt"), "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    # Save the config to file
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        f.write(cfg.dump())

    with open(os.path.join(cfg.OUTPUT_DIR, 'categories.json'), 'w') as file:
        json.dump(categories, file, indent=4)  

    #augmentations
    if args.augmentations == 1:
        augmentations = [
            # Flip left-right with a probability of 50%
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            
            # Flip up-down with a probability of 50%
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            
            # Rotate by a random angle between -180 and 180 degrees
            T.RandomRotation(angle=[-180, 180]),
            
            # Translate by a percentage of the image dimensions (x and y)
            T.RandomExtent(scale_range=[0.8, 1.2], shift_range=[0.2, 0.2]),
            
            # Crop and pad by a random percentage
            T.RandomCrop(crop_type="relative_range", crop_size=(0.75, 0.75)),

            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
            T.RandomSaturation(0.8, 1.2),
            T.RandomLighting(0.4),
            T.ResizeShortestEdge([1024,1024], max_size=1024),
            T.FixedSizeCrop(crop_size=(1024,1024))
        ]
    else:
        augmentations = [
            T.ResizeShortestEdge([1024,1024], max_size=1024),
            T.FixedSizeCrop(crop_size=(1024,1024))
        ]

    test_augmentations = [
            T.ResizeShortestEdge([1024,1024], max_size=1024),
            T.FixedSizeCrop(crop_size=(1024,1024))
        ]

    #build torch.nn.Module from cfg
    model = build_model(cfg)
    train_loader = build_detection_train_loader(cfg,
        mapper=DatasetMapper(cfg, is_train=True, augmentations=augmentations))    #load weights to our model
    if args.val_path != "":
        val_loader = build_detection_test_loader(cfg, "val_set", mapper=DatasetMapper(cfg, is_train=True, augmentations=test_augmentations))
    else:
        val_loader = build_detection_test_loader(cfg, "train_set", mapper=DatasetMapper(cfg, is_train=True, augmentations=test_augmentations))

    print(f"Is CUDA available? {torch.cuda.is_available()}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU device: {torch.cuda.current_device()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    #step 1: freeze the resnets
    for param in model.parameters():
        param.requires_grad = True
    for param in model.backbone.bottom_up.parameters():
        param.requires_grad = False    
    print("phase 1")
    do_train(cfg, model, train_loader=train_loader, val_loader=val_loader,
              N_iter=30*args.its_per_epoch, lr=args.lr, phase=1)
    
    #step 2: unfreeze resnets 4 and 5
    for param in model.backbone.bottom_up.res4.parameters():
        param.requires_grad = True
    for param in model.backbone.bottom_up.res5.parameters():
        param.requires_grad = True

    print("phase 2")

    do_train(cfg, model, train_loader=train_loader, val_loader=val_loader,
              N_iter=30*args.its_per_epoch, lr=args.lr, resume=True, start_iter=30*args.its_per_epoch, phase=2)
    #step 3 and 4: train all params
    for param in model.parameters():
        param.requires_grad = True
    print("phase 3")
    do_train(cfg, model, train_loader=train_loader, val_loader=val_loader, 
             N_iter=30*args.its_per_epoch, lr=args.lr/10, resume=True, start_iter=60*args.its_per_epoch)
    print("phase 4")
    do_train(cfg, model, train_loader=train_loader, val_loader=val_loader, 
             N_iter=30*args.its_per_epoch, lr=args.lr/100, resume=True, start_iter=90*args.its_per_epoch, phase=4)

    
    

