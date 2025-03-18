# Project Name

## Overview
This project involves training a model using Detectron2 for detection of mechanically exfoliated 2D materials in microscope images. It was written as part of the MSc thesis of Benjamin Sluijter, who adapted the method from [1] to the Detectron2 framework. 
It includes steps for dataset preparation, training, inference, visualization, and labeling.

## Installation

1. Create a virtual environment and install dependencies:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

2. Install Detectron2 separately:
   ```sh
   python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

## Folder Structure
Ensure the following directories exist within the repository:

```
project_root/
│-- data/               # Contains datasets
│   ├── my_trainset/    # Example dataset folder
│   │   ├── images/     # Contains images
│   │   ├── annotations.json  # Labels (see LABELLING section)
│-- inferences/         # Stores inference results
│-- models/             # Stores trained models
│   ├── my_pre_trained_model/  # Example model to be used for transfer learning

```

---
## Training

Training is performed using `train.py`. Below is an example command:

```sh
python train.py -t my_trainset -v my_valset -f my_pre_trained_model -n my_model
```

**Arguments:**
- `-t`: Path to the training dataset (inside `data/`).
- `-v`: Path to validation dataset (defaults to training data if not provided).
- `-f`: Path to the pre-trained model in `models/` (defaults to COCO pre-trained weights if not provided).
- `-n`: Name of the output model (saved in `models/`).

Additional training parameters are configurable in `train.py`.

---
## Inference

Run inference using `inference.py`. Example commands:

```sh
python inference.py -m my_model -d my_testset -n image_1 image_2 -o my_inference
```
Or, specifying an image range:
```sh
python inference.py -m my_model -d my_testset -i 10 20 -o my_inference
```

**Arguments:**
- `-m`: Model directory inside `models/`. (`-n` argument in `train.py`)
- `-d`: Dataset directory inside `data/`.
- `-n`: Specific images (omit file extensions like .jpg/.png).
- `-i`: Alternative to `-n`, specifying an index range.
- `-o`: Output directory inside `inferences/`.

---
## Visualization

To visualize inference results:
1. Open `show_results.ipynb`.
2. Locate the section **"Show Inference Results"**.
3. Select the directory of inference results (`-o` argument in `inference.py`).
4. Choose an image to view its overlayed inference results.

---
## Labeling

Creating the `annotations.json` file involves two steps:

### Step 1: Annotate Images with LabelMe

1. Install dependencies:
   ```sh
   pip install labelme
   pip install labelme2coco
   ```

2. Open LabelMe:
   ```sh
   labelme
   ```
3. In LabelMe:
    - Open the dataset directory.
    - For each image:
        - Annotate image using **"Create Polygons"**.
        - Save annotations as JSON file with by selecting _file>save as_, also if image contains no annotations!
    - Make sure all the resulting annotation JSON are saved together in a designated folder.

### Step 2: Convert to COCO Format

1. Download `create_coco_anno.py` and open it
2. Enter the correct paths for *labelme_folder* and *output_file*
3. Run the following command:
   ```sh
   python create_coco_anno.py
   ```
4. Move the resulting `annotations.json` file to the dataset folder (`data/my_trainset/`).

Now, the dataset is ready for training!

---
## License
This project is licensed under the MIT License.


## References

[1] S. Masubuchi, E. Watanabe, Y. Seo, S. Okazaki, T. Sasagawa, K. Watanabe, T. Taniguchi, and T. Machida, "Deep-learning-based image segmentation integrated with optical microscopy for automatically searching for two-dimensional materials," *npj 2D Materials and Applications*, vol. 4, no. 1, pp. 1-6, 2020. https://doi.org/10.1038/s41699-020-0137-z
