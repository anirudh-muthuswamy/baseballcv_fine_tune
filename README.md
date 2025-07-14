# Baseball Object Detection using YOLOv11


### ❗Note❗

* Due to local resource constraints (unable to train fast on apple silicon), the model training and fine-tuning were conducted on a Kaggle notebook (using NVIDIA P100 GPU). The overall fine tuning used in the notebook is similar to the one presented in this GitHub repository. The notebook used is baseballcv.ipynb, is also included for reference and reproducibility.
* Referred to: https://docs.ultralytics.com/guides/model-evaluation-insights/#how-does-fine-tuning-work

## Features

  * **Automated Setup**: A shell script automates the creation of a conda environment, installation of dependencies, and downloading the dataset and pre-trained model.
  * **Data Preparation**: Includes a script to generate the necessary `baseball_data.yaml` and clean up the dataset labels.
  * **Hyperparameter Optimization**: Uses **Optuna** to find the optimal hyperparameters for training.
  * **Model Fine-Tuning**: A script to fine-tune the YOLOv11 model with the prepared data.
  * **Performance Evaluation**: Provides a script to calculate and display performance metrics for both the original and the fine-tuned models.

## File Descriptions

| File | Description |
| --- | --- |
| `setup_baseballcv.sh` | A script that automates the setup of the environment, dependencies, dataset, and pre-trained model. |
| `setup_data.py` | Prepares the data by creating `baseball_data.yaml` and filtering labels. |
| `fine_tune.py` | Handles hyperparameter optimization and fine-tuning of the YOLOv11 model. |
| `get_metrics.py` | Evaluates and compares the performance of the baseline and fine-tuned models. |
| `utils.py` | A utility script for selecting the correct `torch` device (CUDA, MPS, or CPU). |
| `baseball_data.yaml` | The dataset configuration file for YOLO, specifying data paths and class names. |
| `.gitignore` | Specifies files and directories to be ignored by Git. |
| `best_hyperparams.yaml` | (Generated) Stores the best hyperparameters found by Optuna. |
| `fine_tuned_glove_tracking_YOLOv11.pt` | (Generated) The final fine-tuned model. |

## Setup and Installation

To begin, run the setup script. This will create the necessary environment and download all required files.

```bash
bash setup_baseballcv.sh
```

This script will:

1.  Create a `conda` environment named `baseball_venv` using Python 3.10.
2.  Activate the environment.
3.  Clone the `BaseballCV` repository.
4.  Install dependencies with `pip` and `poetry`.
5.  Download and unzip the `baseball_rubber_home_glove.zip` dataset.
6.  Download the pre-trained `glove_tracking_v4_YOLOv11.pt` model.

After the setup, activate the `conda` environment to proceed:

```bash
conda activate baseball_venv
```

## Usage

### 1\. Prepare the Data

First, prepare the dataset for training. The `setup_data.py` script will generate the `baseball_data.yaml` file and filter the labels.

There is a **class_weights** argument for the yaml file which can be used to set the class weights of each class, providing more priority for a particular class.

```bash
python setup_data.py
```

This ensures the data is correctly formatted for the YOLO model.

### 2\. Fine-Tune the Model

The `fine_tune.py` script handles both hyperparameter optimization and the actual fine-tuning process. A freeze_layers parameters can be used to set the number of initial layers to be frozen during fine tuning, which helps in retaining weights for initial feature extraction

#### Hyperparameter Optimization

To run the hyperparameter search with **Optuna**, use `--optimize_hyperparameters` argument when running `python fine_tune.py`.
This will save the best parameters to `best_hyperparams.yaml`.

#### Model Training

Model fine tuning takes place after finding best hyperparameters for batch_size, optimizer, lr, and dropout
The script will use the parameters from `best_hyperparams.yaml` if it exists, or default pretrained model hyperparameters if it does not. The resulting model is saved as `fine_tuned_glove_tracking_YOLOv11.pt`.

### 3\. Evaluate the Model

Use the `get_metrics.py` script to compare the performance of the baseline and fine-tuned models.

```bash
python get_metrics.py
```

This will output key metrics such as **mean average precision (mAP)**, **precision**, and **recall** for both models.

## Dependencies

  * Python 3.10
  * PyTorch
  * Ultralytics
  * Optuna
  * Poetry
  * Conda

## Feature Engineering: Data Augmentation

To ensure the model is robust and generalizes well across various real-world conditions, a specific set of data augmentation techniques was applied during training. The parameters for these augmentations are defined in `fine_tune.py`.

### Albumentation reinitialization

YOLO takes in default albumentation parameters which can be edited.(Using a monkey patch https://medium.com/@k.sunman91/data-augmentation-on-ultralytics-for-training-yolov5-yolov8-97a8dab31fef)
* Used RandomRain, RandomSunFlare, RandomShadow for different weather conditions.
* Used MotionBlur, MedianBlur, MedianBlur for robustness during tracking across frames
* Used CoarseDropout for areas of glove that could be obscured

### Geometric Transformations

These augmentations simulate the dynamic camera work and player movements common in baseball broadcasts.

* **Scaling, Rotation, Shear, and Perspective (`scale`, `degrees`, `shear`, `perspective`):** These are essential for handling changes in camera distance, angle, and tilt. They train the model to detect the glove regardless of its size or orientation on screen.
* **Horizontal Flipping (`fliplr`):** This technique effectively simulates both right-handed and left-handed catchers, doubling the viewpoint diversity of the dataset.

### Color Transformations

* **Hue, Saturation, and Brightness (`hsv_h`, `hsv_s`, `hsv_v`):** These are critical for handling the wide variety of lighting conditions in games; from bright daylight to stadium lights at night. This ensures the model's performance is not biased by a specific time of day or weather condition.

### Other Augmentations

* **Mosaic (`mosaic`):** This method combines four training images into one, forcing the model to recognize objects in different contexts and at smaller scales. It is particularly effective for improving the detection of small objects, such as a glove.
* **Multi_scale** Fine tune the YOLO model to different scales of image size due to small size of the glove.

## Evaluation Results

The models were evaluated on the test set of provided baseball_rubber_home_glove dataset, and the performance of the fine-tuned model was compared against the pre-trained model.

| Model | mAP@.50 | mAP@.50-.95 | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 0.978 | 0.524 | 0.965 | 0.963 |
| **Fine-Tuned** | 0.983 | 0.519 | 0.957 | 0.943 |

### Summary

The results show that the baseline model already performs at a very high level on this dataset. The fine-tuned model shows a slight improvement in **mAP@.50**, indicating it is slightly better at detecting objects when the overlap threshold is less strict.

However, there is a minor decrease in the overall **mAP@.50-.95**, **precision**, and **recall**. This suggests that while the fine-tuning process has specialized the model for this specific dataset, it may have slightly overfit, leading to a marginal drop in performance on the most challenging examples. Overall, both models demonstrate strong performance, with the fine-tuning providing a very slight trade-off between higher confidence at lower overlap and a minor decrease in overall precision.
