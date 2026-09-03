This repository focuses on the research and experimental component.

Repository Structure

RoadGuard-Research/
│
├── Codes/
│   ├── 00_prepare_test_split.py
│   ├── 01_train_all_models.py
│   ├── 02_evaluate_all_models.py
│   ├── 03_results_table.py
│   ├── audit_class_samples.py
│   ├── bbox_editor.py
│   ├── convert_voc_to_yolo.py
│   ├── evaluate_all_models.py
│   ├── train_yolo_roc.py
│   └── train_yolov8_enhanced.py
│
├── Models/
│   ├── yolov5n/
│   ├── yolov8n/
│   └── yolov10n/
│
├── Demonstration Video/
│   └── Demonstration Video.mp4
│
└── Results/
    └── RoadGuard model analysis and experimental results

Codes

The Codes directory contains the scripts used for dataset preparation, annotation processing, model training, evaluation, and result analysis.

Script

Purpose

00_prepare_test_split.py

Prepares the test split for model evaluation

01_train_all_models.py

Trains the selected YOLO models using a consistent setup

02_evaluate_all_models.py

Evaluates trained models on the evaluation data

03_results_table.py

Organizes model performance into comparative tables

audit_class_samples.py

Audits class wise samples and dataset distribution

bbox_editor.py

Supports bounding box inspection and editing

convert_voc_to_yolo.py

Converts Pascal VOC annotations to YOLO format

evaluate_all_models.py

Performs comparative evaluation of multiple models

train_yolo_roc.py

Generates ROC related analysis

train_yolov8_enhanced.py

Trains the enhanced YOLOv8n experiment

Models

The Models directory contains the trained model outputs and associated experimental files for the lightweight YOLO models evaluated in RoadGuard:

YOLOv5n

YOLOv8n

YOLOv10n

These models were evaluated as lightweight object detection architectures suitable for road hazard detection and potential mobile deployment.

Experimental Configuration

The main YOLOv8n training configuration used in the experiments is summarized below.

Parameter

Value

Task

Object Detection

Model

YOLOv8n

Image Size

512 × 512

Epochs

100

Batch Size

24

Optimizer

SGD

Initial Learning Rate (lr0)

0.01

Final Learning Rate Factor (lrf)

0.01

Momentum

0.937

Weight Decay

0.0005

Warmup Epochs

3

Patience

15

Seed

42

Deterministic

True

AMP

True

Device

GPU (0)

Workers

0

Pretrained

True

Validation

Enabled

IoU Threshold

0.7

Maximum Detections

300

Mosaic

1.0

Mixup

0.0

Horizontal Flip

0.5

Translation

0.1

Scale

0.5

HSV Hue

0.015

HSV Saturation

0.7

HSV Value

0.4

Auto Augmentation

RandAugment

Erasing

0.4

Training and Evaluation Workflow

The experimental workflow is organized as follows:

Dataset
   ↓
Annotation Conversion / Verification
   ↓
Test Split Preparation
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Class Wise Analysis
   ↓
Model Comparison
   ↓
Tables, Figures and Results

The scripts are designed to support a consistent comparison of the evaluated YOLO models.

Results

The Results directory contains the experimental outputs and analysis generated during the RoadGuard research process, including model performance comparisons, class wise evaluation, tables, figures, and related result files.

These results support the comparative assessment of the lightweight detection models and the selection of the model used in the RoadGuard system.

Demonstration Video

A demonstration video is included in the repository to show the RoadGuard system operating in a practical road environment.

The video provides a visual representation of the road hazard detection process and the intended real time usage of the system.

Reproducibility

To reproduce the main experiments:

1. Prepare the dataset in YOLO format
2. Configure the dataset YAML file
3. Prepare the test split
4. Train the selected YOLO models
5. Evaluate the trained models
6. Generate comparative tables and analysis

The main experiments use a fixed random seed of 42 and deterministic settings where supported.

RoadGuard Research Scope

This repository covers:

Dataset preparation and annotation processing

Object detection model training

YOLO model comparison

Model evaluation

Class wise performance analysis

ROC related analysis

Experimental result generation

The mobile application, rider warning interface, hazard heatmap, route recommendation, and other application level components are maintained separately in the full RoadGuard application repository.

Citation

If you use this repository or the associated RoadGuard research work in academic research, please cite the corresponding publication.
