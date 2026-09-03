This repository contains the research code, model training scripts,
evaluation tools, analysis, and experimental results for RoadGuard,
a road hazard detection and rider safety system.

The repository is focused on the machine learning and data analysis
component of RoadGuard rather than the complete mobile application.

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

The Codes directory contains the main scripts used throughout the
experimental workflow.

Script                             Purpose

00_prepare_test_split.py         Prepares the test split for model
evaluation

01_train_all_models.py           Trains the selected YOLO models
using a consistent experimental
setup

02_evaluate_all_models.py        Evaluates trained models on the
test/validation data

03_results_table.py              Collects and organizes model
performance results into tables

audit_class_samples.py           Audits class-wise samples and
dataset distribution

bbox_editor.py                   Supports bounding-box
inspection/editing

convert_voc_to_yolo.py           Converts Pascal VOC annotations to
YOLO format

evaluate_all_models.py           Performs comparative evaluation of
multiple trained models

train_yolo_roc.py                Generates ROC-related analysis for
the detection experiments

Models

The Models directory contains the trained model outputs and related
files for the main lightweight YOLO architectures evaluated in
RoadGuard:

YOLOv5n

YOLOv8n

YOLOv10n

These models were selected primarily for their suitability for
lightweight road-hazard detection and potential mobile deployment.

Experimental Configuration

The following configuration represents the main YOLOv8n training setup
used in the experiments.

Parameter                            Value

Task                                 Object Detection
Model                                YOLOv8n
Image Size                           512 × 512
Epochs                               100
Batch Size                           24
Optimizer                            SGD
Initial Learning Rate (lr0)        0.01
Final Learning Rate Factor (lrf)   0.01
Momentum                             0.937
Weight Decay                         0.0005
Warmup Epochs                        3
Patience                             15
Seed                                 42
Deterministic                        True
AMP                                  True
Device                               GPU (0)
Workers                              0
Pretrained                           True
Validation                           Enabled
IoU Threshold                        0.7
Maximum Detections                   300
Mosaic Augmentation                  1.0
Mixup                                0.0
Horizontal Flip                      0.5
Translation                          0.1
Scale                                0.5
HSV Hue                              0.015
HSV Saturation                       0.7
HSV Value                            0.4
Auto Augmentation                    RandAugment
Erasing                              0.4

Training and Evaluation Workflow

The experimental workflow follows these main stages:

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
Class-wise Analysis
   ↓
Model Comparison
   ↓
Tables, Figures and Results

The scripts are organized to support reproducible comparison between the
selected YOLO models under a consistent experimental setup.

Results

The Results directory contains the generated experimental outputs and
analysis used for the RoadGuard research work, including model
performance comparisons, class-wise evaluation, tables, and related
result files.

The results can be used to compare the detection performance of the
evaluated lightweight models and identify the model selected for the
RoadGuard deployment pipeline.

Demonstration

The repository includes a demonstration video showing the RoadGuard
system operating in a practical road environment.

The video provides a visual demonstration of the road-hazard detection
process and the system's intended real-time usage.

Reproducibility

To reproduce the main experiments:

Prepare the dataset in YOLO format.

Configure the dataset YAML file.

Run the test-split preparation script.

Train the selected models using the training scripts.

Evaluate the trained models.

Generate comparative tables and analysis.

The experiments use a fixed random seed of 42 and deterministic
training settings where supported.

RoadGuard

RoadGuard combines lightweight computer vision with mobile sensing to
support road-hazard detection and rider safety. The research component
in this repository focuses on the dataset preparation, object
detection models, training, evaluation, comparative analysis, and
experimental results.

Full RoadGuard Application

The complete source code of the RoadGuard application, including the
mobile application implementation and system features, is available in
the main application repository:

RoadGuard: Hazard Detection, Rider Warning, Heatmap and
YOLOv8n

This research repository contains the model training, evaluation, data
analysis, experiments, and results, while the repository above
contains the full application code and implementation.

Citation

If you use this repository or the associated RoadGuard research work in
academic research, please cite the corresponding publication.
