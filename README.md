# EEG Image Classification for Seizure and Non-Seizure Detection

Train a convolutional neural network (CNN) to classify EEG images as seizure or non-seizure, and explore its predictions with LIME and SHAP explanation scripts.

> **Research use only:** This project is experimental and is not validated for clinical diagnosis or medical decision-making.

## Overview

This project uses TensorFlow/Keras for binary image classification and includes scripts for exploring model predictions.

Although the repository title refers to EEG signal analysis, the current implementation processes RGB images—not raw EEG time-series data. Converting raw EEG recordings into images is outside the documented workflow.

### Features

- CNN training and evaluation in a Jupyter notebook.
- Image preprocessing with resizing and pixel normalization.
- LIME explanations highlighting image regions associated with predictions.
- Descriptive image statistics, including mean, variance, skewness, and kurtosis.
- SHAP visualizations generated through a statistical-feature proxy.

## Repository Structure

```text
.
├── app.ipynb          # Training, evaluation, and analysis notebook
├── lime2.py           # LIME explanations and image-statistics analysis
├── shap2.py           # SHAP analysis using a nearest-neighbor proxy
├── lime_explanations/ # LIME output artifacts
└── shap_explanations/ # SHAP output artifacts
```

The dataset and trained model are not included in the repository.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/PiyushVIT346/EEG-SIGNAL-ANALYSIS-FOR-SEIZURE-AND-NON-SEIZURE-CLASSIFICATION-.git
cd EEG-SIGNAL-ANALYSIS-FOR-SEIZURE-AND-NON-SEIZURE-CLASSIFICATION-
```

### 2. Prepare your Python environment

Create an isolated environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

The project uses TensorFlow/Keras, NumPy, Pillow, plotting libraries, LIME, SHAP, and Jupyter.

The repository does not currently provide a dependency manifest or tested package versions. Review the imports in the notebook and scripts before installing dependencies; compatibility with the latest releases has not been verified.

### 3. Prepare your dataset

1. Obtain labeled seizure and non-seizure images.
2. Separate the images into training, validation, and test sets.
3. Update the dataset paths in `app.ipynb` to point to your local folders.
4. Check that the class labels remain consistent between training and explanation scripts.

For medical data, verify that images from the same patient do not cross dataset splits. The repository does not establish whether the recorded experiment used patient-independent splitting.

### 4. Check preprocessing and training settings

Resolve these issues before running the notebook:

- **Image dimensions:** PIL resizing to `(280, 274)` produces arrays shaped `(274, 280, 3)`, but the model declares `(280, 274, 3)`. Use consistent height and width ordering throughout training and inference.
- **Validation steps:** The notebook's integer-divided step calculation produces zero validation steps for the saved dataset configuration. Remove explicit step counts where Keras can infer them.
- **Local paths:** Replace hard-coded dataset and model paths with paths available on your machine.

### 5. Run the notebook

Start Jupyter:

```bash
jupyter notebook app.ipynb
```

Run the cells in order after completing the configuration checks.

The notebook loads and normalizes images, trains the CNN, and evaluates it on the test set. A successful run should produce training output and evaluation metrics; your results may differ from the saved outputs.

## Model Architecture

The classifier contains:

1. Two convolution and pooling blocks.
2. A dense layer with 64 units.
3. A two-class softmax output layer.

Softmax produces scores for the two classes. Verify the class-index mapping before interpreting predictions.

## Recorded Results

The saved notebook outputs report:

| Metric | Recorded value |
| --- | --- |
| Training images | 480 |
| Validation images | 60 |
| Test images | 60 |
| Test accuracy | 98.33% |

These values describe the saved experiment, not independently reproduced performance. The input-shape and training-step issues should be resolved before using these results as a benchmark.

## Optional: Generate LIME Explanations

LIME (Local Interpretable Model-Agnostic Explanations) highlights image regions associated with an individual prediction.

1. Make a compatible saved Keras model available.
2. Update the model and image paths in `lime2.py`.
3. Check preprocessing and class-index compatibility.
4. Provide at least five readable images for the script's sample selection.
5. Run:

```bash
python lime2.py
```

The script writes explanation figures, image statistics, and distribution plots to `lime_explanations/`.

**Interpretation limit:** The aggregate chart averages importance weights by rank across different images. Its numbered regions do not represent shared spatial regions and should not be treated as global region importance.

## Optional: Generate SHAP Visualizations

SHAP (SHapley Additive exPlanations) assigns contributions to the inputs of the prediction function being explained.

1. Make the expected `model.h5` file available.
2. Update model and image paths in `shap2.py`.
3. Verify that the model has two outputs and that output index `1` represents the intended class.
4. Run:

```bash
python shap2.py
```

The script writes SHAP values and visualization artifacts to `shap_explanations/`.

**Interpretation limit:** This script explains a nearest-neighbor proxy built from image statistics. For a perturbed statistics vector, it selects an existing image and returns that image's CNN prediction. Its SHAP values therefore explain the proxy—not direct contributions of those statistics to the CNN.

## Troubleshooting

| Problem | Action |
| --- | --- |
| Dataset or model file not found | Update hard-coded paths and confirm that the files exist locally. |
| Input-shape error | Align PIL resize dimensions with the model's expected height, width, and channels. |
| Zero validation steps or input-exhaustion warning | Review explicit step counts and let Keras infer them where possible. |
| LIME sampling error | Ensure at least five images load successfully. |
| SHAP output-index error | Check the model's output shape; the script assumes a two-class output. |
| Results differ between runs | Review data ordering, random sampling, package versions, and dataset splits. |





Resolving Conflicts
0 of 0 resolved
