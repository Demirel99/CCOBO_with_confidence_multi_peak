# Iterative Autoregressive Crowd Counting

This project implements an iterative, autoregressive approach to crowd counting. Unlike traditional methods that regress a complete density map from an image in a single pass, this model sequentially identifies one person at a time within an image patch.

The core idea is to treat crowd counting as a sequence generation problem. At each step, the model is given an image and a map of previously detected people. It then performs two tasks:
1.  **Predicts the location of the *next* person** as a Probability Spread Function (PSF).
2.  **Predicts a confidence score** indicating whether it believes there are more people left to find in the patch.

This iterative process continues until the model's confidence drops below a predefined threshold, signaling that all individuals in the patch have likely been located.

## Key Features

-   **Autoregressive Framework**: The model's prediction at step `k` is conditioned on its own predictions from steps `1` to `k-1`.
-   **Dual-Output Head**: The model simultaneously predicts *where* the next person is (PSF map) and *if* there is a next person (confidence score).
-   **High-Resolution Feature Fusion**: Utilizes a VGG19 backbone with a Feature Pyramid Network (FPN) and Atrous Spatial Pyramid Pooling (ASPP) to generate rich, multi-scale features for precise localization.
-   **Confidence-Based Early Stopping**: Inference on a patch terminates automatically when the model is no longer confident about finding new people, making the process efficient.
-   **Dynamic Training Sample Generation**: Training data is generated on-the-fly, creating input/target pairs that simulate the iterative detection process.
-   **Combined Loss Function**: The model is trained using a composite loss that combines KL-Divergence for the location prediction and Binary Cross-Entropy for the confidence prediction.

## How It Works

### Training (`generate_train_sample`)

The model is trained to predict the next point in an ordered sequence of ground-truth annotations. For each training sample:
1.  An image patch and its ground truth points are selected. The points are sorted to create a deterministic order (e.g., top-to-bottom, left-to-right).
2.  A random step `k` is chosen.
3.  **Model Input**: The image patch is combined with an *input PSF*, which is a map representing all ground truth points from `1` to `k-1`.
4.  **Model Targets**:
    -   The **target PSF** is a map representing only the `k`-th point.
    -   The **target confidence** is `1.0` (a "positive" sample, meaning "yes, there is a next person").
5.  **Negative Sampling**: To teach the model when to stop, "negative" samples are also created. For these, the *input PSF* contains *all* ground truth points, the **target PSF** is empty, and the **target confidence** is `0.0`.

### Inference (`main_test_non_overlapping_whole_image.py`)

To count people in a full image:
1.  The image is padded and divided into non-overlapping patches of the model's input size (e.g., 224x224).
2.  For each patch, an iterative detection loop begins with an empty "found points" map.
3.  **Iteration Step**:
    -   The model receives the image patch and the current map of "found points".
    -   It outputs a predicted location map (PSF) and a confidence score.
    -   **Decision**: If the confidence score is above a threshold (e.g., 0.9):
        -   The location of the highest peak in the predicted PSF is identified.
        -   This new point is added to the "found points" map.
        -   The loop continues to the next iteration.
    -   **Stopping**: If the confidence is below the threshold, the loop for that patch terminates.
4.  After processing all patches, the coordinates from all patches are collected.
5.  **Non-Maximum Suppression (NMS)** is applied to the collected points to remove duplicate detections near patch borders.
6.  The final count is the number of points remaining after NMS.

## Model Architecture (`VGG19FPNASPP`)

The model is composed of several key modules:
-   `VGG19Encoder`: Uses a pre-trained VGG19 to extract multi-scale feature maps (C1 to C5) from the input image.
-   `SmallPSFEncoder`: A shallow CNN that encodes the 1-channel input PSF map (the "history" of found points).
-   **Fusion Block**: The highest-level image features (C5) are concatenated with the encoded PSF features. This is the crucial step where the model combines visual information with its "memory" of prior detections.
-   `ASPP`: Atrous Spatial Pyramid Pooling is applied to the fused features to capture context at multiple scales without losing spatial resolution.
-   `FPNDecoder`: A Feature Pyramid Network decoder takes the ASPP output and the lower-level VGG features (C1-C4) to produce a high-resolution, semantically rich feature map.
-   `PSFHead`: A final two-branch head that takes the FPN output and produces:
    1.  A `(B, 1, H, W)` **PSF map** via a series of convolutions followed by a spatial Softmax.
    2.  A `(B, 1)` **confidence logit** via global average pooling and linear layers.

## File Breakdown

-   `config.py`: Centralizes all configuration settings, including dataset paths, model parameters (input size, temperature), and training hyperparameters (learning rate, batch size).
-   `dataset.py`: Handles all data loading and preparation. Its main function, `generate_train_sample`, implements the core logic for creating the iterative training pairs described above. It also includes data augmentation functions.
-   `losses.py`: Defines the `combined_loss` function, which calculates the weighted sum of the KL-Divergence loss for the PSF and the `BCEWithLogitsLoss` for the confidence score.
-   `model.py`: Contains the PyTorch definitions for all neural network modules, including the main `VGG19FPNASPP` model and its sub-components (encoders, ASPP, FPN, heads).
-   `train.py`: The main script for training the model. It sets up the model, optimizer, and scheduler, runs the training loop, periodically evaluates on a validation set, logs progress, and saves the best model.
-   `utils.py`: Provides helper functions for setting random seeds, finding file paths, and splitting data into training and validation sets.
-   `main_test_script_non_overlapping.py`: A testing script designed to run the iterative inference process on a *single patch* of an image and visualize the step-by-step predictions.
-   `main_test_non_overlapping_whole_image.py`: A testing script that processes a *full image*. It performs the complete pipeline of padding, patching, iterative inference per patch, aggregating results, applying NMS, and comparing the final count against the ground truth.
