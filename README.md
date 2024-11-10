# Anomaly Detection Using Convolutional Autoencoder (CAE)

This project is a Python package for anomaly detection using a Convolutional Autoencoder (CAE). It includes modules for image processing, dataset loading, custom data loading, and the CAE model, designed to be flexible and extensible for various deep learning applications.

## Project Structure

- **data/**: Handles data-related tasks, including dataset templates, custom dataloaders, and image utilities.
- **models/**: Contains model definitions, including the CAE model for anomaly detection.
- **options/**: Command-line configurations for training and testing.
- **utils/**: Utility functions for directory management.
- **train.py**: Script to train the CAE model.
- **test.py**: Script to test the CAE model.

## Dataset

The dataset is located in the team drive. To use it, simply add it to the anomaly-detection directory after cloning the repository.

## Installation

### Prerequisites

- Conda (for managing environments)

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/ayesiloz1/anomaly-detection.git
    cd anomaly-detection
    ```

2. Create and activate the environment:
    ```bash
    conda env create -f environment.yml
    conda activate anomaly-detection
    ```

## Visualizer Tool

The project includes a `visualizer.py` tool that provides a GUI for viewing images and anomaly detection results, with interactive controls for thresholds and display options.

### Running the Visualizer

To launch the visualizer, use:
```bash
python -m visualizer.main

License
This project is licensed under the MIT License. See the LICENSE file for details.

Credits
This project was inspired by and adapted from the work by CY-Jeong in their anomaly-detection-mvtec repository. 