Anomaly Detection Using Convolutional Autoencoder (CAE)
This project is a Python package for anomaly detection using a Convolutional Autoencoder (CAE). It includes modules for image processing, dataset loading, custom data loading, and the CAE model, with a structure that supports flexibility and extensibility.

--Project Structure
~ data/ - Data handling (dataset templates, custom dataloaders, image utilities).
~ models/ - Model definitions, including CAE.
~ options/ - Command-line configuration for training/testing.
~ utils/ - Utility functions for directories.
~ train.py - Training script.
~ test.py - Testing script.

--Dataset is in team drive
You can simply add it to anomaly-detection directory
Conda (for managing environments)

git clone https://github.com/your-username/anomaly-detection.git
cd anomaly-detection

Create the environment:

conda env create -f environment.yml
conda activate anomaly-detection

Visualizer Tool
The visualizer.py provides a GUI for displaying images and anomaly results with controls for thresholds and views.

~ ~ Running the Visualizer

python -m visualizer.main

License
This project is licensed under the MIT License. See LICENSE for details.

Credits
This project was inspired by and adapted from the work done by CY-Jeong on their anomaly-detection-mvtec repository. ```

https://github.com/CY-Jeong/anomaly-detection-mvtec/tree/master