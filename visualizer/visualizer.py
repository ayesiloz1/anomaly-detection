"""
# visualizer.py
# Author: Agit Yesiloz
# Description: Main visualizer class for the anomaly detection project. This class is responsible for displaying the images,
anomaly detection results, and various controls to adjust the visualization settings.
It also provides functionality to load images, save images, and save the anomaly counts to a CSV file.
"""

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QGridLayout, QComboBox, QHBoxLayout, QFileDialog
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image
from PyQt5.QtWidgets import QSizePolicy
from .utils import normalize_to_uint8, generate_heatmap, load_images, load_image_at_index, process_all_images_and_save
from .classification import plot_confusion_matrix

class AnomalyVisualizer(QWidget):
    def __init__(self, cae_model, device, img_size):
        super().__init__()
        self.setWindowTitle("Anomaly Visualizer")

        # Load the stylesheet (styles.css) for the UI
        self.load_stylesheet()

        # Create a horizontal layout to split zoom slider and the rest of the UI
        self.main_layout = QHBoxLayout(self)

        # Create the rest of the UI (like images, buttons, etc.)
        self.layout = QVBoxLayout()
        self.cae_model = cae_model.to(device)
        self.device = device
        self.img_size = img_size
        self.anomaly_results = []

        # Initialize anomaly_counts dictionary to store the number of anomalies detected per image
        self.anomaly_counts = {} 

        # Add image selector dropdown
        self.image_selector = QComboBox(self)
        self.image_selector.currentIndexChanged.connect(self.on_image_selected)
        self.image_selector.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        font = self.image_selector.font()
        font.setPointSize(12)  # we can also set the font size of the dropdown items
        self.image_selector.setFont(font)
        self.image_selector.setMinimumSize(200, 40)  # this will set the minimum width and height of the dropdown

        # Dropdown for changing view of our images for both good and reject images
        self.change_view = QComboBox(self)
        self.change_view.addItems(["All", "Original", "Reconstructed", "Residuals", "Anomaly Detection", "Heatmap","Original and Anomaly Detection"])
        self.change_view.currentTextChanged.connect(self.update_images)
        self.change_view.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Buttons to load good and reject images
        self.load_good_button = QPushButton("Load Good Images")
        self.load_good_button.clicked.connect(self.load_good_images)
        self.load_good_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.load_reject_button = QPushButton("Load Reject Images")
        self.load_reject_button.clicked.connect(self.load_reject_images)
        self.load_reject_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Button to save images to local storage
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        #button to save all images in the current view to local storage
        self.save_all_button = QPushButton("Save All Images")
        self.save_all_button.clicked.connect(self.save_all_images)
        self.save_all_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


        # Create a horizontal layout for the image selector, view selector, load buttons, and save button
        self.selector_layout = QHBoxLayout()
        self.selector_layout.addWidget(self.image_selector)
        self.selector_layout.addWidget(self.change_view)
        self.selector_layout.addWidget(self.load_good_button)
        self.selector_layout.addWidget(self.load_reject_button)
        self.selector_layout.addStretch()  # Add a stretchable space
        self.selector_layout.addWidget(self.save_button)
        self.selector_layout.addWidget(self.save_all_button)
        # Add the selector layout to the main layout
        self.layout.addLayout(self.selector_layout)

        # Create grid for images
        self.image_grid = QGridLayout()
        self.layout.addLayout(self.image_grid)

        # Placeholder label
        self.placeholder_label = QLabel("Load Good or Reject Images")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("font-size: 20px; color: #888;")
        self.image_grid.addWidget(self.placeholder_label, 0, 0)

        # Threshold slider with label
        self.threshold_layout = QHBoxLayout()
        self.threshold_label = QLabel("Threshold: 22")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(22)  # we can set the initial value of the slider here
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10) # adjust as you like
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        self.threshold_layout.addWidget(self.threshold_label)
        self.threshold_layout.addWidget(self.threshold_slider)
        self.layout.addLayout(self.threshold_layout)

        # This slider controls the threshold for edge regions
        self.edge_threshold_layout = QHBoxLayout()
        self.edge_threshold_label = QLabel("Edge Threshold Scale: 200%")
        self.edge_threshold_slider = QSlider(Qt.Horizontal)
        self.edge_threshold_slider.setMinimum(0)
        self.edge_threshold_slider.setMaximum(200)
        self.edge_threshold_slider.setValue(200)  # Set scale to 200%
        self.edge_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.edge_threshold_slider.setTickInterval(10)
        self.edge_threshold_slider.valueChanged.connect(self.update_images)
        self.edge_threshold_slider.valueChanged.connect(self.update_edge_threshold_label)
        self.edge_threshold_layout.addWidget(self.edge_threshold_label)
        self.edge_threshold_layout.addWidget(self.edge_threshold_slider)
        self.layout.addLayout(self.edge_threshold_layout)
        
        # This slider controls the threshold for non edge regions
        self.non_edge_threshold_layout = QHBoxLayout()
        self.non_edge_threshold_label = QLabel("Non-Edge Threshold Scale: 200%")
        self.non_edge_threshold_slider = QSlider(Qt.Horizontal)
        self.non_edge_threshold_slider.setMinimum(0)
        self.non_edge_threshold_slider.setMaximum(200)
        self.non_edge_threshold_slider.setValue(200)  # Set scale to 200%
        self.non_edge_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.non_edge_threshold_slider.setTickInterval(178)
        self.non_edge_threshold_slider.valueChanged.connect(self.update_images)
        self.non_edge_threshold_slider.valueChanged.connect(self.update_non_edge_threshold_label)
        self.non_edge_threshold_layout.addWidget(self.non_edge_threshold_label)
        self.non_edge_threshold_layout.addWidget(self.non_edge_threshold_slider)
        self.layout.addLayout(self.non_edge_threshold_layout)

        # Heatmap intensity slider with label to control the intensity of the heatmap overlay
        self.heatmap_intensity_layout = QHBoxLayout()
        self.heatmap_intensity_label = QLabel(f"Heatmap Intensity: {70}")
        self.heatmap_intensity_slider = QSlider(Qt.Horizontal)
        self.heatmap_intensity_slider.setMinimum(0)
        self.heatmap_intensity_slider.setMaximum(100)
        self.heatmap_intensity_slider.setValue(0)
        self.heatmap_intensity_slider.setTickPosition(QSlider.TicksBelow)
        self.heatmap_intensity_slider.setTickInterval(10)
        self.heatmap_intensity_slider.valueChanged.connect(self.update_heatmap_intensity_label)
        self.heatmap_intensity_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.heatmap_intensity_layout.addWidget(self.heatmap_intensity_label)
        self.heatmap_intensity_layout.addWidget(self.heatmap_intensity_slider)
        self.layout.addLayout(self.heatmap_intensity_layout)

        # Anomaly count label to display the number of anomalies detected
        self.anomaly_count_label = QLabel("Anomalies Detected: 0")
        self.anomaly_count_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.anomaly_count_label.setMaximumHeight(30)  # Limit height to avoid stretching
        self.layout.addWidget(self.anomaly_count_label)

        # Navigation buttons for previous and next image
        nav_button_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.prev_button.clicked.connect(self.prev_image)
        self.next_button.clicked.connect(self.next_image)
        self.prev_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.next_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        nav_button_layout.addWidget(self.prev_button)
        nav_button_layout.addWidget(self.next_button)
        self.layout.addLayout(nav_button_layout)

        # Figure size and canvas for images in the grid
        self.fig, self.axs = plt.subplots(1, 4, figsize=(120, 20), dpi=150)  # width, height
        plt.subplots_adjust(wspace=0, hspace=0, left=0.01, right=0.99, top=0.99, bottom=0.01) # Adjust spacing 
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.canvas.setStyleSheet("background-color: #2c3e50;")
        self.image_grid.addWidget(self.canvas, 0, 0)  # this ensures the canvas is added to the grid layout

        # Set aspect ratio to be equal
        for ax in self.axs:
            ax.set_aspect('equal')  # You can also use 'auto' for automatic aspect ratio

        # Save results buttons for good and reject images in the current folder
        save_button_layout = QHBoxLayout()
        self.save_good_button = QPushButton("Save Good Results to CSV")
        self.save_good_button.clicked.connect(lambda: process_all_images_and_save(self, 0))
        save_button_layout.addWidget(self.save_good_button)
        
        self.save_reject_button = QPushButton("Save Reject Results to CSV")
        self.save_reject_button.clicked.connect(lambda: process_all_images_and_save(self, 1))  # Pass self
        save_button_layout.addWidget(self.save_reject_button)
        self.layout.addLayout(save_button_layout)

        # Confusion matrix button to display the classification results
        self.confusion_matrix_button = QPushButton("Show Confusion Matrix")
        self.confusion_matrix_button.clicked.connect(plot_confusion_matrix)
        self.layout.addWidget(self.confusion_matrix_button)

        # Finalize layout
        layout_widget = QWidget()
        layout_widget.setLayout(self.layout)
        self.main_layout.addWidget(layout_widget)

        # Adjust margins and spacing
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(3)
        self.true_labels = []  # To store true labels
        self.predicted_labels = []  # To store predicted labels (0 for good, 1 for reject)

    def load_stylesheet(self):
        """ Load the stylesheet for the UI from the styles.css file. """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        stylesheet_path = os.path.join(script_dir, "styles.css")
        # Open and read the stylesheet file
        with open(stylesheet_path, "r") as f:
            self.setStyleSheet(f.read())

    def center(self):
        """ Center the window on the screen. """
        frame = self.frameGeometry()
        screen = QApplication.primaryScreen().availableGeometry().center()
        frame.moveCenter(screen)
        self.move(frame.topLeft())

    def update_threshold_label(self):
        """ Update the threshold label text and update the images with the new threshold. """
        value = self.threshold_slider.value()
        self.threshold_label.setText(f"Threshold: {value}")
        self.update_images()  # Update images with the new base threshold

    def load_reject_images(self):
        """ Load reject images from the reject folder and update the current folder. """
        self.current_folder = 'reject'
        reject_folder_path = os.path.join(os.getcwd(), "anomaly-detection", "Bowtie", "test", "reject")
        load_images(reject_folder_path, self)  # Pass the required arguments

    def load_good_images(self):
        """ Load good images from the good folder and update the current folder. """
        self.current_folder = 'good'
        good_folder_path = os.path.join(os.getcwd(), "anomaly-detection", "Bowtie", "test", "good")
        load_images(good_folder_path, self)  # Pass the required arguments

    def next_image(self):
        """ Display the next image in the list. """
        if self.image_files and self.current_image_idx < len(self.image_files) - 1:
            self.current_image_idx += 1
            self.image_selector.setCurrentIndex(self.current_image_idx)
            load_image_at_index(self, self.current_image_idx)

    def prev_image(self):
        """ Display the previous image in the list. """
        if self.image_files and self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.image_selector.setCurrentIndex(self.current_image_idx)
            load_image_at_index(self, self.current_image_idx)

    def update_heatmap_intensity_label(self):
        """ Update the heatmap intensity label text and update the images with the new intensity. """
        value = self.heatmap_intensity_slider.value()
        self.heatmap_intensity_label.setText(f"Heatmap Intensity: {value}")
        self.update_images()

    def update_images(self):
        """ Update the displayed images based on the selected view and threshold values. """
        if self.original_img is not None and self.reconstructed_img is not None:
            # Base threshold for non-edge regions
            base_threshold = self.threshold_slider.value()
    
            # Compute residuals (difference)
            original_img_float = self.original_img.astype(np.float32)
            reconstructed_img_float = self.reconstructed_img.astype(np.float32)
            residuals = np.abs(original_img_float - reconstructed_img_float)
            
            # Increase smoothing for detecting larger anomalies
            smoothed_residuals = cv2.GaussianBlur(residuals, (11, 11), 0)  # you can increase this if you would like to smooth more or vice versa
    
            # Convert to grayscale for thresholding
            residuals_gray = np.mean(smoothed_residuals, axis=2)

            """ Edge-aware anomaly detection logic:
            1. Detect edges using Canny edge detection(https://en.wikipedia.org/wiki/Canny_edge_detector)
            2. Dilate the edges to capture more edge pixels for thresholding
            3. Create edge and non-edge masks based on the detected edges
               """
            edges = cv2.Canny(cv2.cvtColor(self.original_img, cv2.COLOR_RGB2GRAY), 50, 150) # 50 is the lower threshold and 150 is the upper threshold
            edge_mask = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2) > 0  # Increased dilation for edges
            non_edge_mask = ~edge_mask
    
            """ Thresholding and anomaly detection:
            1. Compute dynamic thresholds for edge and non-edge regions based on the residuals
            2. Apply the thresholds to create binary anomaly masks
            3. Apply morphological operations to expand anomaly areas for better visualization"""
            # Get edge and non-edge values from residuals
            edge_values = residuals_gray[edge_mask]
            non_edge_values = residuals_gray[non_edge_mask]
    
            # Get scaling factors for edge and non-edge regions from sliders
            edge_threshold_scale = self.edge_threshold_slider.value() / 100.0
            non_edge_threshold_scale = self.non_edge_threshold_slider.value() / 100.0
    
            # Compute dynamic thresholds with adjustments to capture larger areas
            edge_threshold = max(np.mean(edge_values) + edge_threshold_scale * np.std(edge_values), base_threshold * 3)
            non_edge_threshold = max(np.mean(non_edge_values) + non_edge_threshold_scale * np.std(non_edge_values), base_threshold * 2)
    
            # Apply thresholds to create binary anomaly masks
            edge_anomaly_mask = (residuals_gray > edge_threshold) & edge_mask
            non_edge_anomaly_mask = (residuals_gray > non_edge_threshold) & non_edge_mask
            final_anomaly_mask = edge_anomaly_mask | non_edge_anomaly_mask
    
            # Apply morphological operations to expand anomaly areas for better visualization
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  #increase this if you want to expand the anomalies more, decrease if you want to shrink
            processed_mask = cv2.morphologyEx(final_anomaly_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
            # Count detected anomalies
            anomaly_count = np.sum(processed_mask)
            self.anomaly_count_label.setText(f"Anomalies Detected: {anomaly_count}")
    
            # Store anomaly count
            current_image_path = self.image_files[self.current_image_idx]
            self.anomaly_counts[current_image_path] = anomaly_count
    
            # Visualize anomaly overlay and heatmap
            anomaly_overlay = self.original_img.copy()
            anomaly_overlay[processed_mask.astype(bool)] = [255, 0, 0]
            blend_factor = self.heatmap_intensity_slider.value() / 100.0
            blended_heatmap = generate_heatmap(residuals_gray, self.original_img, blend_factor)
    
            # Display images based on selected view
            selected_view = self.change_view.currentText()
            self.fig.clear()
    
            if selected_view == "Original":
                ax = self.fig.add_subplot(111)
                self.display_image(ax, normalize_to_uint8(original_img_float), "Original")
            elif selected_view == "Reconstructed":
                ax = self.fig.add_subplot(111)
                self.display_image(ax, normalize_to_uint8(reconstructed_img_float), "Reconstructed")
            elif selected_view == "Residuals":
                ax = self.fig.add_subplot(111)
                self.display_image(ax, normalize_to_uint8(smoothed_residuals), "Residuals")
            elif selected_view == "Anomaly Detection":
                ax = self.fig.add_subplot(111)
                self.display_image(ax, anomaly_overlay, "Anomaly Overlay")
            elif selected_view == "Heatmap":
                ax = self.fig.add_subplot(111)
                self.display_image(ax, blended_heatmap, "Heatmap")
            elif selected_view == "Original and Anomaly Detection":
                # Clear the figure to ensure no residual plots remain
                self.fig.clear()
                
                # Set a larger figure size for larger images
                self.fig.set_size_inches(10, 5)  # Adjust the size as needed (width, height)

                # Create subplots for Original and Anomaly Detection views
                ax1 = self.fig.add_subplot(1, 2, 1)
                ax2 = self.fig.add_subplot(1, 2, 2)

                # Display the Original image in the first subplot
                ax1.imshow(normalize_to_uint8(original_img_float))
                ax1.set_title("Original", fontsize=14)  # Larger font size for titles
                ax1.axis('off')  # Hide axis ticks

                # Display the Anomaly Detection overlay in the second subplot
                ax2.imshow(anomaly_overlay)
                ax2.set_title("Anomaly Overlay", fontsize=14)
                ax2.axis('off')  # Hide axis ticks

                # Adjust spacing to remove gap between subplots
                plt.subplots_adjust(wspace=0, hspace=0)  # Set wspace and hspace to 0 to remove gaps
                plt.tight_layout()  # Adjust layout to remove extra spaces
                # Draw the canvas to update the display
                self.canvas.draw()


            else:
                self.axs = self.fig.subplots(1, 4)
                self.display_image(self.axs[0], normalize_to_uint8(original_img_float), "Original")
                self.display_image(self.axs[1], normalize_to_uint8(reconstructed_img_float), "Reconstructed")
                self.display_image(self.axs[2], normalize_to_uint8(smoothed_residuals), "Residuals")
                self.display_image(self.axs[3], anomaly_overlay, "Anomaly Overlay")
    
            self.canvas.draw()
 
    def display_image(self, ax, img, title, cmap=None):
        """ Display the image in the specified axis with the given title and colormap. """
        ax.clear()
        """
        colors:
        b: blue
        g: green
        r: red
        """
        ax.set_facecolor('#2c3e50') # Set background color of the axis to a dark blue
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    def on_image_selected(self, index):
        """ Load the selected image when the image selector dropdown value changes. """
        self.current_image_idx = index
        load_image_at_index(self, self.current_image_idx)

    def save_image(self):
        """ Save the currently displayed image based on the selected view. """
        if self.original_img is None:
            print("No image to save!")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)", options=options)
        if not file_path:
            return

        # Retrieve the current threshold values from the sliders
        base_threshold = self.threshold_slider.value()
        edge_threshold_scale = self.edge_threshold_slider.value() / 100.0
        non_edge_threshold_scale = self.non_edge_threshold_slider.value() / 100.0

        # Process the image based on the selected view
        selected_view = self.change_view.currentText()
        if selected_view == "Original":
            img_to_save = self.original_img
        elif selected_view == "Reconstructed":
            img_to_save = self.reconstructed_img
        elif selected_view == "Residuals":
            residuals = np.abs(self.original_img.astype(np.float32) - self.reconstructed_img.astype(np.float32))
            img_to_save = normalize_to_uint8(residuals)  # Normalize residuals to save
        elif selected_view == "Anomaly Detection":
            # Apply anomaly detection processing with current threshold values
            residuals = np.abs(self.original_img.astype(np.float32) - self.reconstructed_img.astype(np.float32))
            residuals_gray = np.mean(residuals, axis=2)

            # Apply dynamic thresholding based on edge and non-edge regions
            edges = cv2.Canny(cv2.cvtColor(self.original_img, cv2.COLOR_RGB2GRAY), 50, 150)
            edge_mask = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2) > 0
            non_edge_mask = ~edge_mask

            # Calculate thresholds for edge and non-edge regions
            edge_values = residuals_gray[edge_mask]
            non_edge_values = residuals_gray[non_edge_mask]
            edge_threshold = max(np.mean(edge_values) + edge_threshold_scale * np.std(edge_values), base_threshold * 3)
            non_edge_threshold = max(np.mean(non_edge_values) + non_edge_threshold_scale * np.std(non_edge_values), base_threshold * 2)

            # Create binary anomaly mask
            edge_anomaly_mask = (residuals_gray > edge_threshold) & edge_mask
            non_edge_anomaly_mask = (residuals_gray > non_edge_threshold) & non_edge_mask
            final_anomaly_mask = edge_anomaly_mask | non_edge_anomaly_mask

            # Apply mask to original image
            anomaly_overlay = self.original_img.copy()
            anomaly_overlay[final_anomaly_mask] = [255, 0, 0]  # Mark anomalies in red
            img_to_save = anomaly_overlay
        elif selected_view == "Heatmap":
            residuals = np.abs(self.original_img.astype(np.float32) - self.reconstructed_img.astype(np.float32))
            residuals_gray = np.mean(residuals, axis=2)
            blend_factor = self.heatmap_intensity_slider.value() / 100.0
            img_to_save = generate_heatmap(residuals_gray, self.original_img, blend_factor)
        elif selected_view == "Original and Anomaly Detection":
            # Same anomaly overlay processing as above
            residuals = np.abs(self.original_img.astype(np.float32) - self.reconstructed_img.astype(np.float32))
            residuals_gray = np.mean(residuals, axis=2)
            
            # Create edge and non-edge anomaly masks with current thresholds
            edges = cv2.Canny(cv2.cvtColor(self.original_img, cv2.COLOR_RGB2GRAY), 50, 150)
            edge_mask = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2) > 0
            non_edge_mask = ~edge_mask

            edge_values = residuals_gray[edge_mask]
            non_edge_values = residuals_gray[non_edge_mask]
            edge_threshold = max(np.mean(edge_values) + edge_threshold_scale * np.std(edge_values), base_threshold * 3)
            non_edge_threshold = max(np.mean(non_edge_values) + non_edge_threshold_scale * np.std(non_edge_values), base_threshold * 2)

            edge_anomaly_mask = (residuals_gray > edge_threshold) & edge_mask
            non_edge_anomaly_mask = (residuals_gray > non_edge_threshold) & non_edge_mask
            final_anomaly_mask = edge_anomaly_mask | non_edge_anomaly_mask

            anomaly_overlay = self.original_img.copy()
            anomaly_overlay[final_anomaly_mask] = [255, 0, 0]
            img_to_save = np.concatenate((self.original_img, anomaly_overlay), axis=1)
        else:
            print("Invalid view selected!")
            return

        # Save the image
        img_to_save_pil = Image.fromarray(img_to_save)
        img_to_save_pil.save(file_path)
        print(f"Image saved successfully at {file_path}")



    def save_all_images(self):
        """ Save side-by-side images (Original and Anomaly Detection) with the exact current threshold settings, using only the original file names. """
        if self.original_img is None:
            print("No images to save!")
            return

        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Save Images", options=options)
        if not folder_path:
            return

        # Loop through all images to save them with current settings
        for idx, img_path in enumerate(self.image_files):
            self.current_image_idx = idx
            load_image_at_index(self, self.current_image_idx)

            # Get the original file name without the directory path
            original_file_name = os.path.basename(img_path)
            file_name, file_ext = os.path.splitext(original_file_name)

            # Retrieve the current threshold values from the sliders
            base_threshold = self.threshold_slider.value()
            edge_threshold_scale = self.edge_threshold_slider.value() / 100.0
            non_edge_threshold_scale = self.non_edge_threshold_slider.value() / 100.0

            # Perform anomaly detection processing as in update_images
            original_img_float = self.original_img.astype(np.float32)
            reconstructed_img_float = self.reconstructed_img.astype(np.float32)
            residuals = np.abs(original_img_float - reconstructed_img_float)
            residuals_gray = np.mean(residuals, axis=2)

            # Edge-aware anomaly detection logic
            edges = cv2.Canny(cv2.cvtColor(self.original_img, cv2.COLOR_RGB2GRAY), 50, 150)
            edge_mask = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2) > 0
            non_edge_mask = ~edge_mask

            # Calculate dynamic thresholds
            edge_values = residuals_gray[edge_mask]
            non_edge_values = residuals_gray[non_edge_mask]
            edge_threshold = max(np.mean(edge_values) + edge_threshold_scale * np.std(edge_values), base_threshold * 3)
            non_edge_threshold = max(np.mean(non_edge_values) + non_edge_threshold_scale * np.std(non_edge_values), base_threshold * 2)

            # Create binary anomaly masks
            edge_anomaly_mask = (residuals_gray > edge_threshold) & edge_mask
            non_edge_anomaly_mask = (residuals_gray > non_edge_threshold) & non_edge_mask
            final_anomaly_mask = edge_anomaly_mask | non_edge_anomaly_mask

            # Apply the mask to create the anomaly overlay
            anomaly_overlay = self.original_img.copy()
            anomaly_overlay[final_anomaly_mask] = [255, 0, 0]  # Mark anomalies in red

            # Combine Original and Anomaly Detection overlay side-by-side, as shown in update_images
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(normalize_to_uint8(original_img_float))
            axs[0].set_title("Original")
            axs[0].axis("off")

            axs[1].imshow(anomaly_overlay)
            axs[1].set_title("Anomaly Overlay")
            axs[1].axis("off")

            # Save the combined figure as a single image using the original file name
            save_path = os.path.join(folder_path, f"{file_name}{file_ext}")
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            print(f"Saved {save_path}")


    def update_edge_threshold_label(self):
        """ Update the edge threshold label text and update the images with the new scale. """
        value = self.edge_threshold_slider.value()
        self.edge_threshold_label.setText(f"Edge Threshold Scale: {value}%")
        self.update_images()  

    def update_non_edge_threshold_label(self):
        """ Update the non-edge threshold label text and update the images with the new scale. """
        value = self.non_edge_threshold_slider.value()
        self.non_edge_threshold_label.setText(f"Non-Edge Threshold Scale: {value}%")
        self.update_images()