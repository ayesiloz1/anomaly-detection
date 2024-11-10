""" Welcome Screen Module for Anomaly Visualizer Pro """

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QApplication, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QIcon
from visualizer.visualizer import AnomalyVisualizer
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import os

# Define the load_cae_model function here
def load_cae_model(encoder_path, decoder_path, latent_dim, img_size):
    from models.networks import Encoder, Decoder
    encoder = Encoder(latent_dim)
    decoder = Decoder((3, img_size, img_size), latent_dim)
    
    # Load the pre-trained model weights
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    
    # Combine the encoder and decoder into a sequential model
    cae_model = torch.nn.Sequential(encoder, decoder)
    cae_model.eval()  # Set the model to evaluation mode
    return cae_model

class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.image_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = 0  # Replace with actual label logic if needed
        return image, label
class WelcomeScreen(QWidget):
    def __init__(self, encoder_path, decoder_path, latent_dim, img_size, device):
        super().__init__()

        # Store these parameters for later use
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.device = device

        # Load the model here once, so it's available for both screens
        self.cae_model = load_cae_model(self.encoder_path, self.decoder_path, self.latent_dim, self.img_size).to(self.device)

        self.initUI()

    def initUI(self):
        # Create a vertical layout
        layout = QVBoxLayout()

        # Set a gradient background
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #f0f4f8, stop: 1 #c0c4c8
                );
            }
        """)

        # Create a horizontal layout for the menu
        menu_layout = QHBoxLayout()

        # Create the anomaly detection button with icon
        anomaly_detection_button = QPushButton(QIcon("image.png"), "Anomaly Detection")
        anomaly_detection_button.clicked.connect(self.open_anomaly_detection)

        # Style the button for the header
        anomaly_detection_button.setFont(QFont("Arial", 14, QFont.Bold))
        anomaly_detection_button.setStyleSheet("""
            QPushButton {
                background-color: #007BFF;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                margin: 5px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                transition: background-color 0.3s, transform 0.3s;
            }
            QPushButton:hover {
                background-color: #0056b3;
                transform: scale(1.05);
            }
            QPushButton:pressed {
                background-color: #003e8b;
            }
        """)
        anomaly_detection_button.setCursor(Qt.PointingHandCursor)

        # Add both buttons to the menu layout
        menu_layout.addWidget(anomaly_detection_button)

        # Add the menu layout to the main layout
        layout.addLayout(menu_layout)

        # Title label with enhanced font
        title_label = QLabel("Anomaly Visualizer Pro")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Verdana", 36, QFont.Bold))

        # Subtitle label with modern font
        subtitle_label = QLabel("Visualize, Detect, Innovate")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setFont(QFont("Verdana", 18))

        # Optional: Add an image/logo (ensure the image path is correct)
        image_label = QLabel(self)
        pixmap = QPixmap("image.png")  # Replace with the path to your logo or image
        pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)

        # Add a "Get Started" button with hover animation
        start_button = QPushButton("Get Started")
        start_button.setFont(QFont("Arial", 18, QFont.Bold))
        start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 20px;
                padding: 15px 30px;
                font-size: 18px;
                border: none;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                transition: all 0.3s ease;
            }
            QPushButton:hover {
                background-color: #45a049;
                transform: scale(1.05);
            }
            QPushButton:pressed {
                background-color: #3c8d40;
            }
        """)
        start_button.clicked.connect(self.close_welcome_screen)
        start_button.setCursor(Qt.PointingHandCursor)

        # Add spacing and alignment for a more appealing layout
        layout.addSpacing(40)  # Space above the logo
        layout.addWidget(image_label)
        layout.addSpacing(20)  # Space between image and title
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addSpacing(40)  # Space between subtitle and button
        layout.addWidget(start_button)

        # Set the layout to the main widget
        self.setLayout(layout)

        # Set window size and title
        self.resize(850, 650)
        self.setWindowTitle("Welcome to Anomaly Visualizer Pro")

        # Center the window on the screen
        self.center()

    def center(self):
        # Center the window on the screen
        frame = self.frameGeometry()
        screen = QApplication.primaryScreen().availableGeometry().center()
        frame.moveCenter(screen)
        self.move(frame.topLeft())

    def close_welcome_screen(self):
        # Close the welcome screen and open the main visualizer
        self.close()

    def open_anomaly_detection(self):
        # Use the preloaded model to launch the AnomalyVisualizer window
        self.window = AnomalyVisualizer(self.cae_model, self.device, self.img_size)
        self.window.show()  # Show the window first
        self.window.center()  # Then center the window
        self.close_welcome_screen()
