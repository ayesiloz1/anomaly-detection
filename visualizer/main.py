""" 
Main script to run the visualizer application.
Loads the CAE model and starts the PyQt application.

"""

from PyQt5.QtWidgets import QApplication
import torch
import sys
from visualizer.welcome_screen import WelcomeScreen
from visualizer.visualizer import AnomalyVisualizer


# Define the load_cae_model function here
def load_cae_model(encoder_path, decoder_path, latent_dim, img_size):
    from models.networks import Encoder, Decoder  # Import the encoder/decoder from networks
    encoder = Encoder(latent_dim)
    decoder = Decoder((3, img_size, img_size), latent_dim)
    
    # Load the pre-trained model weights
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    
    # Combine the encoder and decoder into a sequential model
    cae_model = torch.nn.Sequential(encoder, decoder)
    cae_model.eval()  # Set the model to evaluation mode
    return cae_model

if __name__ == '__main__':
    latent_dim = 100  # Adjust this based on your model architecture
    img_size = 256  # Image size used for training the model
    encoder_path = 'train_results/cae/models/cae_encoder.pth' 
    decoder_path = 'train_results/cae/models/cae_decoder.pth'  

    # Set the device to CUDA if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Start the PyQt application
    app = QApplication(sys.argv)

    # Initialize the WelcomeScreen first
    welcome_screen = WelcomeScreen(encoder_path, decoder_path, latent_dim, img_size, device)  # Pass the required params to WelcomeScreen
    welcome_screen.show()

    sys.exit(app.exec_())
    cae_model = load_cae_model(encoder_path, decoder_path, latent_dim, img_size).to(device)

    # Start the PyQt application
    app = QApplication(sys.argv)

    # Show MetricsScreen with CAE model and parameters
    metrics_screen = MetricsScreen(cae_model, device, img_size)
    metrics_screen.show()

    sys.exit(app.exec_())