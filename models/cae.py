from .base_model import BaseModel
from . import networks
import torch
from utils import utils
from models import init_net
import os
from PIL import Image

class CAE(BaseModel):
    """This class implements the Convolutional AutoEncoder (CAE) for anomaly detection.
    The CAE processes an input image through an encoder and decoder to reconstruct the image,
    and anomaly detection is performed by comparing the reconstructed image with the original.
    """

    @staticmethod
    def add_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        return parser

    def __init__(self, opt):
        """Initialize the CAE model."""
        BaseModel.__init__(self, opt)
        self.opt = opt
        img_size = (self.opt.channels, self.opt.img_size, self.opt.img_size)
        latent = self.opt.latent

        # Initialize encoder and decoder
        if opt.gpu != -1 and torch.cuda.is_available():
            self.encoder = init_net(networks.Encoder(latent).cuda(), gpu=opt.gpu, mode=opt.mode)
            self.decoder = init_net(networks.Decoder(img_size, latent).cuda(), gpu=opt.gpu, mode=opt.mode)
        else:
            self.encoder = init_net(networks.Encoder(latent), gpu=opt.gpu, mode=opt.mode)
            self.decoder = init_net(networks.Decoder(img_size, latent), gpu=opt.gpu, mode=opt.mode)

        self.networks = ['encoder', 'decoder']
        self.criterion = torch.nn.MSELoss()
        self.visual_names = ['generated_imgs']
        self.model_name = self.opt.model
        self.loss_name = ['loss']

        # Initialize optimizers if training
        if self.opt.mode == 'train':
            self.optimizer_e = torch.optim.Adam(self.encoder.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizer_d = torch.optim.Adam(self.decoder.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_e)
            self.optimizers.append(self.optimizer_d)
            self.set_requires_grad(self.decoder, self.encoder, requires_grad=True)

    def forward(self):
        """Forward pass through the encoder and decoder to reconstruct the image."""
        features = self.encoder(self.real_imgs)
        self.generated_imgs = self.decoder(features)

    def backward(self):
        """Compute loss and backpropagate."""
        self.loss = self.criterion(10 * self.real_imgs, 10 * self.generated_imgs)  # Scaling factor of 10 for loss
        self.loss.backward()

    def set_input(self, input):
        """Set the input data for the model."""
        self.real_imgs = input['img'].to(self.device)

    def train(self):
        """Perform a training step."""
        self.forward()
        self.optimizer_d.zero_grad()
        self.optimizer_e.zero_grad()
        self.backward()
        self.optimizer_d.step()
        self.optimizer_e.step()

    def test(self):
        """Perform a testing step (no gradient calculations)."""
        with torch.no_grad():
            self.forward()

    def save_networks(self):
        """Save the encoder and decoder networks to the specified folder."""
        save_dir = os.path.join(self.opt.save_dir, 'cae', 'models')
        utils.mkdirs(save_dir)

        # Save encoder and decoder model parameters
        save_encoder_path = os.path.join(save_dir, f'{self.model_name}_encoder.pth')
        save_decoder_path = os.path.join(save_dir, f'{self.model_name}_decoder.pth')

        torch.save(self.encoder.state_dict(), save_encoder_path)
        torch.save(self.decoder.state_dict(), save_decoder_path)
        print(f"Models saved: {save_encoder_path}, {save_decoder_path}")

    def load_networks(self):
        """Load the encoder and decoder networks from the specified folder."""
        load_dir = os.path.join(self.opt.save_dir, 'cae', 'models')

        # Load encoder and decoder models
        load_encoder_path = os.path.join(load_dir, f'{self.model_name}_encoder.pth')
        load_decoder_path = os.path.join(load_dir, f'{self.model_name}_decoder.pth')

        self.encoder.load_state_dict(torch.load(load_encoder_path))
        self.decoder.load_state_dict(torch.load(load_decoder_path))
        print(f"Models loaded: {load_encoder_path}, {load_decoder_path}")

    def save_loss(self, epoch):
        """Save the loss for each epoch in train_results/cae/loss_history.txt."""
        # Set the directory and file path for saving the loss
        loss_file = os.path.join(self.opt.save_dir, 'cae', 'loss_history.txt')
        
        # Ensure the directory exists
        loss_dir = os.path.dirname(loss_file)
        if not os.path.exists(loss_dir):
            os.makedirs(loss_dir)
            print(f"Directory {loss_dir} created.")
        
        # Append the loss for the current epoch to the file
        with open(loss_file, 'a') as f:
            f.write(f"Epoch {epoch}, Loss: {self.loss.item():.6f}\n")
        
        print(f"Loss saved for epoch {epoch}: {self.loss.item():.6f}")

    def save_images(self, data, epoch):
        """Save original, reconstructed, and anomaly images in train_results/cae/result."""
        images = data['img'].cpu()  # Original images
        generated_imgs = self.generated_imgs.cpu()  # Reconstructed images
        
        # Set paths for saving results
        paths = os.path.join(self.opt.custom_save_dir, 'cae', 'result')
        utils.mkdirs(paths)

        # Save original images
        for i in range(images.size(0)):
            original_img = utils.convert2img(images[i])  # Convert to NumPy
            original_path = os.path.join(paths, f"epoch_{epoch}_original_{i}.png")
            Image.fromarray(original_img).save(original_path)

        # Save generated images
        for i in range(generated_imgs.size(0)):
            reconstructed_img = utils.convert2img(generated_imgs[i])  # Convert to NumPy
            reconstructed_path = os.path.join(paths, f"epoch_{epoch}_reconstructed_{i}.png")
            Image.fromarray(reconstructed_img).save(reconstructed_path)

        # Generate the anomaly image comparison
        anomaly_img = utils.compare_images(images, generated_imgs, threshold=self.opt.threshold)
        
        # Save the anomaly image
        result_path = os.path.join(paths, f"epoch_{epoch}_anomaly.png")
        Image.fromarray(anomaly_img).save(result_path)
        
        print(f"Sample images saved for epoch {epoch} at {result_path}")
