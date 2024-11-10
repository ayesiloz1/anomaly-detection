import torch


class CAEModel(torch.nn.Module):
    def __init__(self, encoder, decoder, latent_dim, img_size):
        super(CAEModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.img_size = img_size

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def load_cae_model(encoder_path, decoder_path, latent_dim, img_size):
    # Import the networks module inside this function to avoid circular import issues
    from models import networks
    
    encoder = networks.Encoder(latent_dim, img_size)
    decoder = networks.Decoder(latent_dim, img_size)
    cae_model = CAEModel(encoder, decoder, latent_dim, img_size)
    return cae_model