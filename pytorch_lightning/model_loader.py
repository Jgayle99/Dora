import torch
from craftsman.models.autoencoders.michelangelo_autoencoder import MichelangeloAutoencoder
from omegaconf import OmegaConf

def load_michelangelo_vae_model(pretrained_model_path, config_path):
    """
    Loads the MichelangeloAutoencoder model from a checkpoint using a YAML config.

    Args:
        pretrained_model_path (str): Path to the pre-trained VAE model checkpoint.
        config_path (str): Path to the YAML configuration file.

    Returns:
        MichelangeloAutoencoder: Loaded MichelangeloAutoencoder model.
    """

    # --- 1. Load Config from YAML ---
    full_cfg = OmegaConf.load(config_path)
    vae_cfg = full_cfg.system.shape_model

    # --- 2. Instantiate VAE Model using Config ---
    autoencoder = MichelangeloAutoencoder(vae_cfg)

    # --- 3. Load the pre-trained weights
    checkpoint = torch.load(pretrained_model_path, map_location='cpu')
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    autoencoder.load_state_dict(checkpoint, strict=False)  # Load with strict=False
    autoencoder.eval() # Set to evaluation mode

    return autoencoder

if __name__ == "__main__":
    pretrained_model = "D://dev/Dora-Release/Dora-VAE-1.1/dora_vae_1_1.ckpt" # Replace with your actual path
    config_yaml = "D:/dev/Dora-Release/Dora/pytorch_lightning/configs/shape-autoencoder/Dora-VAE-test.yaml" # Replace with your actual path

    model = load_michelangelo_vae_model(pretrained_model, config_yaml)
    print("MichelangeloAutoencoder model loaded successfully from model_loader.py!")