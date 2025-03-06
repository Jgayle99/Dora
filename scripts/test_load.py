import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel

try:
    config = AutoConfig.from_pretrained('facebook/dinov2-with-registers-large')
    dinov2_model = AutoModel.from_config(config)
    state_dict = load_file('dinov2-with-registers-large.safetensors', device='cuda')  # Use 'cuda'
    dinov2_model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully!")

except Exception as e:
    print(f"Error: {e}")