# src/models/model.py
import torch
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
def get_mbpp_model(model_dir: str, device: torch.device):
    """
    Loads a code generation model from a local directory containing all model weight files,
    and moves it to the provided device.
    Args:
        model_dir (str): Path to the directory (e.g., "model_weights") containing the model files.
        device (torch.device): The device on which to load the model. 
    Returns:
        model: The loaded model in evaluation mode, on the specified device.
    """
    #Loads the model from the specified directory.
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.to(device)  # Ensure the model is moved to the desired device.
    model.eval()
    return model
