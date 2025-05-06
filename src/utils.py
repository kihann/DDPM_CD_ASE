import torch

def get_available_device():    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device

if __name__ == "__main__":
    pass
else:
    DEVICE = get_available_device()
    MODEL_ID = "google/ddpm-celebahq-256"
    DATASET_ID = "CelebA"