import utils
import torchvision.transforms as T

from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

DATASET_ID = utils.DATASET_ID

def get_dataloader(batch_size: int):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(256),
        T.ToTensor(),
    ])

    dataset = CelebA(root="./data", split="train", download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
