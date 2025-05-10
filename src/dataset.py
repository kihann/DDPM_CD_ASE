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
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = CelebA(root="./data", split="train", download=False, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
