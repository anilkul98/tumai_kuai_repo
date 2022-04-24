from torch.utils.data import Dataset
from PIL import Image

class SatImageDataset(Dataset):
    def __init__(self, images, transform=None):
        self.imgs = images
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = Image.fromarray(self.imgs[idx])
        if self.transform:
            image = self.transform(image)
        return image