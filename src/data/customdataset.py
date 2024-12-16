import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root="/content/drive/MyDrive/Colab Notebooks/OpenAnimalTracks/OpenAnimalTracks",train=True, transform=None, download=False):
        """
        Custom dataset class for loading images.

        Args:
            root (str): Path to the root directory containing images.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.root = root
        self.transform = transform
        self.train = train
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        """Retrieve all image paths from the root directory."""
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]  # Supported formats
        image_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Retrieve an image and apply transformations."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure RGB format

        if self.transform:
            image = self.transform(image)

        return image
