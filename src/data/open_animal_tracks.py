from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob

class OpenAnimalTracks(Dataset):

    def __init__(self, root='./data/open_animal_tracks', train=True, transform=None, download=False):

        self.transform = transform
        self.root = root
        self.train = train
        
        # Train veya test klasörüne göre yolu belirle
        data_folder = 'train' if train else 'test'
        
        # Google Drive'dan veri seti yolu
        self.data_path = os.path.join('/content/drive/MyDrive/Colab Notebooks/OpenAnimalTracks/OpenAnimalTracks', data_folder)
        
        # Dizinin var olup olmadığını kontrol et
        if not os.path.exists(self.data_path):
            raise RuntimeError(f'Dizin bulunamadı: {self.data_path}. Lütfen veri setinin doğru konumda olduğundan emin olun.')
        
        # Görüntüleri listele
        self.image_files = glob(os.path.join(self.data_path, '*.jpg'))
        
        if len(self.image_files) == 0:
            raise RuntimeError(f'Görüntü bulunamadı: {self.data_path}')
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, -1  # Unsupervised öğrenme için etiket -1