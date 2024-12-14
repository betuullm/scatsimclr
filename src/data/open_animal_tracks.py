# src/data/open_animal_tracks.py

from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob

class OpenAnimalTracks(Dataset):
    """
    OpenAnimalTracks veri setini yüklemek için özel Dataset sınıfı.
    Bu sınıf, belirtilen dizindeki hayvan görüntülerini yükler.
    """
    
    def __init__(self, config, train=True, transform=None):
        """
        Parametreler:
            config (dict): Config dosyası
            train (bool): True ise eğitim seti, False ise test seti yüklenir
            transform: Görüntülere uygulanacak dönüşümler
        """
        self.transform = transform
        self.root = config['dataset']['data_root']
        self.train = train
        
        # Train veya test klasörüne göre yolu belirle
        data_folder = config['dataset']['train_dir'] if train else config['dataset']['test_dir']
        self.data_path = os.path.join(self.root, data_folder)
        
        # Görüntüleri listele
        self.image_files = glob(os.path.join(self.data_path, '*.jpg'))
        
        if len(self.image_files) == 0:
            raise RuntimeError(f'Görüntü bulunamadı: {self.data_path}')
    
    def __len__(self):
        """Veri setindeki toplam görüntü sayısını döndürür"""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Belirtilen indeksteki görüntüyü yükler ve döndürür
        
        Parametreler:
            idx (int): Görüntü indeksi
            
        Dönüş:
            tuple: (görüntü, etiket) çifti. Etiket her zaman -1'dir (denetimsiz öğrenme için)
        """
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, -1  # Unsupervised öğrenme için etiket -1