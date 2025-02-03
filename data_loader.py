from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import torch
import os
import random


class MRIDataset(data.Dataset):
    """Dataset class for the MRI dataset."""

    def __init__(self, image_dir, transform, mode):
        """Initialize and preprocess the MRI dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the MRI dataset."""
        all_images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.image_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for filename in os.listdir(class_dir):
                all_images.append((os.path.join(class_dir, filename), self.classes.index(class_name)))

        random.seed(1234)
        random.shuffle(all_images)

        # Split dataset into training and testing sets
        split_index = int(0.8 * len(all_images))  # 80% training, 20% testing
        self.train_dataset = all_images[:split_index]  # Training set
        self.test_dataset = all_images[split_index:]   # Testing set

        print(f"Total images: {len(all_images)}")
        print(f"Training images: {len(self.train_dataset)}")
        print(f"Testing images: {len(self.test_dataset)}")
        print('Finished preprocessing the MRI dataset...')

    def __getitem__(self, index):
        import numpy as np
        """Return one image and its corresponding label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filepath, label = dataset[index]
        image = Image.open(filepath).convert('L')  # Convert grayscale to RGB
        return self.transform(image), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path=None, selected_attrs=None, crop_size=178, image_size=128, 
               batch_size=16, dataset='MRIDataset', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    # if mode == 'train':
    #     transform.append(T.RandomHorizontalFlip())
    transform.append(T.Grayscale(num_output_channels=1)),
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5), std=(0.5)))
    transform = T.Compose(transform)

    if dataset == 'MRIDataset':
        dataset = MRIDataset(image_dir, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader