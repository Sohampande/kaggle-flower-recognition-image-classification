import os
import json
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms

from src.cnn_classifier.utils import logger
from src.cnn_classifier.components.dataset import CustomImageDataset

@dataclass
class DataTransformationArtifact : 
    train_loader : DataLoader
    val_loader : DataLoader
    test_loader : DataLoader
    class_names : list
    dataset_sizes: dict

class DataTransformation : 
    def __init__(self, config):
        self.config = config
    
    def get_transforms(self):
        train_transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.mean,
                std=self.config.std
            )
        ])

        eval_transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.mean,
                std=self.config.std
            )
        ])

        return train_transform, eval_transform
    
    def initiate_data_transformation(self):
        logger.info("Starting data transformation component")

        train_transform, eval_transform = self.get_transforms()
        root_dir = self.config.data_path

        # Base dataset only for editing file paths/classes/labels
        full_dataset = CustomImageDataset(root_dir=root_dir, transform=None)
        class_names = full_dataset.classes

        total_size = len(full_dataset)
        train_size = int( self.config.train_ratio * total_size)
        eval_size = int(self.config.val_ratio * total_size)
        val_size = int(self.config.val_ratio * total_size)
        test_size = total_size - train_size - val_size

        logger.info(f"Total dataset size : {total_size}")
        logger.info(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

        generator = torch.Generator().manual_seed(self.config.seed)
        
        # Splits the datasets into train, val, test subsets.
        train_subset, val_subset, test_subset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=generator
        )
        
        # Create separate datasets with transforms 
        train_dataset_full = CustomImageDataset(root_dir=root_dir, transform=train_transform)
        eval_dataset_full = CustomImageDataset(root_dir=root_dir, transform=eval_transform)
        # These variables contain the dataset with the transforms for train and test applied respt.

        # We finally create the desired datasets with the applied transforms + the respective indicies.
        train_dataset = Subset(train_dataset_full, train_subset.indices)
        val_dataset = Subset(eval_dataset_full, val_subset.indices)
        test_dataset = Subset(eval_dataset_full, test_subset.indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )

        os.makedirs(os.path.dirname(self.config.class_names_path), exist_ok=True)
        with open(self.config.class_names_path, "w") as f:
            json.dump(class_names, f, indent=4)

        dataset_sizes = {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset)
        }

        dataset_sizes_path = '/Users/Soham/Documents/Projects/kaggle-flower-recognition-image-classification/artifacts/data_transformation/dataset_sizes.json'
        with open(dataset_sizes_path, 'w') as f:
            json.dump(dataset_sizes, f, indent=4)

        logger.info("Data transformation completed successfully")

        # Note, we dont save the dataloader. But we create an abstract class and will be passing it
        #           onto the next stage.

        return train_loader, val_loader, test_loader