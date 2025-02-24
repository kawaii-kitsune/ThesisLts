import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DataLoaderFactory:
    """
    A factory class to create data loaders for training, validation, and testing datasets.
    Attributes:
        data_dir (str): The directory where the dataset is stored. It should contain subdirectories 'train', 'valid', and 'test'.
        batch_size (int): The number of samples per batch to load. Default is 32.
        num_workers (int): The number of subprocesses to use for data loading. Default is 4.
        data_transforms (dict): A dictionary containing the data transformations to be applied to the 'train', 'valid', and 'test' datasets.
    Methods:
        get_data_loaders():
            Creates and returns data loaders for the 'train', 'valid', and 'test' datasets.
            Raises:
                FileNotFoundError: If any of the 'train', 'valid', or 'test' directories do not exist in the data_dir.
                ValueError: If any of the 'train', 'valid', or 'test' directories do not contain subdirectories.
    """
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def get_data_loaders(self):
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(self.data_dir, split)
            if not os.path.exists(split_dir):
                raise FileNotFoundError(f"Directory {split_dir} does not exist")
            if not any(os.path.isdir(os.path.join(split_dir, subdir)) for subdir in os.listdir(split_dir)):
                raise ValueError(f"Directory {split_dir} does not contain any subdirectories")

        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x])
                          for x in ['train', 'valid', 'test']}
        
        data_loaders = {x: DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
                        for x in ['train', 'valid', 'test']}
        
        return data_loaders

if __name__ == '__main__':
    data_dir = 'C:\\Users\\babis\\Documents\\GitHub\\ThesisLts\\data\\vit_dataset\\images'
    factory = DataLoaderFactory(data_dir)
    data_loaders = factory.get_data_loaders()
    
    for split in ['train', 'valid', 'test']:
        print(f"Classes in {split} dataset: {data_loaders[split].dataset.classes}")