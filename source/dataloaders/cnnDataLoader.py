import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CnnDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform

        # Group annotations by image filename
        self.image_data = self.annotations.groupby(self.annotations.columns[0])

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image_name, group = list(self.image_data)[idx]
        img_path = os.path.join(self.image_dir, image_name)
        image = Image.open(img_path).convert("RGB")

        # Extract bounding boxes (all rows for this image)
        bboxes = group.iloc[:, 4:].values.astype('float')

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'bboxes': torch.tensor(bboxes, dtype=torch.float32)}

# Custom collate function to handle variable-length bounding boxes
def collate_fn(batch):
    images = torch.stack([transforms.ToTensor()(item['image']) for item in batch])  # Convert images to tensors and stack
    bboxes = [item['bboxes'] for item in batch]  # List of bounding boxes per image
    return {'images': images, 'bboxes': bboxes}

def get_dataloaders(data_dir, batch_size=8, transform=None):
    train_dataset = CnnDataset(
        image_dir=os.path.join(data_dir, 'images/train'),
        annotation_file=os.path.join(data_dir, 'annotations/train/_annotations.csv'),
        transform=transform
    )
    valid_dataset = CnnDataset(
        image_dir=os.path.join(data_dir, 'images/valid'),
        annotation_file=os.path.join(data_dir, 'annotations/valid/_annotations.csv'),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    return train_loader, valid_loader

# Example usage
if __name__ == "__main__":
    data_dir = 'C:\\Users\\babis\\Documents\\GitHub\\ThesisLts\\data\\cnn_dataset'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_loader, valid_loader = get_dataloaders(data_dir, transform=transform)

    for batch in train_loader:
        images, bboxes = batch['images'], batch['bboxes']
        print(f"Images shape: {images.shape}")  # Should be [batch_size, 3, 224, 224]
        print(f"Bounding Boxes: {bboxes}")  # List of tensors with different sizes
        break
