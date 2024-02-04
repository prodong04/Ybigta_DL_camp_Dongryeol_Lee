import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from PIL import Image
from torchvision import transforms as T

valid_images = [".jpg", ".gif", ".png", ".tga", ".jpeg", ".PNG", ".JPG", ".JPEG"]
with open("./class_info.json", 'r') as f:
    class2id = json.load(f)

class FoodDataset(Dataset):
    def __init__(
        self, 
        root: str, 
        split: str, 
        transforms=None
    ):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.totensor = T.ToTensor()
        self.class2id = class2id
        self.data = self.prepare_dataset()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        ##################### fill here ####################
        #   TODO: __getitem__을 정의해주세요
        img_path, label = self.data[index]
        
        # Load image
        img = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transforms:
            img = self.transforms(img)

        # Convert to PyTorch tensor
        img = self.totensor(img)

        return img, label

        ####################################################
            
    def prepare_dataset(self):
        split_base = os.path.join(self.root, self.split)
        data = []
        
        for label in os.listdir(split_base):
            if label not in self.class2id:
                continue
            
            for image_name in os.listdir(os.path.join(split_base, label)):
                if os.path.splitext(image_name)[1] not in valid_images:
                    continue
                data.append((os.path.join(split_base, label, image_name), self.class2id[label]))
        
        return data