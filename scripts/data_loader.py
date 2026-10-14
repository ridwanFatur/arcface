from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
from sklearn.model_selection import train_test_split
import random
import pickle
from io import BytesIO

def preprocess_input(image):
    image /= 255.0 
    image -= 0.5
    image /= 0.5
    return image

def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def get_splitted_df(csv_path):
    df = pd.read_csv(csv_path)
    df_train, df_val = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    # common_labels = set(df_train["label"]) & set(df_val["label"])
    # df_train = df_train[df_train["label"].isin(common_labels)].reset_index(drop=True)
    # df_val = df_val[df_val["label"].isin(common_labels)].reset_index(drop=True)

    return df_train, df_val

class LabeledFaceDataset(Dataset):
    def __init__(self, root_path, df, split="train", width=112, height=112):
        self.df = df
        self.split = split
        self.width = width
        self.height = height 
        self.root_path = root_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        root_path = Path(self.root_path)
        
        image_path = root_path / row["path"].lstrip("/")
        image = Image.open(image_path).convert("RGB")
        label = int(row["label"])
        
        if self.split == "train" and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
        image = resize_image(image, [self.width, self.height], letterbox_image = True)
        arr_image = np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1))
        
        return image, arr_image, label
    
def labeled_face_dataset_collate(batch):
    images  = []
    targets = []
    for _, image, y in batch:
        images.append(image)
        targets.append(y)
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    targets = torch.from_numpy(np.array(targets)).long()
    return images, targets    
    
class PairDataset(Dataset):
    def __init__(self, root_path, dataset_path, images_are_bytes = False, width=112, height=112):
        dataset_path = root_path + "/" + dataset_path
        with open(dataset_path, "rb") as f:
            data = pickle.load(f, encoding="bytes")
            self.images = data[0]
            self.labels = data[1]

        self.images_are_bytes = images_are_bytes
        self.width = width
        self.height = height 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image1_data = self.images[index * 2]
        image2_data = self.images[index * 2 + 1]

        if not self.images_are_bytes:
            image1_data = image1_data.tobytes()
            image2_data = image2_data.tobytes()

        image1 = Image.open(BytesIO(image1_data)).convert("RGB")
        image2 = Image.open(BytesIO(image2_data)).convert("RGB")
        
        image1 = resize_image(image1, [self.width, self.height], letterbox_image = True)
        image2 = resize_image(image2, [self.width, self.height], letterbox_image = True)
        
        arr_image1 = np.transpose(preprocess_input(np.array(image1, np.float32)),[2, 0, 1])
        arr_image2 = np.transpose(preprocess_input(np.array(image2, np.float32)),[2, 0, 1])

        issame = bool(self.labels[index])
        
        return image1, image2, arr_image1, arr_image2, issame
    
def pair_dataset_collate(batch):
    images_1 = []
    images_2 = []
    targets = []

    for _, _, arr_image1, arr_image2, issame in batch:
        images_1.append(arr_image1)
        images_2.append(arr_image2)
        targets.append(int(issame))

    images_1 = torch.from_numpy(np.array(images_1)).float()
    images_2 = torch.from_numpy(np.array(images_2)).float()
    targets = torch.tensor(targets, dtype=torch.long)

    return images_1, images_2, targets

def reverse_preprocess_input(image):
    image = (image * 0.5 + 0.5) * 255.0
    image = image.clamp(0, 255).byte()
    image = image.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(image)