import os
import numpy as np

def create_metadata():
    # Root folder containing the dataset
    root_folder = "outputs"

    # Initialize lists to store metadata
    image_paths = []
    class_labels = []
    subclass_labels = []

    # Traverse the folder structure
    for class_folder in os.listdir(root_folder):
        class_path = os.path.join(root_folder, class_folder)
        if not os.path.isdir(class_path):
            continue

        for video_folder in os.listdir(class_path):
            video_path = os.path.join(class_path, video_folder)
            if not os.path.isdir(video_path):
                continue

            for subclass_folder in os.listdir(video_path):
                subclass_path = os.path.join(video_path, subclass_folder)
                if not os.path.isdir(subclass_path):
                    continue

                for image_file in os.listdir(subclass_path):
                    if image_file.endswith(".png"):
                        image_id = os.path.splitext(image_file)[0]
                        image_path = os.path.join(subclass_path, image_file)

                        # Append the metadata to the lists
                        image_paths.append(image_path)
                        class_labels.append(class_folder)
                        subclass_labels.append(subclass_folder)

    # Convert lists to NumPy arrays
    image_paths = np.array(image_paths)
    labels = np.array(list(zip(class_labels,subclass_labels)))
    return image_paths,labels

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from imageio import imread
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from imageio import imread

class CustomDataset(Dataset):
    def __init__(self):
        self.image_paths, self.labels = create_metadata()
        self.unique_classes = np.unique(self.labels[:, 0])
        self.available_indices = self.get_available_indices()

    def get_available_indices(self):
        available_indices = []
        for idx, subclass_label in enumerate(self.labels[:, 1]):
            if subclass_label != "other":
                available_indices.append(idx)
        return available_indices

    def __len__(self):
        return len(self.available_indices)

    def __getitem__(self, idx):
        class_label = self.labels[self.available_indices[idx], 0]
        subclass_label = self.labels[self.available_indices[idx], 1]

        image_path = self.image_paths[self.available_indices[idx]]

        if np.random.rand() < 0.5:
            # Select a random image from the same class
            same_class_indices = np.where(self.labels[:, 0] == class_label)[0]
            random_idx = np.random.choice(same_class_indices)
        else:
            # Select a random image from a different class
            different_class_indices = np.where(self.labels[:, 0] != class_label)[0]
            random_idx = np.random.choice(different_class_indices)

        random_image_path = self.image_paths[random_idx]

        # Determine if the pair is from the same subclass
        same_subclass = subclass_label == self.labels[random_idx, 1]

        return image_path, random_image_path, same_subclass


if __name__ == '__main__':
    dataset = CustomDataset()
    res = [d for d in dataset]
    np.save("reid/data/fremont.npy", res)

