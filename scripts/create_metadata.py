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
# class_labels = np.array(class_labels)
# subclass_labels = np.array(subclass_labels)
# import pdb; pdb.set_trace()
# lis

# # Save metadata to a NumPy file
# np.savez("metadata.npz", image_paths=image_paths, class_labels=class_labels, subclass_labels=subclass_labels)

