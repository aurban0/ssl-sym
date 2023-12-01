import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, ToTensor
import pandas as pd
import random

# Create partially rotated datasets (MNISTRot60, MNISTMultiple etc)
def create_dataset(dataset_path, new_dataset_title, gaussian=False):
    if "90" in new_dataset_title:
        true_thetas_dict = {0: 60., 1: 60., 2: 60., 3: 60., 4: 60.,
                            5: 90., 6: 90., 7: 90., 8: 90., 9: 90.}
        std_devs_dict = {0: 30., 1: 30., 2: 30., 3: 30., 4: 30.,
                         5: 45., 6: 45., 7: 45., 8: 45., 9: 45.}
    elif "60" in new_dataset_title:
        true_thetas_dict = {0: 60., 1: 60., 2: 60., 3: 60., 4: 60.,
                            5: 60., 6: 60., 7: 60., 8: 60., 9: 60.}
        std_devs_dict = {0: 30., 1: 30., 2: 30., 3: 30., 4: 30.,
                         5: 30., 6: 30., 7: 30., 8: 30., 9: 30.}
    elif "multiple" in new_dataset_title:
        true_thetas_dict = {0: 0, 1: 18, 2: 36, 3: 54, 4: 72,
                            5: 90, 6: 108, 7: 126, 8: 144, 9: 162}
        std_devs_dict = {0: 0, 1: 9, 2: 18, 3: 27, 4: 36,
                         5: 45, 6: 54, 7: 63, 8: 72, 9: 81}

    # Load mnist dataset to transform it with partial rotations
    data = np.loadtxt(dataset_path)
    img_array_raw = data[:, :-1].reshape(len(data), 28, 28)
    labels_array_raw = data[:, -1]

    # Define transforms
    resize28 = Resize(28)
    toTensor = ToTensor()

    x_df = pd.DataFrame()
    rotations_dict = {i: [] for i in range(10)}  # to store rotations applied
    # Separate data by label
    data_by_label = {i: [] for i in range(10)}
    for i in range(len(data)):
        data_by_label[labels_array_raw[i]].append(img_array_raw[i])

    for label in range(10):
        if not gaussian:
            equiv_dict = true_thetas_dict
            theta = equiv_dict[label]
            imgs = data_by_label[label]
            rotations = np.random.uniform(-theta, theta, len(imgs))  # Sample rotations uniformly
        elif gaussian:
            equiv_dict = std_devs_dict
            std_dev = equiv_dict[label]
            imgs = data_by_label[label]
            rotations = np.random.normal(0, std_dev, len(imgs))  # Sample rotations through gaussian

        for img, rotation in zip(imgs, rotations):
            imgRot = Image.fromarray(img)  # PIL Image
            rotations_dict[label].append(rotation)  # store the rotation applied
            imgRot = toTensor(resize28((imgRot.rotate(rotation, Image.BILINEAR)))).reshape(1, 28 * 28)

            imgRotdf = pd.DataFrame(imgRot.numpy())
            imgRotdf["labels"] = [label]
            x_df = pd.concat([x_df, imgRotdf])
    print("Shape: ", x_df.shape)

    # Some visual check of the distribution of the rotation angles applied
    for label, rotations in rotations_dict.items():
        plt.hist(rotations, bins=30, density=True)
        plt.title(f"Rotation distribution for label {label}")
        plt.show()

    x_df.to_pickle("./"+new_dataset_title+".pkl")


def create_cyclic_rotated_dataset(dataset_path, new_dataset_title):
    # Define rotation angles for C2 and C4
    rotation_angles_dict = {
        'C0': [0],
        'C1': [0],
        'C2': [0, 180],
        'C3': [0, 120, -120],
        'C4': [0, 90, 180, 270],
        'C5': [0, 72, 144, -72, -144],
        'C6': [0, 60, 120, 180, -60, -120],
        'C8': [0, 45, 90, 135, 180, -45, -90, -135]
    }

    # Load mnist dataset
    data = np.loadtxt(dataset_path)
    img_array_raw = data[:, :-1].reshape(len(data), 28, 28)
    labels_array_raw = data[:, -1]

    # Define transforms
    resize28 = Resize(28)
    toTensor = ToTensor()

    x_df = pd.DataFrame()
    rotations_dict = {i: [] for i in range(10)}  # to store rotations applied
    data_by_label = {i: [] for i in range(10)}
    for i in range(len(data)):
        data_by_label[labels_array_raw[i]].append(img_array_raw[i])

    for label in range(10):
        imgs = data_by_label[label]
        if label in [0, 1, 2, 3, 4]:
            rotations = np.random.choice(rotation_angles_dict['C2'], len(imgs))
        else:
            rotations = np.random.choice(rotation_angles_dict['C4'], len(imgs))

        for img, rotation in zip(imgs, rotations):
            imgRot = Image.fromarray(img)  # PIL Image
            rotations_dict[label].append(rotation)  # store the rotation applied
            imgRot = toTensor(resize28((imgRot.rotate(rotation, Image.BILINEAR)))).reshape(1, 28 * 28)

            imgRotdf = pd.DataFrame(imgRot.numpy())
            imgRotdf["labels"] = [label]
            x_df = pd.concat([x_df, imgRotdf])

    print("Shape: ", x_df.shape)

    # Visual check of the rotation angles applied
    for label, rotations in rotations_dict.items():
        plt.hist(rotations, bins=30, density=True)
        plt.title(f"Cyclic Group Rotation Distribution for label {label}")
        plt.show()

    x_df.to_pickle("./"+new_dataset_title+".pkl")

def visualize_sample(pickle_path, sampleSize= 20, choice= []):
    df = pd.read_pickle(pickle_path)
    df = df.iloc[:,:-1]
    samples = np.random.randint(df.shape[0],size= sampleSize)
    if choice:
        samples = choice
        sampleSize = len(samples)
    fig, ax = plt.subplots(nrows=1, ncols=sampleSize, figsize=(18,9))

    for i,j in enumerate(samples):
        imgRot = df.iloc[j, :].values.reshape((28,28))
        ax[i].imshow(imgRot, cmap="gray")

    plt.show()

def main():
    # Create uniformly rotated dataset
    """
    partial_datasets = ["mnist_multiple", "mnist60", "mnist60_90"]

    for part_dataset in partial_datasets:
        print(f"Creating {part_dataset} datasets")

        # Train dataset
        train_dataset = part_dataset + "_train"
        create_dataset(dataset_path="./src/datasets/mnist_train.amat", new_dataset_title=train_dataset)
        save_path = "./"+train_dataset+".pkl"
        visualize_sample(save_path)

        # Test dataset
        test_dataset = part_dataset + "_test"
        create_dataset(dataset_path="./src/datasets/mnist_test.amat", new_dataset_title=test_dataset)
        save_path = "./" + test_dataset + ".pkl"
        visualize_sample(save_path)
    """

    # Create gaussian rotated datasets
    """partial_datasets_gaussian = ["mnist_multiple_gaussian"]

    for part_dataset in partial_datasets_gaussian:
        print(f"Creating {part_dataset} datasets")

        # Train dataset
        train_dataset = part_dataset + "_train"
        create_dataset(dataset_path="./src/datasets/mnist_train.amat", new_dataset_title=train_dataset,
                       gaussian=True)
        save_path = "./" + train_dataset + ".pkl"
        visualize_sample(save_path)

        # Test dataset
        test_dataset = part_dataset + "_test"
        create_dataset(dataset_path="./src/datasets/mnist_test.amat", new_dataset_title=test_dataset,
                       gaussian=True)
        save_path = "./" + test_dataset + ".pkl"
        visualize_sample(save_path)"""

    # Create Cn datasets
    """cyclic_datasets = ["mnist_c2c4"]

    for part_dataset in cyclic_datasets:
        print(f"Creating {part_dataset} datasets")

        # Train dataset
        train_dataset = part_dataset + "_train"
        create_cyclic_rotated_dataset(dataset_path="./src/datasets/mnist_train.amat", new_dataset_title=train_dataset)
        save_path = "./" + train_dataset + ".pkl"
        visualize_sample(save_path)

        # Test dataset
        test_dataset = part_dataset + "_test"
        create_cyclic_rotated_dataset(dataset_path="./src/datasets/mnist_test.amat", new_dataset_title=test_dataset)
        save_path = "./" + test_dataset + ".pkl"
        visualize_sample(save_path)"""

if __name__ == "__main__":
    main()
