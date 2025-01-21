# %% Import libraries
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import csv


# %% Settings
# data_path = "/Users/AliA/DTU/IIS/projekt/IKIT-3-ugers/face_age"
data_path = "C:/Users/Ali/Desktop/UNI/AI/3UGERS/IKIT-3-ugers/face_age"

class_labels = np.arange(1, 101)

# Tjek om GPU er tilgænglig
print(torch.cuda.is_available())

# local parameters. Same for all models
batch_size = 128
num_epochs = 100
learning_rate = 0.001
weight_decay = 0.0

# group-specific parameters. Set from "run_tests.py" script.
dbCut = int(os.getenv("DBCUT", "25")) / 100
dSave = os.getenv("DSAVE", "test/MODEL-NUMBER-1")


class_path = [f"{age:03}" for age in class_labels]


# %% Data preprocessing function
def preprocess(image):
    # Convert to color if black-and-white
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    # Convert to floating point numbers between 0 and 1
    image = image.float() / 255

    # Resize to 100x100
    image = torchvision.transforms.functional.resize(image, [100, 100], antialias=True)
    return image


# %% Load data
# Empty lists to store images and labels
images = []
labels = []


# used to shuffle labels and images array in unison. Taken from stack overflow @https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison#:~:text=To%20shuffle%20both%20arrays%20simultaneously,create%20c%20%2C%20a2%20and%20b2%20.
def shuffle_arrays(arrays, set_seed=-1):
    """Shuffles arrays in-place, in the same order, along axis=0

    Parameters:
    -----------
    arrays : List of NumPy arrays.
    set_seed : Seed value if int >= 0, else seed is random.
    """
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2 ** (32 - 1) - 1) if set_seed < 0 else set_seed

    for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)


print("TRAINING LOADING")

for i, label in enumerate(class_labels):
    # Get all JPEG files in directory
    print(i)
    filenames = glob(os.path.join(data_path, class_path[i], "*.[PJp][NPn][Gg]"))

    for file in filenames:
        # Put image on list
        image = torchvision.io.read_image(file)

        image = preprocess(image)

        images.append(image)

        # Put label on list
        labels.append(class_labels[i])

print("images len ", len(images))
shuffle_arrays([images, labels])

images = images[: int(len(images) * dbCut)]
labels = labels[: len(images)]
# Put data into a tensor
images_tensor = torch.stack(images).float()

print("****SIZE***")
print(images_tensor.shape)
labels_tensor = torch.tensor(labels)

# normalize labels
labels_tensor = (labels_tensor - class_labels.min()) / (
    class_labels.max() - class_labels.min()
)


# %% Device
# Run on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% Create dataloader
train_data = TensorDataset(images_tensor, labels_tensor)


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# %% Neural network

net = torch.nn.Sequential(
    torch.nn.Conv2d(3, 32, kernel_size=5),  # b x 32 x 124 x 124
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),  # b x 32 x 62 x 62
    torch.nn.Conv2d(32, 64, kernel_size=3),  # b x 64 x 60 x 60
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),  # b x 64 x 30 x 30
    torch.nn.Conv2d(64, 128, kernel_size=3),  # b x 128 x 28 x 28
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),  # b x 128 x 14 x 14
    torch.nn.Conv2d(128, 256, kernel_size=3),  # b x 256 x 12 x 12
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),  # b x 256 x 6 x 6
    torch.nn.Flatten(),  # 256 x 6 x 6 = 9.216
    torch.nn.Linear(4096, 1),  # 1
    torch.nn.Sigmoid(),
).to(device)


# %% Load trained network from file


# %% Loss and optimizer
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    net.parameters(), lr=learning_rate, weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# %% Train

step = 0


def unnormalize_tensor(tensor):
    return tensor * (class_labels[-1] - class_labels[0]) + class_labels[0]


training_acc_list = []
print(device)


for epoch in range(1, num_epochs + 1):

    first = True

    print("EPOCH IS ", epoch)

    epochOut = torch.empty(0)
    epochOut = epochOut.to(device)
    epochY = torch.empty(0)
    epochY = epochY.to(device)

    for x, y in train_loader:
        step += 1
        # Put data on GPU
        x = x.to(device)
        y = y.to(device)

        out = net(x)
        out = out.squeeze(-1)

        epochOut = torch.cat((epochOut, unnormalize_tensor(out)), 0)
        epochY = torch.cat((epochY, unnormalize_tensor(y)), 0)

        # Compute loss and take gradient step

        loss = loss_function(out, y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if first:
            print("max ", max(y))
            for i in range(5):
                print(
                    "PRED ",
                    out.data[i].item() * (class_labels[-1] - class_labels[0])
                    + class_labels[0],
                    "-TRUE ",
                    y.data[i].item() * (class_labels[-1] - class_labels[0])
                    + class_labels[0],
                )
            print("loss is ", loss.item())
            first = False

    print("learning rate is ", optimizer.param_groups[0]["lr"])
    scheduler.step()
    MAE = (sum(abs(epochOut - epochY)) / len(epochY)).item()

    # Print MAE for epoch
    print("MAE FOR EPOCH IS ", MAE)
    training_acc_list.append(MAE)

    if not dSave == "":
        if not os.path.exists(f"modelgroups/{dSave}"):
            os.makedirs(f"modelgroups/{dSave}")
        torch.save(net.state_dict(), f"modelgroups/{dSave}/EPOCH-{epoch}-net.pt")


if not dSave == "":
    with open(f"./modelgroups/{dSave}/mae_per_epoch.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Training MAE"])
        # Do not write the header row if appending
        for epoch, mae in enumerate(training_acc_list, start=1):
            writer.writerow([epoch, mae])

    plt.ioff()
    plt.show()
