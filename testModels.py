import numpy as np
import torch
import torchvision
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
import os
from glob import glob

import csv

from torchmetrics.regression import MeanAbsoluteError

data_path = "C:/Users/Ali/Desktop/UNI/AI/3UGERS/IKIT-3-ugers/face_age/"  # path til mappen, hvor dataen er placeret
class_labels = np.arange(1, 101)  # alle aldre, som modellen for fodret
num_classes = len(class_labels)
batch_size = 128
num_epochs = 100

weight_decay = 0.0
learning_rate = 0.0001

class_path = [
    "t" + f"{age:03}" for age in class_labels
]  # sætter "t" foran alderen, når modellen testes (t for test, ikke træning)


dsave = os.getenv("DSAVE", "MODELNOVARIABLESET")


loadPath = "C:/Users/Ali/Desktop/UNI/AI/3UGERS/IKIT-3-ugers/modelgroups/"

# når modellen trænes


### rent copy-paste fra Mikkels kode
# modificering og cropping af billederne
def image_processing(image):
    # konverterer billede til "farve", hvis det er sort-hvid
    if image.shape[0] == 1:  # hvis billedet kun har en farvekode
        image = image.repeat(3, 1, 1)  # gentages farvekoden 3 gange, som om det var RGB
    # konverter RGB-værdierne til tal mellem 0 og 1
    image = image.float() / 255
    # image = torchvision.transforms.functional.crop(image, 0, 0, 200, 200)   # cropper billedet til 200x200
    image = torchvision.transforms.functional.resize(
        image, [100, 100], antialias=True
    )  # sætter billedestørrelsen til 48x48 (resolution)
    return image


images = []
labels = []

# Add each image to list
for i, label in enumerate(class_labels):
    print(i)
    # Get all PNG and q JPG files in directory
    filenames = glob(
        os.path.join(data_path, class_path[i], "*.[PJp][NPn][Gg]")
    )  # tjekker for både .png og .jpg filer i mapperne

    for file in filenames:
        # Put image on list
        image = torchvision.io.read_image(file)
        image = image_processing(image)
        images.append(image)
        # Put label on list
        labels.append(class_labels[i])


# Put data into a tensor
images_tensor = torch.stack(images).float()
labels_tensor = torch.tensor(labels)
labels_tensor = (labels_tensor - class_labels.min()) / (
    class_labels.max() - class_labels.min()
)

# kører på GPU'en, hvis muligt
device = "cuda" if torch.cuda.is_available() else "cpu"


train_data = TensorDataset(
    images_tensor, labels_tensor
)  # sætter alle billeder sammen med deres korresponderende label
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)  # vælger data, størrelsen pr. batch og at dataen skal blandes
### slut med ren copy-paste

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


def unnormalize_tensor(tensor):
    return tensor * (class_labels[-1] - class_labels[0]) + class_labels[0]


MAE_ALL_LIST = []
graph_list = []

# Train

# count number of folders to run test on all models in the directory. Taken from stackoverflow at

numOfModels = len(
    [
        name
        for name in os.listdir(
            f'C:/Users/Ali/Desktop/UNI/AI/3UGERS/IKIT-3-ugers/modelgroups/{dsave.split("/")[0]}'
        )
    ]
)
for o in range(1, numOfModels + 1):

    graph_list = []
    print("at MODEL NUM ", o)

    for epoch in range(1, num_epochs + 1):

        load_from = loadPath + f"{dsave}{o}/EPOCH-{epoch}-net.pt"

        net.load_state_dict(torch.load(load_from, weights_only=False))

        epochOut = torch.empty(0)
        epochOut = epochOut.to(device)
        epochY = torch.empty(0)
        epochY = epochY.to(device)

        for x, y in train_loader:

            # Put data on GPU
            x = x.to(device)
            y = y.to(device)

            # Compute loss and take gradient step
            out = net(x)
            out = out.squeeze(-1)

            epochOut = torch.cat((epochOut, unnormalize_tensor(out)), 0)
            epochY = torch.cat((epochY, unnormalize_tensor(y)), 0)

        MAE = (sum(abs(epochOut - epochY)) / len(epochY)).item()

        graph_list.append(MAE)

    MAE_ALL_LIST.append(MAE)
    epochs = []
    training_mae = []

    # Read the existing CSV file and add the new column
    with open(f"./modelgroups/{dsave}{o}/mae_per_epoch.csv", "r") as file:

        reader = list(csv.reader(file))
        print(reader)

        for row in reader[1:]:

            epochs.append(int(row[0]))  # epoch numbers are in the first column
            training_mae.append(float(row[1]))  # training MAE is in the second column

    # Add the new column to each row
    for i, row in enumerate(reader):
        if i == 0:
            row.append("Testing MAE")  # Header for the new column
        else:
            if i - 1 < len(graph_list):  # Check if new_column_list has a value
                row.append(graph_list[i - 1])
            else:
                row.append("")  # Fill with blank if new_column_list is shorter

    # Write back the updated content to the same CSV
    with open(f"./modelgroups/{dsave}{o}/mae_per_epoch.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(reader)


with open(
    f"C:/Users/Ali/Desktop/UNI/AI/3UGERS/IKIT-3-ugers/modelgroups/{dsave.split("/")[0]}/allMAE.csv",
    "a",
    newline="",
) as file:
    writer = csv.writer(file)

    writer.writerow(["Model", "Testing MAE after 100 epochs"])

    for epoch, mae in enumerate(MAE_ALL_LIST, start=1):
        writer.writerow([epoch, mae])
