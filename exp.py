%reload_ext autoreload
%autoreload 2

import torch.optim as optim

from my_datasets.reid_dataset import ReIdDataset

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from utils.split_data import ValidationSplitter
from utils.image_utils import imshow

from models.reid_model import ReIdModel

TRAIN_ROOT = "/media/deayalar/Data/Documents/Unitn/Deep Learning/Assignment/dataset/train"

splitter = ValidationSplitter(train_root=TRAIN_ROOT)
train_set, val_set = splitter.split(train_size=0.75, random_seed=42)

#Create pytorch Datasets

composed = transforms.Compose([transforms.ToTensor()])

train_reid_dataset = ReIdDataset(root_dir=TRAIN_ROOT, 
                                 images_list=train_set,
                                 transform=composed)

length = len(train_reid_dataset)
print(f"Train Dataset length: {length}")
train_loader = DataLoader(train_reid_dataset, batch_size=128, shuffle=True, num_workers=4)

val_reid_dataset = ReIdDataset(root_dir=TRAIN_ROOT, 
                               images_list=val_set,
                               transform=composed)

length = len(val_reid_dataset)
print(f"Validation Dataset length: {length}")


# get random training images from dataloader
dataiter = iter(train_loader)
images, labels = dataiter.next()
# show some images
imshow(torchvision.utils.make_grid(images[0:7]))
labels[0:7]

reid_model = ReIdModel()

criterion = reid_model.get_loss_function()
optimizer = optim.SGD(reid_model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = reid_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')