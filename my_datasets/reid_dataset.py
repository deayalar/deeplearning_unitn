import os
import torch
import logging
import pandas as pd
from numpy import asarray
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset

from utils.image_utils import get_ids_from_images

from PIL import Image

class Market1501(VisionDataset):
  '''
  Dataset class for the re-identification task
  '''
  def __init__(self, root_dir, attributes_file,
               full_train_set = None,
               images_list = None,
               transform = None,
               attr=None,
               target_transform = None):

    super(Market1501, self).__init__(root_dir, transform=transform,
                                      target_transform=target_transform)
    self.root_dir = root_dir #Path to the folder containing the images
    self.transform = transform
    self.target_transform = target_transform
    self.images_list = images_list
    self.attr = attr
    #self.identities = get_ids_from_images(full_train_set)
    self.identities = get_ids_from_images(images_list)

    self.attr_df = pd.read_csv(attributes_file)

    self.classes = list(set(self.identities))
    # self.classes = self.identities

    self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
    # idx = self.class_to_idx
    self.attr = self.getAttribute()

  def __getitem__(self, idx: int):
    '''
    :param idx the integral index of the element to retrieve
    :return the element at index idx
    '''
    image_name = self.images_list[idx]
    X = image_loader(os.path.join(self.root_dir, image_name))

    identity = image_name.split("_")[0]
    y = self.class_to_idx[identity]
    attr = self.attr_df[self.attr_df["id"] == int(identity)].values[0][1:]
    # attr = self.attr_df[self.attr_df["id"] == idx]

    if self.transform is not None:
        X = self.transform(X)

    if self.target_transform is not None:
        y = self.target_transform(y)

    # return X, attr, identity, image_name
    return X, attr

  def __len__(self):
    '''
    :return the number of elements that compose the dataset
    '''
    return len(self.images_list)

  def getAttribute(self):
    idx = self.class_to_idx
    list_attr_keys = list(idx.keys())
    list_attr_keys = list(map(int, list_attr_keys))
    
    attr_list_df = self.attr_df[self.attr_df['id'].isin(list_attr_keys)]
    
    return attr_list_df.to_numpy()

def image_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


