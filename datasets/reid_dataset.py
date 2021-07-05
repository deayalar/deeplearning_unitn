import os
import torch
import logging
import pandas as pd
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset

from utils.image_utils import get_ids_from_images

from PIL import Image

class Market1501(VisionDataset):
  '''
  Dataset class for the re-identification task
  '''
  def __init__(self, root_dir,
               test_dataset = False,
               attributes_file=None,
               images_list = None,
               transform = None,
               target_transform = None):

    super(Market1501, self).__init__(root_dir, transform=transform,
                                      target_transform=target_transform)
    
    self.root_dir = root_dir #Path to the folder containing the images
    self.transform = transform
    self.target_transform = target_transform
    self.images_list = images_list
    self.test_dataset = test_dataset

    if not test_dataset:
      #self.identities = get_ids_from_images(full_train_set)
      self.identities = get_ids_from_images(images_list)

      self.attr_df = pd.read_csv(attributes_file)
      self.convert_attributes()

      self.unique_identities = list(set(self.identities))
      self.class_to_idx = {_class: i for i, _class in enumerate(self.unique_identities)}

  def convert_attributes(self):
    """This function converts the input of the csv to the corresponding categorical avlues"""
    for column in self.attr_df.columns:
      if(column!='id'):
        self.attr_df[column] =np.array((self.attr_df[column].astype('str').replace({'1': '0', '2': '1', '3': '2', '4': '3'})).astype("int64"))

  def __getitem__(self, idx: int):
    '''
    :param idx the integral index of the element to retrieve
    :return the element at index idx
    '''
    image_name = self.images_list[idx]
    X = image_loader(os.path.join(self.root_dir, image_name))
    y = torch.empty(1, dtype=torch.bool)
    attr = torch.empty(1, dtype=torch.bool)
    if not self.test_dataset:
      identity = image_name.split("_")[0]
      
      y = self.class_to_idx[identity]
      attr = self.attr_df[self.attr_df["id"] == int(identity)].values[0][1:]
      
      if self.target_transform is not None:
        y = self.target_transform(y)

    if self.transform is not None:
        X = self.transform(X)

    return X, y, attr


  def __len__(self):
    '''
    :return the number of elements that compose the dataset
    '''
    return len(self.images_list)

def image_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
