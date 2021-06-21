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
  def __init__(self, root_dir, attributes_file,
               full_train_set = None,
               images_list = None,
               transform = None,
               target_transform = None):

    super(Market1501, self).__init__(root_dir, transform=transform,
                                      target_transform=target_transform)
    
    self.root_dir = root_dir #Path to the folder containing the images
    self.transform = transform
    self.target_transform = target_transform
    self.images_list = images_list

    #self.identities = get_ids_from_images(full_train_set)
    self.identities = get_ids_from_images(images_list)

    self.attr_df = pd.read_csv(attributes_file)
    self.convert_attributes_01()

    #self.classes = list(set(self.identities))
    #self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

  def convert_attributes_01(self):
    """This function converts the input of the csv (1 and 2) to binary values (0 and 1)"""
    for column in self.attr_df.columns:
      if(column!='age' and column!='id'):
        self.attr_df[column] =np.array((self.attr_df[column].astype('str').replace({'1': '0', '2': '1'})).astype("int64"))

  def __getitem__(self, idx: int):
    '''
    :param idx the integral index of the element to retrieve
    :return the element at index idx
    '''
    image_name = self.images_list[idx]
    X = image_loader(os.path.join(self.root_dir, image_name))

    identity = image_name.split("_")[0]
    #y = self.class_to_idx[identity]
    try:
      attr = self.attr_df[self.attr_df["id"] == int(identity)].values[0][1:]
    except:
      print("id to int: ",int(identity))
      print('identity: ',identity)
      print("dataframe lenght: ", len(self.attr_df))
      print(self.attr_df.head(20))
      print("results: ",self.attr_df[self.attr_df["id"] == int(identity)])

    if self.transform is not None:
        X = self.transform(X)

    if self.target_transform is not None:
        #y = self.target_transform(y)
        identity = self.target_transform(int(identity))
    return X, identity, attr

  def __len__(self):
    '''
    :return the number of elements that compose the dataset
    '''
    return len(self.images_list)

def image_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')