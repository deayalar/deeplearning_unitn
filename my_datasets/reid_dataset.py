import os
import torch
from numpy import asarray
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset

from utils.image_utils import get_ids_from_images

from PIL import Image

class Market1501(VisionDataset):
  '''
  Dataset class for the re-identification task
  '''
  def __init__(self, root_dir,
               images_list = None,
               transform = None,
               target_transform = None):

    super(Market1501, self).__init__(root_dir, transform=transform,
                                      target_transform=target_transform)
    
    self.root_dir = root_dir #Path to the folder containing the images
    self.transform = transform
    self.target_transform = target_transform

    full_images_list = os.listdir(self.root_dir)

    if not images_list:
      print(f"Loading images from {self.root_dir}")
      self.images_list = full_images_list
    else:
      self.images_list = images_list

    self.identities = get_ids_from_images(full_images_list)

    self.classes = list(set(self.identities))
    self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

  def __getitem__(self, idx: int):
    '''
    :param idx the integral index of the element to retrieve
    :return the element at index idx
    '''
    image_name = self.images_list[idx]
    X = image_loader(os.path.join(self.root_dir, image_name))

    identity = image_name.split("_")[0]
    y = self.class_to_idx[identity]

    if self.transform is not None:
        X = self.transform(X)

    if self.target_transform is not None:
        y = self.target_transform(y)

    return X, y

  def __len__(self):
    '''
    :return the number of elements that compose the dataset
    '''
    return len(self.images_list)

def image_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')