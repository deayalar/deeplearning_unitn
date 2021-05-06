import os
import torch
from numpy import asarray
from torch.utils.data import Dataset

from PIL import Image

class ReIdDataset(Dataset):
  '''
  Dataset class for the re-identification task
  '''
  def __init__(self, root_dir,
               images_list = None,
               transform = None,
               target_transform = None):

    super(ReIdDataset, self).__init__()
    self.root_dir = root_dir #Path to the folder containing the images
    self.transform = transform
    self.target_transform = target_transform
    if not images_list:
      print(f"Loading images from {self.root_dir}")
      self.images_list = os.listdir(self.root_dir)
    else:
      self.images_list = images_list


  def __getitem__(self, idx):
    '''
    :param idx the integral index of the element to retrieve
    :return the element at index idx
    '''
    image_name = self.images_list[idx]
    im = Image.open(os.path.join(self.root_dir, image_name))
    X = asarray(im) #Images as RGB
    y = image_name.split("_")[0] #Get image label (identity)
    y = int(y)
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
