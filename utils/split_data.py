import os
import math
from sklearn.model_selection import train_test_split

def get_ids_from_images(images_list):
    return [name.split("_")[0] for name in images_list]

class ValidationSplitter:
    """ 
    This class handles the strategy to create a validation set from the 
    training directory

    Parameters
    ----------
    train_root : Directory that contains the training examples
    """
    def __init__(self, train_root: str): 
        if not os.path.exists(train_root):
            raise FileNotFoundError("Can't find directory: " + train_root)
        self.train_root = train_root
    
    def split(self, train_size:float, split_identities: bool=False,random_seed:int=None):
        """
        Returns a tuple containing two list of images, one for training set
        and other for validation set

        Parameters
        ----------

        train_size : Proportion of the data included in training set e.g: 0.7

        split_identities: Set this flag to False in order to keep the all the 
        identities in the same dataset either in training or validation. 
        Otherwise, it will perform a raw split in the images regardless the identity

        random_seed: Seed for reproducibility

        """
        images_list = os.listdir(self.train_root)
        all_ids = get_ids_from_images(images_list)

        if not split_identities:
            unique_ids = list(set(all_ids)) #unique ids
            ids_train_set, ids_val_set = train_test_split(unique_ids,
                                                  train_size=train_size,
                                                  random_state=random_seed)

            train_set = [im for im in images_list for i in ids_train_set if im.startswith(i)]
            val_set = [im for im in images_list for i in ids_val_set if im.startswith(i)]
        
        else:
            train_set, val_set = train_test_split(images_list,
                                                  train_size=train_size,
                                                  random_state=random_seed)
        
        print(f"Identities in train set size: {len(set(get_ids_from_images(train_set)))}")
        print(f"Identities in val set size: {len(set(get_ids_from_images(val_set)))}")
        print(f"Train set size: {len(train_set)}")
        print(f"Validation set size: {len(val_set)}")

        return train_set, val_set
    
    #TODO: Include some statistics of the identities in the datasets, eg. distribution of identities. see split attempt

class TrainingSplitter():
    
    @staticmethod
    def split(full_training_data, train_size: float=0.8, random_seed=None):
        '''
        Stratified split of the training set to get the internal validation set
        '''
        identities  = get_ids_from_images(full_training_data)
        train_set, val_set = train_test_split(full_training_data,
                                                  train_size=train_size,
                                                  stratify=identities,
                                                  random_state=random_seed)

        print(f"Final Train set size: {len(train_set)}")
        print(f"Training Validation set size: {len(val_set)}")
        return train_set, val_set