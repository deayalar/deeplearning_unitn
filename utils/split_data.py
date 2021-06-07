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
    def __init__(self, train_root: str, test_root: str, queries_root: str): 
        if not os.path.exists(train_root):
            raise FileNotFoundError(f"Can't find directory {train_root}")
        self.train_root = train_root
        self.test_root = test_root
        self.queries_root = queries_root
    
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
        val_set, val_queries = self.__extract_queries(val_set, keep_proportion=True)

        print(f"Identities in train set: {len(set(get_ids_from_images(train_set)))}")
        print(f"Identities in validation set: {len(set(get_ids_from_images(val_set)))}")
        print(f"Train set size: {len(train_set)}")
        print(f"Validation set size: {len(val_set)}")
        print(f"Number of validation queries: {len(val_queries)}")

        return train_set, val_set, val_queries
    
    def __extract_queries(self, full_validation_set: list, keep_proportion=False, queries_size=.15):
        """
        Extract queries from validation set in the same proportion than test/queries
        TODO: Maybe we can include some junk images in this validation set to see how it performs in the estimation of test
        """
        if keep_proportion:
            if not self.test_root or not self.queries_root:
                raise RuntimeError("Set the test and queries directory to keep the same proportion for queries extraction")

            test_len = len(os.listdir(self.test_root))
            queries_len = len(os.listdir(self.queries_root))
            #This is to make an stratified split since we need to ensure results for the queries
            identities  = get_ids_from_images(full_validation_set)  

            queries_size = truncate((queries_len / test_len) , 2)
        print(f"Extract queries proportion: {queries_size}")
        validation_set, val_queries = train_test_split(full_validation_set,
                                            train_size=1 - queries_size,
                                            stratify=identities,
                                            random_state=42)
        return validation_set, val_queries
    
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
    
import math

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    https://kodify.net/python/math/truncate-decimals/
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor