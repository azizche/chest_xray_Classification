"""
importing the data and then putting them in dataloaders
"""
from custom_dataset import XrayChestDataset
import torchvision
from torch.utils.data import DataLoader
def create_dataloaders(batch_size:int,transforms:torchvision.transforms.Compose,num_workers:int):
  """Creates training and testing DataLoaders.

  Downloads the data from huggingspace and turns
  it into PyTorch Dataset and then into PyTorch DataLoaders.

  Args:
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.

  """

  from datasets import load_dataset

  ds = load_dataset("keremberke/chest-xray-classification", name="full")
  train_data=XrayChestDataset(ds=ds['train'],transform=transforms,)

  test_data=XrayChestDataset(ds=ds['test'],transform=transforms,)


  train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
  test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
  class_names=['NORMAL','PNEUMONIA']
  return train_dataloader, test_dataloader, class_names

