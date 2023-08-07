"""
defines the dataset class of the problem
"""

from torch.utils.data import Dataset
class XrayChestDataset(Dataset):
  def __init__(self,ds,transform=None):
    self.ds=ds
    self.transform = transform
  def __len__(self)->int:
    return self.ds.num_rows
  def __getitem__(self, index):
    img= self.ds[index]['image']
    class_indx= self.ds[index]['labels']
    if self.transform:
      img=self.transform(img)
    return img, class_indx



