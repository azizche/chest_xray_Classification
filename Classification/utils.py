from torch.utils import tensorboard
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from typing import List, Tuple

from PIL import Image
def create_writer( 
                  model_name: str, 
                  epoch:int,
                  batch_size:int,
                  learning_rate:int,
                  seed:int,
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
 
    from datetime import datetime
    import os

    
    if extra:
        # Create log directory path
        log_dir = os.path.join("runs",model_name, 'epochs='+str(epoch),'batch_size= '+str(batch_size),'lr='+str(learning_rate),'seed='+str(seed) ,extra)
    else:
        log_dir = os.path.join("runs", model_name, 'epochs='+str(epoch),'batch_size= '+str(batch_size),'lr='+str(learning_rate),'seed='+str(seed))
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return tensorboard.SummaryWriter(log_dir=log_dir)
def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
