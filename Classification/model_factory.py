import torchvision.models as M
from torch import nn
import torch
VALID_MODELS= ['alexnet','efficient_net_b7']

def get_pretrained_model(model_name):
  device='cuda' if torch.cuda.is_available() else 'cpu'

  
  if model_name == 'alexnet':
    model=M.alexnet()
    weights=M.AlexNet_Weights.DEFAULT
    transfo=weights.transforms()
    for param in model.features.parameters():
      param.requires_grad=False
    model.classifier= nn.Sequential(
      nn.Dropout(p=0.2),
      nn.Linear(in_features=9216,out_features=1)
    ).to(device)
  if model_name == 'efficient_net_b7':
    model=M.efficientnet_b7()
    weights=M.EfficientNet_B7_Weights.DEFAULT
    transfo=weights.transforms()
    for param in model.features.parameters():
      param.requires_grad=False
    model.classifier= nn.Sequential(
      nn.Dropout(p=0.2),
      nn.Linear(in_features=2560,out_features=1)
    ).to(device)
  else: Exception(
            f"Model %s is not in known models ({','.join(VALID_MODELS)}")
  return model ,weights,transfo
