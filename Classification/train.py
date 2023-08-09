import torchvision.models as M
import argparse
from model_factory import get_pretrained_model
from data_setup import create_dataloaders
from engine import train
from utils import set_seeds, create_writer
import torch
from torch import nn
from pathlib import Path

def training(args):
  device='cuda' if torch.cuda.is_available() else 'cpu'


  model,weights,transfo = get_pretrained_model(args.model_name)

  train_dataloader , test_dataloader,class_names=create_dataloaders( batch_size=args.batch_size,transforms=transfo,num_workers=args.num_workers)
  set_seeds(args.seed)
  train(model.to(device), train_dataloader,test_dataloader,nn.BCEWithLogitsLoss(),torch.optim.Adam(model.parameters(),lr=args.lr),args.epochs,device,create_writer(model_name=f'{model}'.split('(')[0],epoch=args.epochs,batch_size=args.batch_size,learning_rate=args.lr,seed=args.seed))
if __name__ == '__main__':
    VALID_MODELS= ['alexnet','efficient_net_b7']

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch XrayChest Classification')

    parser.add_argument('--model_name', metavar='N',required=True,
                        help='the model for training', choices=VALID_MODELS)

    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')

    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=.01, metavar='LR',
                        help='learning rate (default: .01)')

    parser.add_argument('--num_workers', type=int, default=1, metavar='W',
                        help='number of workers (default: 1)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    training(args)
