"""
Contains the training and test steps + training the model
"""
import torch
from torch.utils import tensorboard

def training_step(model:torch.nn.Module, train_dataloader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module, optimizer:torch.optim.Optimizer,  device:torch.device):
    model.train()
    train_loss,train_acc=0,0
    for X,y in train_dataloader:
      X,y= X.to(
        device), y.to(device)
      preds= model(X)

      loss= loss_fn(preds,y.reshape(-1,1).float())
      train_loss+=loss
      train_acc+= (y==torch.round(torch.sigmoid(preds.squeeze()))).sum().item()/len(y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    train_acc=train_acc/len(train_dataloader)
    train_loss=train_loss/len(train_dataloader)
    return train_loss, train_acc

def test_step(model:torch.nn.Module, test_dataloader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module,   device:torch.device):
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Stores metrics to specified writer log_dir if present.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").
      writer: A SummaryWriter() instance to log model results to.

    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for
      each epoch.
    """
    model.eval()
    with torch.inference_mode():
      test_loss,test_acc=0,0
      for X,y in test_dataloader:
        X,y= X.to(device), y.to(device)
        preds= model(X)

        loss= loss_fn(preds,y.reshape(-1,1).float())
        test_loss+=loss
        test_acc+= (y==torch.round(torch.sigmoid(preds.squeeze()))).sum().item()/len(y)
      test_acc=test_acc/len(test_dataloader)
      test_loss=test_loss/len(test_dataloader)
    return test_loss, test_acc

def train(model:torch.nn.Module, train_dataloader:torch.utils.data.DataLoader, test_dataloader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module, optimizer:torch.optim.Optimizer,epochs:int,  device:torch.device, writer: tensorboard.writer.SummaryWriter):
  results={
      'train_loss':[],
      'test_loss':[],
      'train_acc':[],
      'test_acc':[],
  }

  for epoch in range(epochs):
    train_loss,train_acc=training_step(model,train_dataloader,loss_fn,optimizer,device)

    test_loss, test_acc= test_step(model,test_dataloader,loss_fn,device)

    print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )
            # Update results dictionary
    results["train_loss"].append(train_loss.item())
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss.item())
    results["test_acc"].append(test_acc)

    if writer:
      writer.add_scalars(
          main_tag="Loss",
          tag_scalar_dict={
              'train loss': train_loss,
              'test loss': test_loss
          },
          global_step=epoch,
      )
      writer.add_scalars(
          main_tag="Accuracy",
          tag_scalar_dict={
              'train acc': train_acc,
              'test acc': test_acc
          },
          global_step=epoch,
      )
      writer.close()
  return results



