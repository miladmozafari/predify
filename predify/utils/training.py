from typing import Callable
import torch

def train_pcoders(net: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_function: Callable, epoch: int, train_loader: torch.utils.data.DataLoader, device: str, writer: torch.utils.tensorboard.SummaryWriter=None):
    r"""
    Trains the feedback modules of PCoders using a distance between the prediction of a PCoder and the
    representation of the PCoder below.

    Args:
        net (torch.nn.Module): Predified network including all the PCoders
        optimizer (torch.optim.Optimizer): PyTorch-compatible optimizer object
        loss_function (Callable): A callable function that receives two tensors and returns the distance between them
        epoch (int): Training epoch number
        train_loader (torch.utils.data.DataLoader): DataLoader for training samples
        writer (torch.utils.tensorboard.SummaryWrite, optional): Tensorboard summary writer to track training history. Default: None
        device (str): Training device (e.g. 'cpu', 'cuda:0')
    """
    
    net.train()
    net.backbone.eval()

    nb_trained_samples = 0
    for batch_index, (images, _) in enumerate(train_loader):
        net.reset()
        images = images.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        for i in range(net.number_of_pcoders):
            if i == 0:
                a = loss_function(net.pcoder1.prd, images)
                loss = a
            else:
                pcoder_pre = getattr(net, f"pcoder{i}")
                pcoder_curr = getattr(net, f"pcoder{i+1}")
                a = loss_function(pcoder_curr.prd, pcoder_pre.rep)
                loss += a
            if writer is not None:
                writer.add_scalar(f"MSE Train/PCoder{i+1}", a.item(), (epoch-1) * len(train_loader) + batch_index)
        
        nb_trained_samples += images.shape[0]

        loss.backward()
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}'.format(
            loss.item(),
            epoch=epoch,
            trained_samples=nb_trained_samples,
            total_samples=len(train_loader.dataset)
        ))
        if writer is not None:
            writer.add_scalar(f"MSE Train/Sum", loss.item(), (epoch-1) * len(train_loader) + batch_index)


def eval_pcoders(net: torch.nn.Module, loss_function: Callable, epoch: int, eval_loader: torch.utils.data.DataLoader, device: str, writer: torch.utils.tensorboard.SummaryWriter=None):
    r"""
    Evaluates the feedback modules of PCoders using a distance between the prediction of a PCoder and the
    representation of the PCoder below.

    Args:
        net (torch.nn.Module): Predified network including all the PCoders
        loss_function (Callable): A callable function that receives two tensors and returns the distance between them
        epoch (int): Evaluation epoch number
        test_loader (torch.utils.data.DataLoader): DataLoader for evaluation samples
        writer (torch.utils.tensorboard.SummaryWrite, optional): Tensorboard summary writer to track evaluation history. Default: None
        device (str): Training device (e.g. 'cpu', 'cuda:0')
    """

    net.eval()

    final_loss = [0 for i in range(net.number_of_pcoders)]
    for batch_index, (images, _) in enumerate(eval_loader):
        net.reset()
        images = images.to(device)
        with torch.no_grad():
            outputs = net(images)
        for i in range(net.number_of_pcoders):
            if i == 0:
                final_loss[i] += loss_function(net.pcoder1.prd, images).item()
            else:
                pcoder_pre = getattr(net, f"pcoder{i}")
                pcoder_curr = getattr(net, f"pcoder{i+1}")
                final_loss[i] += loss_function(pcoder_curr.prd, pcoder_pre.rep).item()
    
    loss_sum = 0
    for i in range(net.number_of_pcoders):
        final_loss[i] /= len(eval_loader)
        loss_sum += final_loss[i]
        if writer is not None:
            writer.add_scalar(f"MSE Eval/PCoder{i+1}", final_loss[i], epoch-1)
            
            
    print('Training Epoch: {epoch} [{evaluated_samples}/{total_samples}]\tLoss: {:0.4f}'.format(
        loss_sum,
        epoch=epoch,
        evaluated_samples=len(eval_loader.dataset),
        total_samples=len(eval_loader.dataset)
    ))
    if writer is not None:
        writer.add_scalar(f"MSE Eval/Sum", loss_sum, epoch-1)