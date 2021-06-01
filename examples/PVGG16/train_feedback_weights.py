#########################
# In this script we train p-EfficientNets on ImageNet
# We use the pretrained model and only train feedback connections.
#########################
#%%
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np



import os
import toml
import time

#TODO : Milad
from .... import .... as PVGG16


################################################
#       Global configs
################################################

class Args():

    def __init__(self,config_file):

        config = toml.load(config_file)
        for k,v in config.items():
            setattr(self,k,v)

args = Args('train_config.toml')



#%%

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.GPU_TO_USE)


# Setup the training
if args.RANDOM_SEED:
    np.random.seed(args.RANDOM_SEED)
    torch.manual_seed(args.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



if not os.path.exists(args.TASK_NAME):
    print (f'Creating a new dir : {args.TASK_NAME}')
    os.mkdir(args.TASK_NAME)

device = torch.device('cuda:0')



################################################
#          Net , optimizers
################################################
net = torchvision.models.vgg16(pretrained=True)
pnet = PVGG(net,build_graph=True,random_init=False)
pnet.to(device)

NUMBER_OF_PCODERS = pnet.number_of_pcoders


loss_function = nn.MSELoss()
optimizer = optim.SGD([{'params':getattr(pnet,f"pcoder{x+1}").pmodule.parameters()} for x in range(NUMBER_OF_PCODERS)],
                        lr=args.LR,
                        weight_decay=args.WEIGHT_DECAY
                    )

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=3)

################################################
#       Dataset and train-test helpers
################################################
transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

data_root  = args.IMAGENET_DIR
train_ds     = ImageNet(data_root, split='train', download=False, transform=transform_val)
train_loader = torch.utils.data.DataLoader(train_ds,  batch_size=args.BATCHSIZE, shuffle=True, drop_last=False,num_workers=args.NUM_WORKERS,pin_memory=True)

val_ds     = ImageNet(data_root, split='val', download=False, transform=transform_val)
val_loader = torch.utils.data.DataLoader(val_ds,  batch_size=args.BATCHSIZE, shuffle=True, drop_last=False,num_workers=args.NUM_WORKERS,pin_memory=True)



def train_pcoders(net, epoch, writer,train_loader,verbose=True):

    ''' A training epoch '''
    
    net.train()

    tstart = time.time()
    for batch_index, (images, _) in enumerate(train_loader):
        net.reset()
        images = images.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        for i in range(NUMBER_OF_PCODERS):
            if i == 0:
                a = loss_function(net.pcoder1.prd, images)
                loss = a
            else:
                pcoder_pre = getattr(net, f"pcoder{i}")
                pcoder_curr = getattr(net, f"pcoder{i+1}")
                a = loss_function(pcoder_curr.prd, pcoder_pre.rep)
                loss += a
            sumwriter.add_scalar(f"MSE Train/PCoder{i+1}", a.item(), epoch * len(train_loader) + batch_index)

        loss.backward()
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.BATCHSIZE + len(images),
            total_samples=len(train_loader.dataset)
        ))
        print ('Time taken:',time.time()-tstart)
        sumwriter.add_scalar(f"MSE Train/Sum", loss.item(), epoch * len(train_loader) + batch_index)


def test_pcoders(net, epoch, writer,test_loader,verbose=True):

    ''' A testing epoch '''

    net.eval()

    tstart = time.time()
    final_loss = [0 for i in range(NUMBER_OF_PCODERS)]
    for batch_index, (images, _) in enumerate(test_loader):
        net.reset()
        images = images.to(device)
        with torch.no_grad():
            outputs = net(images)
        for i in range(NUMBER_OF_PCODERS):
            if i == 0:
                final_loss[i] += loss_function(net.pcoder1.prd, images).item()
            else:
                pcoder_pre = getattr(net, f"pcoder{i}")
                pcoder_curr = getattr(net, f"pcoder{i+1}")
                final_loss[i] += loss_function(pcoder_curr.prd, pcoder_pre.rep).item()
    
    loss_sum = 0
    for i in range(NUMBER_OF_PCODERS):
        final_loss[i] /= len(test_loader)
        loss_sum += final_loss[i]
        sumwriter.add_scalar(f"MSE Test/PCoder{i+1}", final_loss[i], epoch * len(test_loader))
    sumwriter.add_scalar(f"MSE Test/Sum", loss_sum, epoch * len(test_loader))

    print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
        loss_sum,
        optimizer.param_groups[0]['lr'],
        epoch=epoch,
        trained_samples=batch_index * args.BATCHSIZE + len(images),
        total_samples=len(test_loader.dataset)
    ))
    print ('Time taken:',time.time()-tstart)





################################################
#        Load checkpoints if given...
################################################

if args.RESUME:

    assert len(args.RESUME_CKPTS) == NUMBER_OF_PCODERS ; 'the number os ckpts provided is not equal to the number of pcoders'

    print ('-'*30)
    print (f'Loading checkpoint from {args.RESUME_CKPTS}')
    print ('-'*30)

    for x in range(NUMBER_OF_PCODERS):
        checkpoint = torch.load(args.RESUME_CKPTS[x])
        args.START_EPOCH = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        getattr(pnet,f"pcoder{x+1}").pmodule.load_state_dict({k[len('pmodule.'):]:v for k,v in checkpoint['pcoderweights'].items()})

    print ('Checkpoint loaded...')

else :
    print ("Training from scratch...")



# summarywriter
sumwriter = SummaryWriter(f'{args.LOG_DIR}/{args.TASK_NAME}', filename_suffix=f'')
optimizer_text = f"Optimizer   :{args.OPTIM_NAME}  \n lr          :{optimizer.defaults['lr']} \n batchsize   :{args.BATCHSIZE} \n weight_decay:{args.WEIGHT_DECAY} \n {args.EXTRA_STUFF_YOU_WANT_TO_ADD_TO_TB}"
sumwriter.add_text('Parameters',optimizer_text,0)


################################################
#              Train loops
################################################
for epoch in range(args.START_EPOCH, args.NUM_EPOCHS):
    train_pcoders(pnet, epoch, sumwriter,train_loader)

    test_pcoders(pnet, epoch, sumwriter,val_loader)
    

    for pcod_idx in range(NUMBER_OF_PCODERS):
        torch.save({
                    'pcoderweights':getattr(pnet,f"pcoder{pcod_idx+1}").state_dict(),
                    'optimizer'    :optimizer.state_dict(),
                    'epoch'        :epoch,
                    }, f'{args.TASK_NAME}/pnet_pretrained_pc{pcod_idx+1}_{epoch:03d}.pth')



