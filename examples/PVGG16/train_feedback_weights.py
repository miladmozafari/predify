#########################
# In this script we train PVGG on ImageNet
# We use the pretrained model and only train feedback connections.
#########################
#%%
import torch
import torchvision
ddimport torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np



import os
import toml
import time




################################################
#       Global configs
################################################

class Args():

    def __init__(self,config_file):

        config = toml.load(config_file)
        for k,v in config.items():
            setattr(self,k,v)
    
    def print_params(self):
        for x in vars(self):
            print ("{:<20}: {}".format(x, getattr(args, x)))




args = Args('train_config.toml')
args.print_params()


#%%

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.GPU_TO_USE)


# Setup the training
if args.RANDOM_SEED:
    np.random.seed(args.RANDOM_SEED)
    torch.manual_seed(args.RANDOM_SEED)
    torch.backends.cudnn.deterministic = args.CUDNN_DETERMINISTIC
    torch.backends.cudnn.benchmark = args.CUDNN_BENCHMARK



if not os.path.exists(args.SAVE_DIR):
    print (f'Creating a new dir : {args.SAVE_DIR}')
    os.mkdir(args.SAVE_DIR)

device = torch.device('cuda:0')



################################################
#          Net , optimizers
################################################
## Change this to change the network
from pvgg16_separate import PVGGSeparateHP as PVGG16


net = torchvision.models.vgg16(pretrained=True)
pnet = PVGG16(net,build_graph=True,random_init=False)
pnet.to(device)

NUMBER_OF_PCODERS = pnet.number_of_pcoders


loss_function = nn.MSELoss()

if args.OPTIM_NAME=='sgd':
    optimizer = optim.SGD([{'params':getattr(pnet,f"pcoder{x+1}").pmodule.parameters()} for x in range(NUMBER_OF_PCODERS)],
                            lr=args.LR,
                            weight_decay=args.WEIGHT_DECAY
                        )


if args.SCHEDULER == 'cosine_annealing':
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

if args.RESUME_TRAINING:

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
sumwriter = SummaryWriter(f'{args.SAVE_DIR}/{args.TB_DIR}', filename_suffix=f'')

main_str = ''
for x in vars(args):
    main_str += f"{x:<20}: {getattr(args,x)}\n"
sumwriter.add_text( 'Parameters',f"{main_str}\n{args.EXTRA_STR_YOU_WANT_TO_ADD_TO_TB }",0)


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



