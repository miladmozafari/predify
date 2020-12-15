import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from   torch.utils.data import DataLoader, Dataset, ConcatDataset
from   os.path import join as opj
from   PIL import Image
import numpy as np
import torch
from   torch.optim.lr_scheduler import _LRScheduler
from   torch.nn import Sequential, Conv2d, MaxPool2d, BatchNorm2d, ReLU, Flatten, Linear

class CIFAR_C(Dataset):
    """
    CIFAR10/100-C dataset including images of specific noise type and noise level.
    """
    def __init__(self, root, noise_type, noise_level, transform=None, target_transform=None):
        all_noises = ["brightness",
                    "contrast",
                    "defocus_blur",
                    "elastic_transform",
                    "fog",
                    "frost",
                    "gaussian_blur",
                    "gaussian_noise",
                    "glass_blur",
                    "impulse_noise",
                    "jpeg_compression",
                    "motion_blur",
                    "pixelate",
                    "saturate",
                    "shot_noise",
                    "snow",
                    "spatter",
                    "speckle_noise",
                    "zoom_blur"]
        if noise_type not in all_noises:
            raise Exception(f"Wrong noise type. Choose one from {','.join(all_noises)}.")
        if noise_level < 1 or noise_level > 5:
            raise Exception(f"noise_level should be an integer between 1-5.")
        self.data = np.load(opj(root,f"{noise_type}.npy"))[(noise_level-1)*10000:noise_level*10000]
        self.targets = np.load(opj(root,f"labels.npy"))[(noise_level-1)*10000:noise_level*10000].astype(np.int)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)

def get_training_dataloader(mean, std, cifar10=False, simple=False, batch_size=16, num_workers=2, shuffle=True, root='./data', noise_transform=None, after_norm=True, drop_last=False):
    """ return training dataloader
    Args:
        mean: mean of cifar100/10 training dataset
        std: std of cifar100/10 training dataset
        path: path to cifar100/10 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: train_data_loader:torch dataloader object
    """
    if simple:
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    else:
        transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    if noise_transform is not None:
        if after_norm:
            transform_list.append(noise_transform)
        else:
            transform_list.insert(1, noise_transform)

    transform_train = transforms.Compose(transform_list)
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    if cifar10:
        cifar_training = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)    
    else:
        cifar_training = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    cifar_training_loader = DataLoader(
        cifar_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, drop_last=drop_last)

    return cifar_training_loader

def get_test_dataloader(mean, std, cifar10=False, batch_size=16, num_workers=2, shuffle=True, root='./data', noise_transform=None, sampler=None, after_norm=True, drop_last=False):
    """ return training dataloader
    Args:
        mean: mean of cifar100/10 test dataset
        std: std of cifar100/10 test dataset
        path: path to cifar100/10 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
        after_norm: if true (false) applies the noise transformation after (before) normalization
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    if noise_transform is not None:
        if after_norm:
            transform_list.append(noise_transform)
        else:
            transform_list.insert(1, noise_transform)

    transform_test = transforms.Compose(transform_list)
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    if cifar10:
        cifar_test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)    
    else:
        cifar_test = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    cifar_test_loader = DataLoader(
        cifar_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, sampler=sampler, drop_last=drop_last)

    return cifar_test_loader

def get_cifarc_dataloader(mean=None, std=None, batch_size=16, num_workers=2, shuffle=True, root='./data', noise_type=None, noise_level=None, sampler=None, drop_last=False):
    """ return cifar10/100-c dataloader
    Args:
        mean: normalization mean
        std: normalization std
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
        root: path to cifar10/100-c dataset
        noise_type: "brightness",
                    "contrast",
                    "defocus_blur",
                    "elastic_transform",
                    "fog",
                    "frost",
                    "gaussian_blur",
                    "gaussian_noise",
                    "glass_blur",
                    "impulse_noise",
                    "jpeg_compression",
                    "motion_blur",
                    "pixelate",
                    "saturate",
                    "shot_noise",
                    "snow",
                    "spatter",
                    "speckle_noise",
                    "zoom_blur"
        noise_level: integer in range [1,5]
        sampler: data sampler
    Returns: cifarc_loader
    """
    all_noises = ["brightness",
                    "contrast",
                    "defocus_blur",
                    "elastic_transform",
                    "fog",
                    "frost",
                    "gaussian_blur",
                    "gaussian_noise",
                    "glass_blur",
                    "impulse_noise",
                    "jpeg_compression",
                    "motion_blur",
                    "pixelate",
                    "saturate",
                    "shot_noise",
                    "snow",
                    "spatter",
                    "speckle_noise",
                    "zoom_blur"]

    transform_list = [
        transforms.ToTensor(),
    ]
    if mean is not None and std is not None:
        transform_list.append(transforms.Normalize(mean, std))
    transform_test = transforms.Compose(transform_list)

    if noise_type is None and noise_level is not None:
        all_ds = []
        for noise in all_noises:
            all_ds.append(CIFAR_C(root, noise, noise_level, transform=transform_test))
        cifarc = ConcatDataset(all_ds)
    elif noise_type is not None and noise_level is None:
        all_ds = []
        for i in range(1,6):
            all_ds.append(CIFAR_C(root, noise_type, i, transform=transform_test))
        cifarc = ConcatDataset(all_ds)
    else:
        cifarc  = CIFAR_C(root, noise_type, noise_level, transform=transform_test)

    cifarc_loader = DataLoader(
        cifarc, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, sampler=sampler, drop_last=drop_last)

    return cifarc_loader

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class NetArgs():
  def __init__(self, gpu=True, workers=2, batch_size=128, shuffle=True, warm=1, init_lr=0.1, net_name='resnet18'):
    self.gpu  = gpu
    self.w    = workers
    self.b    = batch_size
    self.s    = shuffle
    self.warm = warm
    self.lr   = init_lr
    self.net  = net_name

def to_pair(input):
    if isinstance(input, tuple):
        return input[:2]
    else:
        return input, input

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + (torch.randn(tensor.size()) * self.std + self.mean)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddSaltPepperNoise(object):
    def __init__(self, probability=0.1, salt_v=1, pepper_v=-1):
        self.p = probability
        self.salt_v = salt_v
        self.pepper_v = pepper_v
        
    def __call__(self, tensor):
        count  = int(self.p * tensor.nelement())
        salt   = count // 2
        idx    = torch.randperm(tensor.nelement())
        
        noisy = tensor.clone().view(-1)
        noisy[idx[:salt]]      = self.salt_v
        noisy[idx[salt:count]] = self.pepper_v
        return noisy.view(tensor.size())
    
    def __repr__(self):
        return self.__class__.__name__ + '(probability={0:0.3f})'.format(self.p)

class AddSaltPepperNoiseVA(object):
    def __init__(self, probability=0.1):
        self.salt   = probability/2
        self.pepper = 1 - self.salt

    def __call__(self, tensor):
        saltNpepper = torch.rand(tensor.shape[-2], tensor.shape[-1]).repeat(3,1,1)
        noisy = tensor.clone()
        
        salt_v   = torch.max(tensor)
        pepper_v = torch.min(tensor)
        noisy = torch.where(saltNpepper >= self.salt, noisy, salt_v)
        noisy = torch.where(saltNpepper <= self.pepper, noisy, pepper_v)
        
        return noisy
    
    def __repr__(self):
        return self.__class__.__name__ + '(probability={0:0.3f})'.format(self.p)