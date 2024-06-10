import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import utils.cutout as cutout
import utils.pixmix_utils as pixmix_utils

pixmix_utils.IMAGE_SIZE = 224

def pixmix(orig, mixing_pic, preprocess, args):
  
  mixings = pixmix_utils.mixings
  tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
  if np.random.random() < 0.5:
    mixed = tensorize(augment_input(orig, args))
  else:
    mixed = tensorize(orig)
  
  for _ in range(np.random.randint(args.k_mixing + 1)):
    
    if np.random.random() < 0.5:
      aug_image_copy = tensorize(augment_input(orig, args))
    else:
      aug_image_copy = tensorize(mixing_pic)

    mixed_op = np.random.choice(mixings)
    mixed = mixed_op(mixed, aug_image_copy, args.beta)
    mixed = torch.clip(mixed, 0, 1)

  return normalize(mixed)


def augment_input(image, args):
  aug_list = pixmix_utils.augmentations_all if args.all_ops else pixmix_utils.augmentations
  op = np.random.choice(aug_list)
  return op(image.copy(), args.aug_severity)


class PixMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform PixMix."""

  def __init__(self, dataset, mixing_set, preprocess, args):
    self.dataset = dataset
    self.mixing_set = mixing_set
    self.preprocess = preprocess
    self.args = args

  def __getitem__(self, i):
    x, y = self.dataset[i]
    rnd_idx = np.random.choice(len(self.mixing_set))
    mixing_pic, _ = self.mixing_set[rnd_idx]
    return pixmix(x, mixing_pic, self.preprocess, self.args), y

  def __len__(self):
    return len(self.dataset)
  

class DataLoaderAugmentation:
    def __init__(self, args, master_path):
        self.args = args
        self.master_path = master_path

    def get_data_loaders(self):
        
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 

        # Load the dataset for specified augmentation type
        if self.args.augmentation == "pixmix":
            to_tensor = transforms.ToTensor()
            
            train_data = ImageFolder(
                self.master_path + 'imagenet/train/',
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip()
                ])
            )
            
            mixing_set = datasets.ImageFolder(
                self.args.mixing_set, 
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(224)
                ])
            )

            train_dataset = PixMixDataset(train_data, mixing_set, {'normalize': normalizer, 'tensorize': to_tensor}, self.args)

        if self.args.augmentation == "cutout":
            train_dataset = ImageFolder(
                self.master_path + 'imagenet/train/',
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalizer,
                    cutout.Cutout(n_holes= 1, length=self.args.cutout_length)
                ])
            )

        if self.args.augmentation == "augmix":
            train_dataset = ImageFolder(
                self.master_path + 'imagenet/train/',
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.AugMix(),
                    transforms.ToTensor(),
                    normalizer,
                ])
            )

        if self.args.augmentation == "autoaugment":
            train_dataset = ImageFolder(
                self.master_path + 'imagenet/train/',
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET),
                    transforms.ToTensor(),
                    normalizer,
                ])
            )

        if self.args.augmentation == "standard":
            train_dataset = ImageFolder(
                self.master_path + 'imagenet/train/',
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalizer,
                ])
            )

        #validation dataset
        val_loader = torch.utils.data.DataLoader(
        ImageFolder(
            self.master_path + 'imagenet/validation/', 
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalizer,
            ])
        ), batch_size=self.args.batch_size, shuffle=False, num_workers=8, pin_memory=True) 


        def wif(id):
            uint64_seed = torch.initial_seed()
            ss = np.random.SeedSequence([uint64_seed])
            np.random.seed(ss.generate_state(4))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=8, pin_memory=True, sampler=None, worker_init_fn=wif) 
        
        
        return train_loader, val_loader