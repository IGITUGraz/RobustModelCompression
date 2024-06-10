#!/bin/bash

model_name="NAME_OF_PRUNED_MODEL"
path="/MODEL_PATH/" + $model_name
arch="wide_resnet50_2"
pruning_type="global"
pruning_ratio=0.7
augmentation="pixmix" 

# LWM pruning and finetuning
python -u prune_with_augmentation.py --augmentation $augmentation --pretrained --arch $arch --model_name $model_name --prune --pruning_type $pruning_type --pruning_ratio $pruning_ratio --lr 0.01 --batch_size 256 --epochs 20

# Evaluate on ImageNet
python -u evaluate_imagenet_c.py --arch $arch --path $path

# Evaluate on ImageNet-C
python -u evaluate_imagenet_c.py --arch $arch --path $path --corruption 

#Evalate on ImageNet-Patch
for ((seed=0;seed<=19;seed++)); do
  python -u evaluate_imagenet_patch.py --arch $arch --path $path  --patch "bernoulli" --seed $seed
  python -u evaluate_imagenet_patch.py --arch $arch --path $path  --patch "srs_rn18" --seed $seed  
  python -u evaluate_imagenet_patch.py --arch $arch --path $path  --patch "adv" --seed $seed 
  python -u evaluate_imagenet_patch.py --arch $arch --path $path  --patch "black" --seed $seed
  
done