#!/bin/bash
#SBATCH -p batch_sw_grad
#SBATCH -w sw8
#SBATCH --gres gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=20G
#SBATCH --time=7-00:00:0


python /data/kide004/repos/VideoMAE/dataset/epic/resize.py /data/datasets/epickitchens/EPIC-KITCHENS/ /data/datasets/epic_resized/EPIC-KITCHENS/ --dense --level 2 --to-mp4