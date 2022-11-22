#!/bin/bash
#SBATCH -p batch_sw_grad 
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=20G
#SBATCH --time=7-00:00:0


if [ ${SLURM_NODELIST} == sw8 ]
then
    echo -e "This node is: ${SLURM_NODELIST}\n"
    export NCCL_SOCKET_IFNAME=enp194s0
else
    echo -e "This node is: ${SLURM_NODELIST}\n"
    export NCCL_SOCKET_IFNAME=enp28s0f1
fi


cd ~/.cache
rm -r torch_extensions
export NCCL_DEBUG=INFO

DATA_PATH=/data/jong980812/project/VideoMAE_cross/dataset/mini_ssv2
MODEL_PATH=/data/jong980812/project/VideoMAE_cross/pretrained/ssv2_pretrained.pth
OUTPUT_DIR=/data/jong980812/project/VideoMAE_cross/result/DDP_3N_8G/OUT # weight저장.
MASTER_NODE=$1
torchrun --nproc_per_node=8 \
    --master_port $3 --nnodes=3 \
    --node_rank=$2 --master_addr=${MASTER_NODE} \
    /data/jong980812/project/VideoMAE_cross/run_cross_finetuning.py \
    --data_set MINI_SSV2 \
    --nb_classes 87 \
    --cross_attn \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 12 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 5 \
    --num_sample 1 \
    --num_frames 16 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100 \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --num_workers 8 \
    --seed 0 \
    --enable_deepspeed \
    --freeze_vmae \
    --warmup_epochs 15  