#!/bin/bash

MASTER_PORT=$((12000 + $RANDOM % 20000))
ORIGINAL_SH=/data/jong980812/project/VideoMAE_cross/scripts/original.sh
#원본 스크립트. 여기서 arg수정해야함.
echo "SET ORIGINAL_SH"

EACH_NODE_SH=/data/jong980812/project/VideoMAE_cross/result/DDP_3N_8G/SH
#복사될 위치. 
echo "SET EACH_NODE_SH"

MASTER_NODE=sw10
# 마스터 노드 설정.
echo "SET MASTER_NODE" 

SLURM_OUT_DIR=/data/jong980812/project/VideoMAE_cross/result/DDP_3N_8G/OUT/%j.out
SLURM_ERR_DIR=/data/jong980812/project/VideoMAE_cross/result/DDP_3N_8G/OUT/%j.err
#슬럼 OUTPUT, ERR 로그 찍는 곳.
echo "SET SLURM_OUT_DIR"

GPU_PER_NODE=8

#노드 이름을 순서대로 적는다. 
nodelist=("sw10" "sw12" "sw15")
echo -e "SHOW NODES: ${nodelist[@]}\n\n"

NODE_RANK=0
#건들일 필요없음.
for node in ${nodelist[@]}
do
    cp -i $ORIGINAL_SH $EACH_NODE_SH/$node.sh    
    echo "Make $node script"
    #Original SH를 EACHNODESH에 가서 
    #하나씩 노드별로 스크립트 폼 만들어줌.
    sbatch -w $node --gres=gpu:$GPU_PER_NODE --out $SLURM_OUT_DIR --error $SLURM_ERR_DIR $EACH_NODE_SH/$node.sh ${MASTER_NODE} ${NODE_RANK} ${MASTER_PORT}   

    sleep 1s
    #위에서 만든 스크립트를, 특정 $node에, $GPUPER_BATCH에 맞게 올림. sh뒤에 노드 랭크 1씩 더해서 넣어줌. 0~ (L-1)
    #if 문은, master 노드에서만 output찍힐 수 있도록함.
    echo -e "\n"
    let NODE_RANK+=1
done
