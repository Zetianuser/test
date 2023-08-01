#cd ~/cod/experiment/SAM-COD-PyTorch-main-v1
#echo "start runing cod-sam-vit-h-attention-prompt_global"
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes 1 --node_rank=0 --nproc_per_node 2  train.py --config configs/cod-sam-vit-h-attention-prompt_global.yaml
#wait
#echo "cod-sam-vit-h-attention-prompt_global finish"

#cd ~/cod/experiment/SAM-COD-PyTorch-main-v1
#echo "start runing cod-sam-vit-h-attention-prompt_all"
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes 1 --node_rank=0 --nproc_per_node 2  train.py --config configs/cod-sam-vit-h-attention-prompt_all.yaml
#wait
#echo "cod-sam-vit-h-attention-prompt_all finish"

#cd ~/cod/experiment/SAM-COD-PyTorch-main-v1
#echo "start runing cod-sam-vit-h-attention-prompt_block1"
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes 1 --node_rank=0 --nproc_per_node 2  train.py --config configs/cod-sam-vit-h-attention-prompt_block1.yaml
#wait
#echo "cod-sam-vit-h-attention-prompt_block1 finish"

#cd ~/cod/experiment/SAM-COD-PyTorch-main-v1
#echo "start runing cod-sam-vit-h-attention-prompt_block4"
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes 1 --node_rank=0 --nproc_per_node 2  train.py --config configs/cod-sam-vit-h-attention-prompt_block4.yaml
#wait
#echo "cod-sam-vit-h-attention-prompt_block4 finish"

#cd ~/cod/experiment/SAM-COD-PyTorch-main-v1
#echo "start runing cod-sam-vit-h-attention-prompt_local"
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes 1 --node_rank=0 --nproc_per_node 2  train.py --config configs/cod-sam-vit-h-attention-prompt_local.yaml
#wait
#echo "cod-sam-vit-h-attention-prompt_local finish"

cd ~/cod/experiment/SAM-COD-PyTorch-main-v1
echo "start runing cod-sam-vit-h-attention-prompt_conv"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes 1 --node_rank=0 --nproc_per_node 2  train.py --config configs/cod-sam-vit-h-attention-prompt_conv.yaml
wait
echo "cod-sam-vit-h-attention-prompt_conv finish"

#cd ~/cod/experiment/SAM-COD-PyTorch-main-v1
#echo "start runing cod-sam-vit-h-attention-prompt_attention"
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes 1 --node_rank=0 --nproc_per_node 2  train.py --config configs/cod-sam-vit-h-attention-prompt_attention.yaml
#wait
#echo "cod-sam-vit-h-attention-prompt_attention finish"

#cd ~/cod/experiment/SAM-COD-PyTorch-main-v1
#echo "start runing cod-sam-vit-h-attention-prompt_deformable"
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes 1 --node_rank=0 --nproc_per_node 2  train.py --config configs/cod-sam-vit-h-attention-prompt_deformable.yaml
#wait
#echo "cod-sam-vit-h-attention-prompt_deformable finish"

#cd ~/cod/experiment/SAM-COD-PyTorch-main-v1
#echo "start runing cod-sam-vit-h-attention-prompt_deformable2"
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes 1 --node_rank=0 --nproc_per_node 2  train.py --config configs/cod-sam-vit-h-attention-prompt_deformable2.yaml
#wait
#echo "cod-sam-vit-h-attention-prompt_deformable2 finish"

#cd ~/cod/experiment/SAM-COD-PyTorch-main-v1
#echo "start runing cod-sam-vit-h-attention-prompt_global_conv"
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes 1 --node_rank=0 --nproc_per_node 2  train.py --config configs/cod-sam-vit-h-attention-prompt_global_conv.yaml
#wait
#echo "cod-sam-vit-h-attention-prompt_global_conv finish"

#cd ~/cod/experiment/SAM-COD-PyTorch-main-v1
#echo "start runing cod-sam-vit-h-attention-prompt_all_conv"
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes 1 --node_rank=0 --nproc_per_node 2  train.py --config configs/cod-sam-vit-h-attention-prompt_all_conv.yaml
#wait
#echo "cod-sam-vit-h-attention-prompt_all_conv finish"

#cd ~/cod/experiment/SAM-COD-PyTorch-main-v1
#echo "start runing cod-sam-vit-h-attention-prompt_block1_conv"
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes 1 --node_rank=0 --nproc_per_node 2  train.py --config configs/cod-sam-vit-h-attention-prompt_block1_conv.yaml
#wait
#echo "cod-sam-vit-h-attention-prompt_block1_conv finish"

#cd ~/cod/experiment/SAM-COD-PyTorch-main-v1
#echo "start runing cod-sam-vit-h-attention-prompt_block4_conv"
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes 1 --node_rank=0 --nproc_per_node 2  train.py --config configs/cod-sam-vit-h-attention-prompt_block4_conv.yaml
#wait
#echo "cod-sam-vit-h-attention-prompt_block4_conv finish"

#cd ~/cod/experiment/SAM-COD-PyTorch-main-v1
#echo "start runing cod-sam-vit-h-attention-prompt_local_conv"
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes 1 --node_rank=0 --nproc_per_node 2  train.py --config configs/cod-sam-vit-h-attention-prompt_local_conv.yaml
#wait
#echo "cod-sam-vit-h-attention-prompt_local_conv finish"
