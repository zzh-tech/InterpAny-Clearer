NUM_GPU=$1
CFG=$2
PORT=$3
CUDA_VISISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node $NUM_GPU \
--master_port $PORT train.py -c $CFG