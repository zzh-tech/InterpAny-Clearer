import os
from argparse import ArgumentParser

if __name__ == '__main__':
    """
    cmd:
    RIFE:
    python train.py --model RIFE --variant T
    python train.py --model RIFE --variant D
    python train.py --model RIFE --variant TR
    python train.py --model RIFE --variant DR
    
    AMT-S:
    python train.py --model AMT-S --variant T
    python train.py --model AMT-S --variant D
    python train.py --model AMT-S --variant TR
    python train.py --model AMT-S --variant DR
    
    IFRNet:
    python train.py --model IFRNet --variant T
    python train.py --model IFRNet --variant D
    python train.py --model IFRNet --variant TR
    python train.py --model IFRNet --variant DR
    
    EMA-VFI:
    python train.py --model EMA-VFI --variant T
    python train.py --model EMA-VFI --variant D
    python train.py --model EMA-VFI --variant TR
    python train.py --model EMA-VFI --variant DR
    """
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='RIFE', help='[ RIFE | IFRNet | AMT-S | EMA-VFI ]')
    parser.add_argument('--variant', type=str, default='D', help='[ T | D | TR | DR ]')
    parser.add_argument('--num_gpus', type=int, default=4)
    args = parser.parse_args()

    cmd = ''

    if args.model == 'RIFE':
        os.chdir('models/DI-RIFE')
        if args.variant == 'T':
            cmd = f'python3 -m torch.distributed.launch --nproc_per_node={args.num_gpus} --master_port 29501 train_m.py --batch_size 16 --exp_name T-RIFE'
        if args.variant == 'D':
            cmd = 'python3 -m torch.distributed.launch --nproc_per_node={} --master_port 29501 train_sdi_m.py --world_size={}'.format(args.num_gpus, args.num_gpus)
            cmd += ' --batch_size 16 --exp_name D-RIFE --sdi_name dis_index_{}_{}_{}.npy --clip --blur --cont'
        if args.variant == 'TR':
            cmd = 'python3 -m torch.distributed.launch --nproc_per_node={}'.format(args.num_gpus)
            cmd += ' --master_port 29501 train_sdi_m_mask_recur.py --batch_size 16 --sdi_name dis_index_{}_{}_{}.npy --exp_name TR-RIFE --clip --cont --no_avg --blur --no_order --no_mask --no_sdi'
        if args.variant == 'DR':
            cmd = 'python3 -m torch.distributed.launch --nproc_per_node={}'.format(args.num_gpus)
            cmd += ' --master_port 29501 train_sdi_m_mask_recur.py --batch_size 16 --sdi_name dis_index_{}_{}_{}.npy --exp_name DR-RIFE --clip --cont --no_avg --blur --no_order --no_mask'
    elif args.model == 'AMT-S':
        os.chdir('models/DI-AMT-and-IFRNet')
        if args.variant == 'T':
            cmd = f'sh ./scripts/train.sh {args.num_gpus} cfgs/AMT-S_septuplet_wofloloss.yaml 14514'
        if args.variant == 'D':
            cmd = f'sh ./scripts/train.sh {args.num_gpus} cfgs/SDI-AMT-S_septuplet_wofloloss.yaml 14514'
        if args.variant == 'TR':
            cmd = f'sh ./scripts/train.sh {args.num_gpus} cfgs/R-AMT-S_v1_septuplet_wofloloss.yaml 14514'
        if args.variant == 'DR':
            cmd = f'sh ./scripts/train.sh {args.num_gpus} cfgs/SDI-R-AMT-S_v1_septuplet_wofloloss.yaml 14514'
    elif args.model == 'IFRNet':
        os.chdir('models/DI-AMT-and-IFRNet')
        if args.variant == 'T':
            cmd = f'sh ./scripts/train.sh {args.num_gpus} cfgs/IFRNet_septuplet_wofloloss.yaml 14514'
        if args.variant == 'D':
            cmd = f'sh ./scripts/train.sh {args.num_gpus} cfgs/SDI-IFRNet_septuplet_wofloloss.yaml 14514'
        if args.variant == 'TR':
            cmd = f'sh ./scripts/train.sh {args.num_gpus} cfgs/R-IFRNet_septuplet_wofloloss.yaml 14514'
        if args.variant == 'DR':
            cmd = f'sh ./scripts/train.sh {args.num_gpus} cfgs/SDI-R-IFRNet_septuplet_wofloloss.yaml 14514'
    elif args.model == 'EMA-VFI':
        os.chdir('models/DI-EMA-VFI')
        if args.variant == 'T':
            cmd = f'python -m torch.distributed.launch --nproc_per_node={args.num_gpus} --master_port 29501 train_sdi_m_mask.py --world_size {args.num_gpus} --batch_size 8 --exp_name T-EMA-VFI'
        if args.variant == 'D':
            cmd = f'python -m torch.distributed.launch --nproc_per_node={args.num_gpus} --master_port 29501 train_sdi_m_mask.py --world_size {args.num_gpus} --batch_size 8 --exp_name D-EMA-VFI --use_sdi'
        if args.variant == 'TR':
            cmd = f'python -m torch.distributed.launch --nproc_per_node={args.num_gpus} --master_port 29501 train_sdi_m_mask_recur.py --world_size {args.num_gpus} --batch_size 8 --exp_name TR-EMA-VFI'
        if args.variant == 'DR':
            cmd = f'python -m torch.distributed.launch --nproc_per_node={args.num_gpus} --master_port 29501 train_sdi_m_mask_recur.py --world_size {args.num_gpus} --batch_size 8 --exp_name DR-EMA-VFI --use_sdi'
    else:
        raise ValueError

    os.system(cmd)
