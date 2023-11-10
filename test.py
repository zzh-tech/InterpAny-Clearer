import os
from argparse import ArgumentParser

if __name__ == '__main__':
    """
    cmd:
    RIFE:
    python test.py --model RIFE --variant T
    python test.py --model RIFE --variant D
    python test.py --model RIFE --variant D --uniform
    python test.py --model RIFE --variant TR
    python test.py --model RIFE --variant DR 
    python test.py --model RIFE --variant DR --uniform
    
    AMT-S:
    python test.py --model AMT-S --variant T
    python test.py --model AMT-S --variant D
    python test.py --model AMT-S --variant D --uniform
    python test.py --model AMT-S --variant TR
    python test.py --model AMT-S --variant DR
    python test.py --model AMT-S --variant DR --uniform
    
    IFRNet:
    python test.py --model IFRNet --variant T
    python test.py --model IFRNet --variant D
    python test.py --model IFRNet --variant D --uniform
    python test.py --model IFRNet --variant TR
    python test.py --model IFRNet --variant DR
    python test.py --model IFRNet --variant DR --uniform
    
    EMA-VFI:
    python test.py --model EMA-VFI --variant T
    python test.py --model EMA-VFI --variant D
    python test.py --model EMA-VFI --variant D --uniform
    python test.py --model EMA-VFI --variant TR
    python test.py --model EMA-VFI --variant DR
    python test.py --model EMA-VFI --variant DR --uniform
    """
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='RIFE', help='[ RIFE | IFRNet | AMT-S | EMA-VFI ]')
    parser.add_argument('--variant', type=str, default='D', help='[ T | D | TR | DR ]')
    parser.add_argument('--uniform', action='store_true', help='use uniform distance map')
    args = parser.parse_args()

    cmd = ''

    if args.model == 'RIFE':
        os.chdir('models/DI-RIFE/benchmark')
        if args.variant == 'T':
            cmd = 'python Vimeo90K_m.py --model_dir ../../../checkpoints/RIFE/T-RIFE/train_m_log_official --testset_path ../../../dataset/vimeo_septuplet/'
        if args.variant == 'D':
            if args.uniform:
                cmd = 'python Vimeo90K_sdi_unif_m.py --model_dir ../../../checkpoints/RIFE/D-RIFE/train_sdi_log --testset_path ../../../dataset/vimeo_septuplet/ --sdi_name dis_index_{}_{}_{}.npy'
            else:
                cmd = 'python Vimeo90K_sdi_m.py --model_dir ../../../checkpoints/RIFE/D-RIFE/train_sdi_log --testset_path ../../../dataset/vimeo_septuplet/ --sdi_name dis_index_{}_{}_{}.npy --clip --blur'
        if args.variant == 'TR':
            cmd = 'python Vimeo90K_sdi_unif_m_recur.py --model_dir ../../../checkpoints/RIFE/TR-RIFE/train_sdi_log --testset_path ../../../dataset/vimeo_septuplet/ --sdi_name dis_index_{}_{}_{}.npy --iters 2'
        if args.variant == 'DR':
            if args.uniform:
                cmd = 'python Vimeo90K_sdi_unif_m_recur.py --model_dir ../../../checkpoints/RIFE/DR-RIFE/train_sdi_log --testset_path ../../../dataset/vimeo_septuplet/ --sdi_name dis_index_{}_{}_{}.npy --iters 2'
            else:
                cmd = 'python Vimeo90K_sdi_m_recur.py --model_dir ../../../checkpoints/RIFE/DR-RIFE/train_sdi_log --testset_path ../../../dataset/vimeo_septuplet/ --sdi_name dis_index_{}_{}_{}.npy --clip --blur --iters 2'
    elif args.model == 'AMT-S':
        os.chdir('models/DI-AMT-and-IFRNet/benchmarks')
        if args.variant == 'T':
            cmd = 'python Vimeo90K_m.py --testset_path ../../../dataset/vimeo_septuplet/ --ckpt ../../../checkpoints/AMT-S/T-AMT-S/ckpts/latest.pth --config ../../../checkpoints/AMT-S/T-AMT-S/T-AMT-S.yaml'
        if args.variant == 'D':
            if args.uniform:
                cmd = 'python Vimeo90K_sdi_m.py --testset_path ../../../dataset/vimeo_septuplet/ --ckpt ../../../checkpoints/AMT-S/D-AMT-S/ckpts/latest.pth --config ../../../checkpoints/AMT-S/D-AMT-S/D-AMT-S.yaml --unif'
            else:
                cmd = 'python Vimeo90K_sdi_m.py --testset_path ../../../dataset/vimeo_septuplet/ --ckpt ../../../checkpoints/AMT-S/D-AMT-S/ckpts/latest.pth --config ../../../checkpoints/AMT-S/D-AMT-S/D-AMT-S.yaml'
        if args.variant == 'TR':
            cmd = 'python Vimeo90K_sdi_m_recur.py --testset_path ../../../dataset/vimeo_septuplet/ --ckpt ../../../checkpoints/AMT-S/TR-AMT-S/ckpts/latest.pth --config ../../../checkpoints/AMT-S/TR-AMT-S/TR-AMT-S.yaml --iters 2 --unif'
        if args.variant == 'DR':
            if args.uniform:
                cmd = 'python Vimeo90K_sdi_m_recur.py --testset_path ../../../dataset/vimeo_septuplet/ --ckpt ../../../checkpoints/AMT-S/DR-AMT-S/ckpts/latest.pth --config ../../../checkpoints/AMT-S/DR-AMT-S/DR-AMT-S.yaml --iters 2 --unif'
            else:
                cmd = 'python Vimeo90K_sdi_m_recur.py --testset_path ../../../dataset/vimeo_septuplet/ --ckpt ../../../checkpoints/AMT-S/DR-AMT-S/ckpts/latest.pth --config ../../../checkpoints/AMT-S/DR-AMT-S/DR-AMT-S.yaml --iters 2'
    elif args.model == 'IFRNet':
        os.chdir('models/DI-AMT-and-IFRNet/benchmarks')
        if args.variant == 'T':
            cmd = 'python Vimeo90K_m.py --testset_path ../../../dataset/vimeo_septuplet/ --ckpt ../../../checkpoints/IFRNet/T-IFRNet/ckpts/latest.pth --config ../../../checkpoints/IFRNet/T-IFRNet/T-IFRNet.yaml'
        if args.variant == 'D':
            if args.uniform:
                cmd = 'python Vimeo90K_sdi_m.py --testset_path ../../../dataset/vimeo_septuplet/ --ckpt ../../../checkpoints/IFRNet/D-IFRNet/ckpts/latest.pth --config ../../../checkpoints/IFRNet/D-IFRNet/D-IFRNet.yaml --unif'
            else:
                cmd = 'python Vimeo90K_sdi_m.py --testset_path ../../../dataset/vimeo_septuplet/ --ckpt ../../../checkpoints/IFRNet/D-IFRNet/ckpts/latest.pth --config ../../../checkpoints/IFRNet/D-IFRNet/D-IFRNet.yaml'
        if args.variant == 'TR':
            cmd = 'python Vimeo90K_sdi_m_recur.py --testset_path ../../../dataset/vimeo_septuplet/ --ckpt ../../../checkpoints/IFRNet/TR-IFRNet/ckpts/latest.pth --config ../../../checkpoints/IFRNet/TR-IFRNet/TR-IFRNet.yaml --iters 2 --unif'
        if args.variant == 'DR':
            if args.uniform:
                cmd = 'python Vimeo90K_sdi_m_recur.py --testset_path ../../../dataset/vimeo_septuplet/ --ckpt ../../../checkpoints/IFRNet/DR-IFRNet/ckpts/latest.pth --config ../../../checkpoints/IFRNet/DR-IFRNet/DR-IFRNet.yaml --iters 2 --unif'
            else:
                cmd = 'python Vimeo90K_sdi_m_recur.py --testset_path ../../../dataset/vimeo_septuplet/ --ckpt ../../../checkpoints/IFRNet/DR-IFRNet/ckpts/latest.pth --config ../../../checkpoints/IFRNet/DR-IFRNet/DR-IFRNet.yaml --iters 2'
    elif args.model == 'EMA-VFI':
        os.chdir('models/DI-EMA-VFI/benchmark')
        if args.variant == 'T':
            cmd = 'python Vimeo90K_m.py --testset_path ../../../dataset/vimeo_septuplet/ --checkpoint ../../../checkpoints/EMA-VFI/T-EMA-VFI/train_sdi_log/'
        if args.variant == 'D':
            if args.uniform:
                cmd = 'python Vimeo90K_sdi_m.py --testset_path ../../../dataset/vimeo_septuplet/ --checkpoint ../../../checkpoints/EMA-VFI/D-EMA-VFI/train_sdi_log/ --unif'
            else:
                cmd = 'python Vimeo90K_sdi_m.py --testset_path ../../../dataset/vimeo_septuplet/ --checkpoint ../../../checkpoints/EMA-VFI/D-EMA-VFI/train_sdi_log/'
        if args.variant == 'TR':
            cmd = 'python Vimeo90K_sdi_m_recur.py --testset_path ../../../dataset/vimeo_septuplet/ --checkpoint ../../../checkpoints/EMA-VFI/TR-EMA-VFI/train_sdi_log/ --iters 2 --unif'
        if args.variant == 'DR':
            if args.uniform:
                cmd = 'python Vimeo90K_sdi_m_recur.py --testset_path ../../../dataset/vimeo_septuplet/ --checkpoint ../../../checkpoints/EMA-VFI/DR-EMA-VFI/train_sdi_log/ --iters 2 --unif'
            else:
                cmd = 'python Vimeo90K_sdi_m_recur.py --testset_path ../../../dataset/vimeo_septuplet/ --checkpoint ../../../checkpoints/EMA-VFI/DR-EMA-VFI/train_sdi_log/ --iters 2'
    else:
        raise ValueError

    os.system(cmd)
