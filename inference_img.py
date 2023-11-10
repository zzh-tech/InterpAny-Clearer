import os
import os.path as osp
from argparse import ArgumentParser
from glob import glob

if __name__ == '__main__':
    """
    cmd:
    RIFE:
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model RIFE --variant D --checkpoint ./checkpoints/RIFE/D-RIFE-pro --save_dir ./results/I0_results_D-RIFE-pro --num 1 1 1 --gif
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model RIFE --variant DR --checkpoint ./checkpoints/RIFE/DR-RIFE-pro --save_dir ./results/I0_results_DR-RIFE-pro --num 1 1 1 --gif
    
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model RIFE --variant T --checkpoint ./checkpoints/RIFE/T-RIFE --save_dir ./results/I0_results_T-RIFE --num 1 1 1 --gif
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model RIFE --variant D --checkpoint ./checkpoints/RIFE/D-RIFE --save_dir ./results/I0_results_D-RIFE --num 1 1 1 --gif
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model RIFE --variant TR --checkpoint ./checkpoints/RIFE/TR-RIFE --save_dir ./results/I0_results_TR-RIFE --num 1 1 1 --gif
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model RIFE --variant DR --checkpoint ./checkpoints/RIFE/DR-RIFE --save_dir ./results/I0_results_DR-RIFE --num 1 1 1 --gif
    
    AMT-S:
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model AMT-S --variant T --checkpoint ./checkpoints/AMT-S/T-AMT-S --save_dir ./results/I0_results_T-AMT-S --num 1 1 1 --gif
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model AMT-S --variant D --checkpoint ./checkpoints/AMT-S/D-AMT-S --save_dir ./results/I0_results_D-AMT-S --num 1 1 1 --gif
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model AMT-S --variant TR --checkpoint ./checkpoints/AMT-S/TR-AMT-S --save_dir ./results/I0_results_TR-AMT-S --num 1 1 1 --gif
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model AMT-S --variant DR --checkpoint ./checkpoints/AMT-S/DR-AMT-S --save_dir ./results/I0_results_DR-AMT-S --num 1 1 1 --gif
    
    IFRNet:
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model IFRNet --variant T --checkpoint ./checkpoints/IFRNet/T-IFRNet --save_dir ./results/I0_results_T-IFRNet --num 1 1 1 --gif
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model IFRNet --variant D --checkpoint ./checkpoints/IFRNet/D-IFRNet --save_dir ./results/I0_results_D-IFRNet --num 1 1 1 --gif
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model IFRNet --variant TR --checkpoint ./checkpoints/IFRNet/TR-IFRNet --save_dir ./results/I0_results_TR-IFRNet --num 1 1 1 --gif
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model IFRNet --variant DR --checkpoint ./checkpoints/IFRNet/DR-IFRNet --save_dir ./results/I0_results_DR-IFRNet --num 1 1 1 --gif
    
    EMA-VFI:
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model EMA-VFI --variant T --checkpoint ./checkpoints/EMA-VFI/T-EMA-VFI --save_dir ./results/I0_results_T-EMA-VFI --num 1 1 1 --gif
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model EMA-VFI --variant D --checkpoint ./checkpoints/EMA-VFI/D-EMA-VFI --save_dir ./results/I0_results_D-EMA-VFI --num 1 1 1 --gif
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model EMA-VFI --variant TR --checkpoint ./checkpoints/EMA-VFI/TR-EMA-VFI --save_dir ./results/I0_results_TR-EMA-VFI --num 1 1 1 --gif
    python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model EMA-VFI --variant DR --checkpoint ./checkpoints/EMA-VFI/DR-EMA-VFI --save_dir ./results/I0_results_DR-EMA-VFI --num 1 1 1 --gif
    """
    parser = ArgumentParser()
    parser.add_argument('--img0', type=str, default='./demo/I0_0.png', help='path of the starting image')
    parser.add_argument('--img1', type=str, default='./demo/I0_1.png', help='path of the ending image')
    parser.add_argument('--model', type=str, default='RIFE', help='[ RIFE | IFRNet | AMT-S | EMA-VFI ]')
    parser.add_argument('--variant', type=str, default='DR', help='[ T | D | TR | DR ]')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/RIFE/DR-RIFE-pro/', help='path of checkpoint')
    parser.add_argument('--save_dir', type=str, default='./results/I0_results_DR-RIFE-pro/', help='where to save image results')
    parser.add_argument('--num', type=int, nargs='+', default=[1, 1, 1, 1, 1, 1, 1], help='number of extracted images')
    parser.add_argument('--iters', type=int, default=2, help='number of iterations')
    parser.add_argument('--gif', action='store_true', help='whether to generate the corresponding gif')
    args = parser.parse_args()

    prefix_path = '../..'
    args.img0 = osp.join(prefix_path, args.img0)
    args.img1 = osp.join(prefix_path, args.img1)
    args.checkpoint = osp.join(prefix_path, args.checkpoint)
    args.save_dir = osp.join(prefix_path, args.save_dir)

    if args.model == 'RIFE':
        os.chdir('models/DI-RIFE')
        if args.variant == 'T':
            args.checkpoint = osp.join(args.checkpoint, 'train_m_log')
            cmd = 'python inference_img_plus.py --img0 {} --img1 {} --save_dir {} --checkpoint {} --num {}'.format(
                args.img0, args.img1, args.save_dir, args.checkpoint, ' '.join([str(x) for x in args.num])
            )
        if args.variant == 'D':
            args.checkpoint = osp.join(args.checkpoint, 'train_sdi_log')
            cmd = 'python inference_img_plus_sdi.py --img0 {} --img1 {} --save_dir {} --checkpoint {} --num {}'.format(
                args.img0, args.img1, args.save_dir, args.checkpoint, ' '.join([str(x) for x in args.num])
            )
        if args.variant == 'TR' or args.variant == 'DR':
            args.checkpoint = osp.join(args.checkpoint, 'train_sdi_log')
            cmd = 'python inference_img_plus_sdi_recur.py --img0 {} --img1 {} --save_dir {} --checkpoint {} --num {} --iters {}'.format(
                args.img0, args.img1, args.save_dir, args.checkpoint, ' '.join([str(x) for x in args.num]), args.iters
            )
        if args.gif:
            cmd += ' --gif'
        os.system(cmd)
    if args.model == 'IFRNet' or args.model == 'AMT-S':
        os.chdir('models/DI-AMT-and-IFRNet')
        cfg_name = osp.basename(glob(osp.join(args.checkpoint, '*.yaml'))[0])
        if args.variant == 'T':
            cmd = 'python inference_img_plus.py --img0 {} --img1 {} --save_dir {} --checkpoint {} --cfg_name {} --num {}'.format(
                args.img0, args.img1, args.save_dir, args.checkpoint, cfg_name, ' '.join([str(x) for x in args.num])
            )
        if args.variant == 'D':
            cmd = 'python inference_img_plus_sdi.py --img0 {} --img1 {} --save_dir {} --checkpoint {} --cfg_name {} --num {}'.format(
                args.img0, args.img1, args.save_dir, args.checkpoint, cfg_name, ' '.join([str(x) for x in args.num])
            )
        if args.variant == 'TR' or args.variant == 'DR':
            cmd = 'python inference_img_plus_sdi_recur.py --img0 {} --img1 {} --save_dir {} --checkpoint {} --cfg_name {} --num {} --iters {}'.format(
                args.img0, args.img1, args.save_dir, args.checkpoint, cfg_name, ' '.join([str(x) for x in args.num]),
                args.iters
            )
        if args.gif:
            cmd += ' --gif'
        os.system(cmd)
    if args.model == 'EMA-VFI':
        os.chdir('models/DI-EMA-VFI')
        args.checkpoint = osp.join(args.checkpoint, 'train_sdi_log')
        if args.variant == 'T':
            cmd = 'python inference_img_plus.py --img0 {} --img1 {} --save_dir {} --checkpoint {} --num {}'.format(
                args.img0, args.img1, args.save_dir, args.checkpoint, ' '.join([str(x) for x in args.num])
            )
        if args.variant == 'D':
            cmd = 'python inference_img_plus_sdi.py --img0 {} --img1 {} --save_dir {} --checkpoint {} --num {}'.format(
                args.img0, args.img1, args.save_dir, args.checkpoint, ' '.join([str(x) for x in args.num])
            )
        if args.variant == 'TR' or args.variant == 'DR':
            cmd = 'python inference_img_plus_sdi_recur.py --img0 {} --img1 {} --save_dir {} --checkpoint {} --num {} --iters {}'.format(
                args.img0, args.img1, args.save_dir, args.checkpoint, ' '.join([str(x) for x in args.num]),
                args.iters
            )
        if args.gif:
            cmd += ' --gif'
        os.system(cmd)
