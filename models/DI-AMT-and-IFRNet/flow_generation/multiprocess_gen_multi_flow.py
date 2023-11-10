import argparse
import subprocess
import time
import os
import os.path as osp

if __name__ == '__main__':
    """
    cmd:
    sudo chmod -R +w /mnt/disks/ssd0/dataset/vimeo_septuplet/flow/
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 screen python flow_generation/multiprocess_gen_multi_flow.py --num_gpus 8 --workers_per_gpu 8
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--workers_per_gpu', type=int, default=4)
    args = parser.parse_args()

    # split sample_paths
    num_processes = args.num_gpus * args.workers_per_gpu

    # launch multiprocess for masks generation
    pool = []
    for i in range(num_processes):
        cmd = ['python', 'flow_generation/gen_multi_flow.py',
               '--root=/mnt/disks/ssd0/dataset/vimeo_septuplet',
               '--worker_id={}'.format(i), '--num_workers={}'.format(num_processes)]
        env = {
            **os.environ,
            'CUDA_VISIBLE_DEVICES': str(i // args.workers_per_gpu)
        }
        p = subprocess.Popen(cmd, env=env)
        pool.append(p)
    exit_codes = [p.wait() for p in pool]
