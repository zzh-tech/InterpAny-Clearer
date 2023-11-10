import argparse
import subprocess
import time
import os
import os.path as osp

if __name__ == '__main__':
    """
    cmd:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python multiprocess_create_dis_index.py --num_gpus 4 --num_workers 5
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 screen python multiprocess_create_dis_index.py --num_gpus 7 --num_workers 4 --path /mnt/disks/ssd0/dataset/gopro/ --sample_list_path train.txt --downsample_ratio 4. --sample_length 9
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 screen python multiprocess_create_dis_index.py --num_gpus 7 --num_workers 4 --path /mnt/disks/ssd0/dataset/gopro/ --sample_list_path test.txt --downsample_ratio 4. --sample_length 9
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 screen python multiprocess_create_dis_index.py --num_gpus 7 --num_workers 4 --path /mnt/disks/ssd0/dataset/dvd/ --sample_list_path train.txt --downsample_ratio 4. --sample_length 9
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 screen python multiprocess_create_dis_index.py --num_gpus 8 --num_workers 4 --path ./dataset/vimeo_triplet/ --sample_list_path tri_trainlist.txt --sample_length 3
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 screen python multiprocess_create_dis_index.py --num_gpus 8 --num_workers 4 --path ./dataset/vimeo_triplet/ --sample_list_path tri_testlist.txt --sample_length 3
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--path', type=str, default='/mnt/disks/ssd0/dataset/vimeo_septuplet/')
    # argument for the path of sample list
    parser.add_argument('--sample_list_path', nargs='+', type=str, default=['sep_trainlist.txt', 'sep_testlist.txt'])
    parser.add_argument('--downsample_ratio', type=float, default=2.)
    parser.add_argument('--sample_length', type=int, default=7)
    args = parser.parse_args()

    path = args.path
    sample_paths = []
    for set_name in args.sample_list_path:
        sep_list = osp.join(path, set_name)
        with open(sep_list) as f:
            samples = f.readlines()
        for sample in samples:
            if '/' not in sample:
                continue
            sample_path = osp.join(path, 'sequences', sample.strip())
            sample_paths.append(sample_path)

    # split sample_paths
    num_processes = args.num_gpus * args.num_workers
    for i in range(num_processes):
        gpu_sample_paths = sample_paths[i::num_processes]
        with open(f'sample_paths_{i}.txt', 'w') as f:
            f.write('\n'.join(gpu_sample_paths))

    # launch multiprocess for masks generation
    pool = []
    for i in range(num_processes):
        cmd = ['python', 'process_create_dis_index.py', '--sample_list_path', f'sample_paths_{i}.txt',
               '--downsample_ratio', str(args.downsample_ratio), '--sample_length', str(args.sample_length)]
        print(' '.join(cmd))
        env = {
            **os.environ,
            'CUDA_VISIBLE_DEVICES': str(i // args.num_workers)
        }
        p = subprocess.Popen(cmd, env=env)
        pool.append(p)
    exit_codes = [p.wait() for p in pool]

    # clean split result
    for i in range(num_processes):
        os.remove(f'sample_paths_{i}.txt')
