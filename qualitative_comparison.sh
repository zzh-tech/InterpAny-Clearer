SAMPLES=(00068/0261 00080/0050 00037/0345 00006/0277 00029/0419 00036/0310 00036/0320 00071/0366 00071/0808)
for SAMPLE in ${SAMPLES[*]}; do
    CUDA_VISIBLE_DEVICES=0 python inference_img.py --img0 ./dataset/vimeo_septuplet/sequences/${SAMPLE}/im1.png --img1 ./dataset/vimeo_septuplet/sequences/${SAMPLE}/im7.png --model RIFE --variant T --save_dir ./results/webapp_demo/${SAMPLE}_RIFE_T/ --gif --num 1 1 1 1 1 1 1 --checkpoint ./checkpoints/RIFE/T-RIFE
    CUDA_VISIBLE_DEVICES=0 python inference_img.py --img0 ./dataset/vimeo_septuplet/sequences/${SAMPLE}/im1.png --img1 ./dataset/vimeo_septuplet/sequences/${SAMPLE}/im7.png --model RIFE --variant DR --save_dir ./results/webapp_demo/${SAMPLE}_RIFE_DR/ --gif --num 1 1 1 1 1 1 1 --checkpoint ./checkpoints/RIFE/DR-RIFE --iters 2
done