# InterpAny-Clearer

#### :rocket: <u style="color: hotpink; text-decoration: underline dotted hotpink;">[ECCV2024 Oral] Clearer Frames,</u> <u style="color: dodgerblue; text-decoration: underline dotted dodgerblue;">Anytime</u>: Resolving Velocity Ambiguity in Video Frame Interpolation

by [Zhihang Zhong](https://zzh-tech.github.io/)<sup>
1,*</sup>, [Gurunandan Krishnan](https://scholar.google.com/citations?user=BKYVv4MAAAAJ&hl=en)<sup>
2</sup>, [Xiao Sun](https://jimmysuen.github.io/)<sup>
1</sup>, [Yu Qiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl=en)<sup>
1</sup>, [Sizhuo Ma](https://sizhuoma.netlify.app/)<sup>2,†</sup>, and [Jian Wang](https://jianwang-cmu.github.io/)<sup>
2,†</sup>

<sup>*</sup>First author, <sup>†</sup>Co-corresponding authors

<sup>1</sup>[Shanghai AI Laboratory, OpenGVLab](https://github.com/OpenGVLab), <sup>
2</sup>[Snap Inc.](https://snap.com/en-US)

<br>
We strongly recommend referring to the project page and interactive demo for a better understanding:

:point_right: [**project page**](https://zzh-tech.github.io/InterpAny-Clearer/)  
:point_right: [**interactive demo**](http://ai4sports.opengvlab.com/interpany-clearer/)  
:point_right: [**OpenXLab demo**](https://openxlab.org.cn/apps/detail/ZhihangZhong/InterpAny-Clearer)  
:point_right: [arXiv](http://arxiv.org/abs/2311.08007)  
:point_right: [slides](https://docs.google.com/presentation/d/1_aIkH_iZUZ2sdSRO9eict1HNAJbX-vQs/edit?usp=sharing&ouid=116575787119851482947&rtpof=true&sd=true)

Please leave a 🌟 if you like this project! 🔥🔥🔥

#### News
- :tada: **2024-08-12**: Luckily, this work is recognized as **Oral** by ECCV2024! 🏁
- :tada: **2024-07-01**: This work is accepted to ECCV2024! 🎆
- :tada: **2024-05-02**: Our technology is used by [CCTV5 and CCTV5+](./demo/cctv5_interpany-clearer.mp4) for slow motion demonstrations of athletes jumping in the 2024 Thomas & Uber Cup! 🔥
- :tada: **2023-11-28**: We have added an interface for video inference to
  the [interactive demo](http://ai4sports.opengvlab.com/interpany-clearer/), and
  uploaded [checkpoints](https://drive.google.com/drive/folders/1zCyySQT7Or9P2Q2qOhG116RRdcaDsjr5?usp=sharing) trained
  with the LPIPS loss.

#### Application in CCTV

https://github.com/zzh-tech/InterpAny-Clearer/assets/68437458/a4b0bc95-d051-45ac-aaf1-400266a290d2

#### TL;DR:

We addressed velocity ambiguity in video frame interpolation through innovative distance indexing and iterative
reference-based
estimation strategies, resulting in:  
<b style="color: orangered">Clearer anytime frame interpolation</b> & <b style="color: orangered">Manipulated
interpolation of anything</b>

<img src="./demo/teaser.jpg">

#### Time indexing vs. Distance indexing

Comparison of x128 interpolation using only 2 frames as inputs:

<table style="width: 1200px">
  <tr>
    <td align="center" style="font-size:18px; border: none;">[T] RIFE</td>
    <td align="center" style="font-size:18px; border: none;">[D,R] RIFE (Ours)</td>
    <td align="center" style="font-size:18px; border: none;">[D,R] RIFE-vgg (Ours)</td>
  </tr>
  <tr>
    <td valign="top" style="border: none;"><img src="demo/T-RIFE_0.gif"></td>
    <td valign="top" style="border: none;"><img src="demo/DR-RIFE_0.gif"></td>
    <td valign="top" style="border: none;"><img src="demo/DR-RIFE-vgg_0.gif"></td>
  </tr>
  <tr>
    <td valign="top" style="border: none;"><img src="demo/T-RIFE_1.gif"></td>
    <td valign="top" style="border: none;"><img src="demo/DR-RIFE_1.gif"></td>
    <td valign="top" style="border: none;"><img src="demo/DR-RIFE-vgg_1.gif"></td>
  </tr>
</table>

[D]: distance indexing   
[R]: iterative reference-based estimation   

The results of [D,R] RIFE-vgg are perceptually clearer, but may suffer from undesirable distortions (see second row). We
recommend using [D,R] RIFE for more stable results.

## Preparation

### Environment installation:

You can try Anaconda or Docker to setup the environment.

#### Anaconda

```shell
conda create -n InterpAny python=3.8
conda activate InterpAny
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

#### Docker

You can build a docker image with all dependencies installed.  
See [docker/README.md](./docker/README.md) for more details.

### Download checkpoints
We provide checkpoints for four different models, including RIFE, IFRNet, AMT-S, and EMA-VFI.

Download checkpoints
from [here (full)](https://drive.google.com/file/d/14GJSqsX4H5EcQjd-tLb5CM_jzD-577bl/view?usp=sharing) / [here (separate)](https://drive.google.com/drive/folders/11MY60fpDk5oAlGasQRZ3ss3xVdrOBIiE?usp=sharing).

*P.S., RIFE-pro denotes the RIFE model trained with more data and epochs; RIFE-vgg denotes the RIFE model trained with
the LPIPS loss.*

## Inference

### Two images

```shell
python inference_img.py --img0 [IMG_0] --img1 [IMG_1] --output_dir [OUTPUT_DIR] --model [MODEL_NAME] --variant [VARIANT] --num [NUM] --gif
```

Examples:

`python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model RIFE --variant DR --checkpoint ./checkpoints/RIFE/DR-RIFE-pro --save_dir ./results/I0_results_DR-RIFE-pro --num 1 1 1 1 1 1 1 --gif`

`python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model RIFE --variant DR --checkpoint ./checkpoints/RIFE/DR-RIFE-vgg --save_dir ./results/I0_results_DR-RIFE-vgg --num 1 1 1 1 1 1 1 --gif`

`python inference_img.py --img0 ./demo/I0_0.png --img1 ./demo/I0_1.png --model EMA-VFI --variant DR --checkpoint ./checkpoints/EMA-VFI/DR-EMA-VFI --save_dir ./results/I0_results_DR-EMA-VFI/ --num 1 1 1 1 1 1 1 --gif`

`--num NUM` means to interpolate `NUM` frames between every two frames.   
`--num NUM1 NUM2 ...` means that `NUM1` frames are interpolated between every two frames, then `NUM2` frames are interpolated between every two frames for the result of the interpolation, and so on.

### Video

```shell
python inference_video.py --video [VIDEO] --output_dir [OUTPUT_DIR] --model [MODEL_NAME] --variant [VARIANT] --num [NUM]
```

Examples:

`python inference_video.py --video ./demo/demo.mp4 --model RIFE --variant DR --checkpoint ./checkpoints/RIFE/DR-RIFE-pro --save_dir ./results/demo_results_DR-RIFE-pro --num 3 --fps 15`

P.S., if without `--fps`, the output video will have the same fps as the input video.

## Manipulation

### Manipulated interpolation of anything

<img src="./demo/manipulation.jpg"/>

### Demos

<table>
  <tr>
    <td><img src="./demo/manipulation1.gif"></td>
    <td><img src="./demo/manipulation2.gif"></td>
    <td><img src="./demo/manipulation3.gif"></td>
  </tr>
</table>

### Webapp

You can play with the [interactive demo](http://ai4sports.opengvlab.com/interpany-clearer/) or install the webapp
locally.

#### Install the webapp locally

P.S., not required if you use docker

Follow [./webapp/backend/README.md](./webapp/backend/README.md) to setup the environment for Segment Anything.  
Follow [./webapp/webapp/README.md](./webapp/webapp/README.md) to setup the environment for the webapp.

#### Run the app

```shell
cd ./webapp/backend/
python app.py

# open a new terminal
cd ./webapp/webapp/
yarn && yarn start
```

## Dataset

You can download the splited Vimeo90K dataset with our distance indexing maps
from [here](https://drive.google.com/drive/folders/1mYPjleTX3P069hghOad3plGDrUp4d7xJ?usp=sharing) (
or [full dataet](https://drive.google.com/file/d/1qImY1rLNIcgOu4sX6cQi02br-p-HNs9H/view?usp=sharing)), and then merge
them:

```shell
cat vimeo_septuplet_split.zipa* > vimeo_septuplet_split.zip
```

Alternatively, you can download original Vimeo90K dataset from [here](http://toflow.csail.mit.edu/), and then generate
distance indexing (P.S.
Download [checkpoints](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing) for RAFT
and put them under `./RAFT/models/` in advance):

```shell
python multiprocess_create_dis_index.py
```

## Train

Training command:

```shell
python train.py --model [MODEL_NAME] --variant [VARIANT]
```

Examples:

`python train.py --model RIFE --variant D`

`python train.py --model RIFE --variant DR`

`python train.py --model AMT-S --variant D`

`python train.py --model AMT-S --variant DR`

## Test

Testing with precomputed distance maps:

```shell
python test.py --model [MODEL_NAME] --variant [VARIANT]
```

Examples:

`python test.py --model RIFE --variant D`

`python test.py --model RIFE --variant DR`

Testing using uniform distance maps with the same inputs as the time indexes:

```shell
python test.py --model [MODEL_NAME] --variant [VARIANT] --uniform
```

Examples:

`python test.py --model RIFE --variant D --uniform`

`python test.py --model RIFE --variant DR --uniform`

## Citation

If you find this repository useful, please consider citing:

```bibtex
@article{zhong2023clearer,
  title={Clearer Frames, Anytime: Resolving Velocity Ambiguity in Video Frame Interpolation},
  author={Zhong, Zhihang and Krishnan, Gurunandan and Sun, Xiao and Qiao, Yu and Ma, Sizhuo and Wang, Jian},
  journal={arXiv preprint arXiv:2311.08007},
  year={2023}
}
```

## Acknowledgements

We thank Dorian Chan, Zhirong Wu, and Stephen Lin for their insightful feedback and advice. Our thanks also go to Vu An
Tran for developing the web application, and to Wei Wang for coordinating the user study.

Moreover, we appreciate the following projects for releasing their code:

[[CVPR 2018] The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://github.com/richzhang/PerceptualSimilarity)  
[[ECCV 2020] RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://github.com/princeton-vl/RAFT)  
[[ECCV 2022] Real-Time Intermediate Flow Estimation for Video Frame Interpolation](https://github.com/megvii-research/ECCV2022-RIFE)  
[[CVPR 2022] IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation](https://github.com/ltkong218/IFRNet)  
[[CVPR 2023] AMT: All-Pairs Multi-Field Transforms for Efficient Frame Interpolation](https://github.com/MCG-NKU/AMT)  
[[CVPR 2023] Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation](https://github.com/ltkong218/IFRNet)  
[[ICCV 2023] Segment Anything](https://github.com/facebookresearch/segment-anything)  
