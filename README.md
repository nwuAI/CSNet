# LoFGAN-pytorch

The official pytorch implementation of our paper SelfNet: Self-attention and Convolution Fusion Network for Few-shot Image Generation.


![framework](images/framework_all.png)

>  SelfNet: Self-attention and Convolution Fusion Network for Few-shot Image Generationn
> 
> Zhan Li, Shijie Shi, Chuan Peng, Yuning Wang, Lei Wang, JiaBao Shi
> 


## Prerequisites
- Pytorch 1.5

## Preparing Dataset
Download the [datasets](https://drive.google.com/drive/folders/1nGIqXPEjyhZjIsgiP_-Rb5t6Ji8RdiCA?usp=sharing) and unzip them in `datasets` folder.

## Training
```shell
python train.py --conf configs/flower_lofgan.yaml \
--output_dir results/flower_lofgan \
--gpu 0
```

* You may also customize the parameters in `configs`.
* It takes about 30 hours to train the network on a V100 GPU.


## Testing
```shell
python test.py --name results/flower_lofgan --gpu 0
```

The generated images will be saved in `results/flower_lofgan/test`.


## Evaluation
```shell
python main_metric.py --gpu 0 --dataset flower \
--name results/flower_lofgan \
--real_dir datasets/for_fid/flower --ckpt gen_00100000.pt \
--fake_dir test_for_fid
```

## Citation
If you use this code for your research, please cite our paper.

    @inproceedings{gu2021lofgan,
    title={LoFGAN: Fusing Local Representations for Few-Shot Image Generation},
    author={Gu, Zheng and Li, Wenbin and Huo, Jing and Wang, Lei and Gao, Yang},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={8463--8471},
    year={2021}
    }


## Acknowledgement
Our code is designed based on [FUNIT](https://github.com/NVlabs/FUNIT).

The code for calculate FID is based on [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
