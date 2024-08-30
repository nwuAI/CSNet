# CSNet

The official pytorch implementation of our paper CSNet: Combining Global and Local Feature Fusion with Space Generation for few-shot image generation.


>  CSNet: Combining Global and Local Feature Fusion with Space Generation for few-shot image generation
> 
> Zhan Li, etc.
> 


## Prerequisites
- Pytorch 1.5

## Preparing Dataset
Download the [datasets](https://drive.google.com/drive/folders/1nGIqXPEjyhZjIsgiP_-Rb5t6Ji8RdiCA?usp=sharing) and unzip them in `datasets` folder.

## Training
```shell
python train.py --conf configs/flower_selfunet.yaml \
--output_dir results/flower_selfunet \
--gpu 0
```

* You may also customize the parameters in `configs`.
* It takes about 30 hours to train the network on a RTX A6000 GPU.


## Testing
```shell
python test.py --name results/flower_selfunet --gpu 0
```

The generated images will be saved in `results/flower_csnet/test`.


## Evaluation
```shell
python main_metric.py --gpu 0 --dataset flower \
--name results/flower_selfunet \
--real_dir datasets/for_fid/flower --ckpt gen_00100000.pt \
--fake_dir test_for_fid
```

## Citation
If you use this code for your research, please cite our paper.

    @inproceedings{lizhan2023,
    title={CSNet: Combining Global and Local Feature Fusion with Space Generation for few-shot image generation},
    author={Zhan, etc.},
    booktitle={},
    pages={},
    year={2023}
    }



