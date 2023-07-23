import argparse
import os
import yaml
import shutil
import torch
import random
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from dataset import FSGDataset


def make_result_folders(output_directory, remove_first=True):
    if remove_first:
        if os.path.exists(output_directory):
            #shuti.rmtree为递归地删除路径
            '''
            mkdir -p foo/bar
            python
            import shutil
            shutil.rmtree('foo/bar')即只删除bar
            '''
            shutil.rmtree(output_directory)
    #在输出路径下 增添images子目录
    image_directory = os.path.join(output_directory, 'images')
    #如果不存在该目录，便创建该目录
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    #同上
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    #同上
    log_directory = os.path.join(output_directory, 'logs')
    if not os.path.exists(log_directory):
        print("Creating directory: {}".format(log_directory))
        os.makedirs(log_directory)
    return checkpoint_directory, image_directory, log_directory


#def write_loss(iterations, trainer):
def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if ((not callable(getattr(trainer, attr))
                    and not attr.startswith("__"))
                   and ('loss' in attr
                        or 'grad' in attr
                        or 'nwd' in attr
                        or 'accuracy' in attr
                        or 'obsv' in attr))]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def write_image(iterations, dir, im_ins, im_outs, format='jpeg'):
    B, K1, C, H, W = im_ins.size()
    B, K2, C, H, W = im_outs.size()
    file_name = os.path.join(dir, '%08d' % (iterations + 1) + '.' + format)
    image_tensor = torch.cat([im_ins, im_outs], dim=1)
    image_tensor = image_tensor.view(B*(K1+K2), C, H, W)
    image_grid = vutils.make_grid(image_tensor.data, nrow=K1+K2, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1, format=format)


def get_config(config):
    #读取配置文件数据
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def get_loader(dataset, root, mode, n_sample, num_for_seen, batch_size, num_workers, shuffle, drop_last,
               new_size=None, height=28, width=28, crop=False, center_crop=False):

    assert dataset in ['flower', 'vggface', 'animal']

    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if center_crop:
        transform_list = [transforms.CenterCrop((height, width))] + \
                         transform_list if crop else transform_list
    else:
        transform_list = [transforms.RandomCrop((height, width))] + \
                         transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list \
        if new_size is not None else transform_list

    transform = transforms.Compose(transform_list)
    dataset = FSGDataset(root, mode, num_for_seen, n_sample, transform)

    #drop_last=True表示最后一个batch不足batch_size时，不报错，而是返回空
    loader = DataLoader(dataset,
                        batch_size,
                        shuffle=shuffle,
                        drop_last=drop_last,
                        num_workers=num_workers)
    return loader


def get_loaders(conf):
    dataset = conf['dataset']#数据集
    root = conf['data_root']#数据集路径
    batch_size = conf['batch_size']#批大小
    num_workers = conf['num_workers']#线程数
    train_loader = get_loader(
        dataset=dataset,
        root=root,
        mode='train',
        n_sample=conf['n_sample_train'],
        #119个可见类
        num_for_seen=conf['dis']['num_classes'],
        #8
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True
    )
    test_loader = get_loader(
        dataset=dataset,
        root=root,
        mode='test',
        n_sample=conf['n_sample_test'],
        num_for_seen=conf['dis']['num_classes'],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False
    )
    return train_loader, test_loader


def unloader(img):
    img = (img + 1) / 2
    tf = transforms.Compose([
        transforms.ToPILImage()
    ])
    return tf(img)


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and
                  key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def batched_index_select(input, dim, index):
    #32 128 64 dim=2 index等于随机的32个数
    #[32,1,-1]
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    #[32,128,64]
    expanse = list(input.shape)
    #[-1,128,64]
    expanse[0] = -1
    #[-1,128,-1]
    expanse[dim] = -1
    #32 32 ->32 1 32
    index = index.view(views)
    #32 1 32 ->32 128 32
    index = index.expand(expanse)
    ''' out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2'''
    return torch.gather(input, dim, index)


def batched_scatter(input, dim, index, src):
    #[32 1 -1]
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views)
    index = index.expand(expanse)
    #将src插入进input中
    return torch.scatter(input, dim, index, src)


def cal_para(model):
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params: %.3fM' % (trainable_num / 1e6))


class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')