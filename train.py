import os
import random
import shutil
import sys
import argparse
import torch
import time
import numpy as np
from tensorboardX import SummaryWriter

from trainer import Trainer
from utils import make_result_folders, write_image, write_loss, get_config, get_loaders

#解析命令行代码
parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str)
parser.add_argument('--output_dir', type=str,)
parser.add_argument('-r', "--resume", action="store_true")
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

#这里类似字典的键值对形式
config = get_config(args.conf)
#指定所要使用的显卡即主卡
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

#输出路径
output_directory = args.output_dir

remove_first = not args.resume

#checkpoint即实验中或停止时，保留此时模型的参数。以便下次开始时不需要重新训练。
#checkpoint路径、图片路径、日志路径
#创建3个目录，并返回
checkpoint_directory, image_directory, log_directory = make_result_folders(output_directory, remove_first=remove_first)
#复制文件  当configs.yaml是文件夹时，即将args.conf复制到该文件夹中，是文件时即替换之中内容，若无则创建并粘贴内容
shutil.copy(args.conf, os.path.join(output_directory, 'configs.yaml'))
#创建可视化事件
train_writer = SummaryWriter(log_directory)
#最大迭代数
max_iter = config['max_iter']
#批处理数据集
train_dataloader, test_dataloader = get_loaders(config)

if __name__ == '__main__':
    #随机种子
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    #开始时间；为了计时
    start = time.time()
    #将网络的各个参数初始化
    trainer = Trainer(config)
    #trainer.load_ckpt(checkpoint_directory)
    #将网络的各个参数传至GPU
    trainer.cuda()
    #变成迭代器
    imgs_test, _ = iter(test_dataloader).next()
    #调回至检查点时网络的状态
    iterations = trainer.resume(checkpoint_directory) if args.resume else 0
    while True:
        with torch.autograd.set_detect_anomaly(True):

            for it, (imgs, label) in enumerate(train_dataloader):
                #更新学习率
                trainer.update_lr(iterations, max_iter)
                #调用GPU
                imgs = imgs.cuda()
                label = label.cuda()
                #梯度清零并更新参数
                trainer.dis_update(imgs, label)
                trainer.gen_update(imgs, label)
                #输出训练时间并输出损失
                if (iterations + 1) % config['snapshot_log_iter'] == 0:
                    end = time.time()
                    print("Iteration: [%06d/%06d], time: %d, loss_adv_dis: %04f, loss_adv_gen: %04f"
                          % (iterations + 1, max_iter, end-start, trainer.loss_adv_dis, trainer.loss_adv_gen))
                    #write_loss(iterations, trainer)
                    write_loss(iterations, trainer, train_writer)
                #每搁多少次保存一次图片
                if (iterations + 1) % config['snapshot_val_iter'] == 0:
                    with torch.no_grad():
                        imgs_test = imgs_test.cuda()
                        fake_xs = []
                        for i in range(config['num_generate']):
                            fake_xs.append(trainer.generate(imgs_test).unsqueeze(1))
                        fake_xs = torch.cat(fake_xs, dim=1)
                        write_image(iterations, image_directory, imgs_test.detach(), fake_xs.detach())
                #每训练多少次保存一次检查点
                if (iterations + 1) % config['snapshot_save_iter'] == 0:
                    trainer.save(checkpoint_directory, iterations)
                    print('Saved model at iteration %d' % (iterations + 1))

                iterations += 1
                if iterations >= max_iter:
                    print("Finish Training")
                    sys.exit(0)

