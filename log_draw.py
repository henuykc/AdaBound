# -*- coding: utf-8 -*-

#    --log_dir为日志存放地址，输入格式为--log_dir=C:\Users\DELL\Desktop\logs
#    --tools为所用框架名称，输入格式为--tools=tensorflow,pytorch,mxnet'
#    --models为每个框架下模型的名称，输入格式为--models=alexnet,resnet50,vgg16,inception3
#    --data_type为训练所用的数据类型(real代表真实数据，synt代表合成数据)，输入格式为--data_type=real,synt

#    当可以运行两机八卡时，请将代码被注释的部分取消注释
#    不同模型的日志提取只需要在read_one_file与throughput_average中相应特性即可





import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# 设置中文字体和负号正常显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

GPUs = [1, 4]


def parse_arguements(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, help='', default='./log')
    parser.add_argument('--output_dir', type=str, help='', default='./output')
    parser.add_argument('--model_type',type=str,help='',default='mnist')
    parser.add_argument('--model_name', type=str, help='', default='adabound,adaboundW,myAdaboundW')
    return parser.parse_args(argv)


def read_one_file(model_type,filename):
    #此处根据日志特点进行提取适配
    if model_type == 'mnist' or model_type == 'cifar10' or model_type == 'ptb' :
        job_str = open(filename).read()
        lines = job_str.splitlines()
        epochs = []
        train_accs = []
        test_accs = []
        for l in lines:
            if r"epoch:" in l:
                epoch = l.split(':')[-1].strip()
                epochs.append(int(epoch)+1)
            if r"train acc" in l:
                train_acc = l.split(' ')[-1].strip()
                train_accs.append(float(train_acc))
            if r" test acc" in l:
                test_acc = l.split(' ')[-1].strip()
                test_accs.append(float(test_acc))

        return epochs,train_accs,test_accs

def curve(args,model_type,logdir):
    models = args.model_name.split(',')
    txt_path = os.path.join(logdir, model_type)
    length=len(os.listdir(txt_path))
    plt.figure()
    train_fig=plt.subplot(121)
    test_fig = plt.subplot(122)
    for i in range(length):
        filename=os.path.join(txt_path,models[i]+'.txt')
        epochs,train_accs,test_accs=read_one_file(model_type,filename)
        train_fig.plot(epochs,train_accs,label=models[i])
        test_fig.plot(epochs,test_accs,label=models[i])

    plt.ylim(95, 110)
    plt.ylabel("Train accuracy")
    plt.ylim(80, 100)
    plt.ylabel("Test accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

def start_draw(args):
    model_types=args.model_type.split(',')
    log_dir=args.log_dir
    for model_type in model_types:
        curve(args,model_type,log_dir)


def draw_bound():
    x=np.arange(0,201,1)
    y1=0.1-(0.1/((0.01)*x+1))
    y2 = 0.1 + 0.1 / ((0.01) * x )
    plt.plot(x,y1,label='low')
    plt.plot(x, y2, label='up')
    plt.legend()
    plt.show()

def main(args):
    start_draw(args)


if __name__ == '__main__':
    main(parse_arguements(sys.argv[1:]))