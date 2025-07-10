# logger & progress bar

from __future__ import absolute_import
import os
import sys
import time
import torch.nn as nn
import torch.nn.init as init

# __all__ = ['Logger', "progress_bar"]

'''
将训练过程中的所有metric记录进一个txt文件
usage
# logger = Logger('pid.txt', title='mnist')
# logger.set_names(['LearningRate', 'TrainLoss', 'ValidLoss', 'TrainAcc', 'ValidAcc'])
# logger.append(['htfh', 1, 2, 3, 4])
'''

class Logger(object):
    '''Save training process to log file.'''
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title is None else title
        if fpath is not None:
            if resume:                                                         #中断后重新开始，重新导入相关数据
                self.file = open(fpath, 'r')                                   #打开文件，只读
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')                     #删除 string 字符串末尾的指定字符（默认为空格）
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')                                   #打开文件，附加写
            else:    # build a file
                self.file = open(fpath, 'w')                                   #新建文件，只写（覆盖原文件）

    def set_names(self, names):    # names for every line
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}                                                      #生成key为所有name的字典
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')#一个空格
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()                                                      #刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            if index == 0:
                 self.file.write(num)
                 self.file.write('\t')
            else:
                self.file.write("{0:6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)                        #字典对应key更新value列表
        self.file.write('\n')
        self.file.flush()

    def write(self,content):
        self.file.write(content)
        self.file.write('\n')

    def close(self):
        if self.file is not None:
            self.file.close()



