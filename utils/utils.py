# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import csv
import cv2
import imgaug as ia
import random
import scipy.misc as misc
import shutil
from skimage import measure
import math
import traceback
from sklearn import metrics
import zipfile


def setup_seed(seed=1234):
     torch.manual_seed(seed)                                                   #为CPU设置种子用于生成随机数，以使得结果是确定的
     torch.cuda.manual_seed(seed)                                              #为当前GPU设置随机种子
     torch.cuda.manual_seed_all(seed)                                          #torch.cuda.manual_seed_all()为多个GPU设置种子
     np.random.seed(seed)
     random.seed(seed)
     ia.seed(seed)
     
     cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True
     
     
class AverageMeter(object):
    '''计算并存储val平均值和当前值'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state,best_pred,best_pred_Test, epoch,is_best,checkpoint_path,stage="val",filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("Model Saving................")
        shutil.copyfile(filename, osp.join(checkpoint_path,'model_{}_{:03d}_{:.6f}_{:.6f}.pth.tar'.format(
            stage,(epoch + 1),best_pred,best_pred_Test)))

def adjust_learning_rate(opt, optimizer, epoch):                        
    """
    将学习速率设置为初始LR经过每30个epoch衰减10% (step = 30)
    """
    if opt.lr_mode == 'step':
        lr = opt.lr * (0.1 ** (epoch // opt.step))
    elif opt.lr_mode == 'poly':
        lr = opt.lr * (1 - epoch / opt.num_epochs) ** 0.9
    elif opt.lr_mode == 'normal':
        lr = opt.lr
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_mode))

    for param_group in optimizer.param_groups:                                 #可以为一个网络设置多个优化器，每个优化器对应一个字典包括参数组及其对应的学习率,动量等等，optimizer.param_groups是由所有字典组成的列表
        param_group['lr'] = lr                                                 #动态修改学习率
    return lr
        
  
def one_hot_it(label, label_info):
	'''
    return semantic_map -> [c, H, W]
    label_info是类别颜色123列表
    '''
	semantic_map = []
	for info in label_info:
		equality = np.equal(label, info)
		class_map = equality.astype('uint8')
		semantic_map.append(class_map)
	semantic_map = np.stack(semantic_map, axis=0)
	return semantic_map


def reverse_one_hot_it(label,label_info):
    '''
    return [c, H, W] -> semantic_map
    label_info是类别123列表
    '''
    semantic_map = np.argmax(label, axis=0).astype('uint8')
    # semantic_map = torch.argmax(image, dim=0)
    return semantic_map


def target_seg2class(label): 
    '''
    label为torch.tensor，返回class_num_all也为tensor
    '''
    label_arr = label.numpy().astype('uint8')
    class_num_all = torch.zeros(label_arr.shape[0],3)
    for i in range(label_arr.shape[0]):
        array_np = np.unique(label_arr[i])
        for item in array_np[1:]:
            class_num_all[i,item-1] = 1
    return class_num_all.float()


def save_dice_single(is_best, filename='dice_single.txt'):
    '''
    is_best更新，另外保存一份此时的dice_single.txt为dice_best.txt
    '''
    if is_best:
        shutil.copyfile(filename, 'dice_best.txt')


def compute_dice_score(predict, gt, forground = 1):
    '''
    计算二分类dice，返回dice值（加smooth）
    输入原标签0123
    可算slice与cube
    '''
    # score = 0
    # count = 0
    assert(predict.shape == gt.shape)
    overlap = 2.0 * ((predict == forground)*(gt == forground)).sum()
    #print('overlap:',overlap)
    
    return (overlap + 0.001) / (((predict == forground).sum() + (gt == forground).sum()) + 0.001)
  

def compute_average_dice(predict, gt, class_num = 4):
    '''
    计算多分类dice，将无关类置0 分为4次二分类dice计算，返回各类dice及其平均值（加smooth）
    输入原标签0123
    可算slice与cube
    train里用的是这个！！！！！最重要！！！
    '''
    Dice = 0
    Dice_list = []

    for i in range(1,class_num):
        predict_copy = predict.copy()
        gt_copy = gt.copy()
        predict_copy[predict_copy != i] = 0
        gt_copy[gt_copy != i] = 0
        dice = compute_dice_score(predict_copy, gt_copy, forground = i)
        Dice += dice
        Dice_list.append(dice)
    return Dice/(class_num - 1),Dice_list[0],Dice_list[1],Dice_list[2]


def compute_score(predict, gt, forground = 1):
    '''
    计算二分类Dice,Precsion,Recall,Jaccard，返回其值（不加smooth）
    输入原标签0123
    可算slice与cube
    
    overlap=0的情况：
        gt和predict无交集，正常计算各指标为0
        gt无 predict有，recall为Nan，输出0
        gt有 predict无，precsion为Nan,输出0
        gt无 predict无，全部为Nan,输出0
    '''
    # score = 0
    # count = 0
    assert(predict.shape == gt.shape)
    overlap = ((predict == forground)*(gt == forground)).sum()                 #TP
    #print('overlap:',overlap)
    if(overlap > 0):
        return 2*overlap / ((predict == forground).sum() + (gt == forground).sum()),overlap /  (predict == forground).sum(), overlap / (gt == forground).sum(),overlap / ((predict == forground).sum() + (gt == forground).sum() - overlap)
        # dice,precsion,recall,Jaccard
    else:
        return 0,0,0,0


def eval_seg(predict, gt, forground = 1):
    '''
    计算多分类Dice,Precsion,Recall,Jaccard，分为4次二分类计算，背景也包括在内（不加smooth）
    输入one-hot编码
    计算slice，不计算cube
    '''
    assert(predict.shape == gt.shape)
    Dice = 0
    Precsion = 0
    Recall = 0
    Jaccard = 0
    n = predict.shape[0]
    for i in range(n):
        dice,precsion,recall,jaccard = compute_score(predict[i],gt[i])
        Dice += dice
        Precsion += precsion
        Recall += recall
        Jaccard += jaccard

    return Dice/n,Precsion/n,Recall/n,Jaccard/n


def compute_segment_score(ret_segmentation,cubes=5):
    '''
    计算所有cube的平均指标，不计算金标准Nan情况，该类有几个cube分母为几
    ret_segmentation列表长度为cube数，其元素也是列表[cube_dice0,cube_dice1,cube_dice2,cube_dice3]
    最终的is_best评判标准！！！！
    '''
    IRF_segementation, SRF_segementation, PED_segementation = 0.0, 0.0, 0.0
    n1, n2, n3 = 0, 0, 0
    for i in range(cubes):
        if not math.isnan(ret_segmentation[i][1]):
            IRF_segementation += ret_segmentation[i][1]
            n1 += 1
        if not math.isnan(ret_segmentation[i][2]):
            SRF_segementation += ret_segmentation[i][2]
            n2 += 1
        if not math.isnan(ret_segmentation[i][3]):
            PED_segementation += ret_segmentation[i][3]
            n3 += 1

    IRF_segementation /= n1
    SRF_segementation /= n2
    PED_segementation /= n3
    avg_segmentation = (IRF_segementation + SRF_segementation + PED_segementation) / 3

    return avg_segmentation,IRF_segementation,SRF_segementation,PED_segementation


def compute_single_segment_score(ret_segmentation):
    '''
    计算单个cube的平均指标，不计算金标准Nan情况，该cube有几类分母为几
    ret_segmentation是列表[cube_dice0,cube_dice1,cube_dice2,cube_dice3]
    val中用到！！
    '''
    IRF_segementation, SRF_segementation, PED_segementation = 0.0, 0.0, 0.0
    n1, n2, n3 = 0, 0, 0
    
    if not math.isnan(ret_segmentation[1]):
        IRF_segementation += ret_segmentation[1]
        n1 += 1
    if not math.isnan(ret_segmentation[2]):
        SRF_segementation += ret_segmentation[2]
        n2 += 1
    if not math.isnan(ret_segmentation[3]):
        PED_segementation += ret_segmentation[3]
        n3 += 1

    avg_segmentation = (IRF_segementation + SRF_segementation + PED_segementation) / (n1+n2+n3)

    return avg_segmentation


def count_param(model):
    param_count = 0
    for param in model.parameters():                                           #迭代打印model.parameters()将会打印每一次迭代元素的param而不会打印名字（和named_parameters区别），两者都可以用来改变requires_grad的属性
        param_count += param.view(-1).size()[0]                                #.view(-1)表示将张量拉成一维，降维
    return param_count


def zip_dir(dirname,zipfilename):
    '''
    压缩文件
    dirname（文件or文件夹）,zipfilename为str
    '''
    filelist = []
    if osp.isfile(dirname):                                                    #压缩某文件
        filelist.append(dirname)
    else :                                                                     #压缩某文件夹
        for root, dirs, files in os.walk(dirname):                             #os.walk输入一个路径名称，以yield的方式返回一个三元组。dirpath为目录的路径，dirnames列出了目录路径下面所有存在的目录的名称，filenames列出了目录路径下面所有文件的名称。
            for name in files:                                                 #filelist获得一个路径下面所有的文件路径（从dirname开始）
                filelist.append(osp.join(root, name))
        
    zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)              #object
    for tar in filelist:
        zf.write(tar)
    zf.close()
       
    
def aic_fundus_lesion_classification(ground_truth, prediction, num_samples=128):
    """
    Classification task auc metrics.
    :param ground_truth: numpy matrix, (num_samples, 3)
    :param prediction: numpy matrix, (num_samples, 3)
    :param num_samples: int, default 128
    :return list:[AUC_1, AUC_2, AUC_3]
    """
#     assert (ground_truth.shape == (num_samples, 3))
#     assert (prediction.shape == (num_samples, 3))

    try:
        ret = [0.5, 0.5, 0.5]
        for i in range(3):
            #计算ROC曲线（纵坐标tpr、横坐标fpr、选取的阈值threshold），pos_label即正类标签
            fpr, tpr, thresholds = metrics.roc_curve(ground_truth[:, i], prediction[:, i], pos_label=1)
            ret[i] = metrics.auc(fpr, tpr)

        
        # fpr, tpr, thresholds = metrics.roc_curve(ground_truth[:,0], prediction[:,0], pos_label=1)
        # ret = metrics.auc(fpr, tpr)
    except Exception as e:
        traceback.print_exc()                                                  #捕获并打印详细的异常信息
        print("ERROR msg:", e)                                                 #打印异常名包括AttributeError、KeyError、NameError等
        return None
    return ret

def aic_fundus_lesion_segmentation(ground_truth, prediction, num_samples=128):
    """
    Segmentation task dice metrics.
    :param ground_truth: numpy matrix, (num_samples, 1024, 512) 标签为0123
    :param prediction: numpy matrix, (num_samples, 1024, 512)
    :param num_samples: int, default 128
    :return list:[Dice_0, Dice_1, Dice_2, Dice_3]
    """
    #assert (ground_truth.shape == (num_samples, 1024, 512))
    #assert (prediction.shape == (num_samples, 1024, 512))

    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    try:
        ret = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            mask1 = (ground_truth == i)
            mask2 = (prediction == i)
            if mask1.sum() != 0:                                               #只计算金标准存在的类别，不存在的类别返回Nan
                ret[i] = 2 * ((mask1 * (ground_truth == prediction)).sum()) / (mask1.sum() + mask2.sum())
            else:
                ret[i] = float('nan')
    except Exception as e:
        traceback.print_exc()
        print("ERROR msg:", e)
        return None
    return ret

#--------------------------------------------------------------------------------------------------------------------
def eval_multi_seg_2D_all(predict,target,num_classes=4,smooth=0.001):
    pred_seg = predict.data.cpu().numpy().astype(dtype=np.int)
    label_seg = target.data.cpu().numpy().astype(dtype=np.int)
    assert(pred_seg.shape == label_seg.shape)
    
    Dice1 = []
    Dice2 = []
    Dice3 = []
    Acc = []
    
    n = pred_seg.shape[0]
    
    for i in range(n):
        overlap1=((pred_seg[i]==1)*(label_seg[i]==1)).sum()
        union1=(pred_seg[i]==1).sum()+(label_seg[i]==1).sum()
        Dice1.append((2*overlap1+smooth)/(union1+smooth))
        
        overlap2=((pred_seg[i]==2)*(label_seg[i]==2)).sum()
        union2=(pred_seg[i]==2).sum()+(label_seg[i]==2).sum()
        Dice2.append((2*overlap2+smooth)/(union2+smooth))
        
        overlap3=((pred_seg[i]==3)*(label_seg[i]==3)).sum()
        union3=(pred_seg[i]==3).sum()+(label_seg[i]==3).sum()
        Dice3.append((2*overlap3+smooth)/(union3+smooth))
        
        acc=(pred_seg[i]==label_seg[i]).sum()/(pred_seg.shape[1]*pred_seg.shape[2])
        Acc.append(acc)
        
    return Dice1,Dice2,Dice3,Acc


def eval_multi_seg_3D_infer(predict, target,num_classes=6):
    pred_seg = predict.data.cpu().numpy().astype(dtype=np.int)
    label_seg = target.data.cpu().numpy().astype(dtype=np.int)
    assert(pred_seg.shape == label_seg.shape)

    # acc=(pred_seg==label_seg).sum()/(pred_seg.shape[0]*pred_seg.shape[1]*pred_seg.shape[2])




    Dice=[]
    IoU = []
    SE = []
    Pre = []
    Acc = []

    True_label=[]
    for classes in range(1,num_classes):
        overlap=((pred_seg==classes)*(label_seg==classes)).sum()
        union=(pred_seg==classes).sum()+(label_seg==classes).sum()
        Dice.append((2*overlap+1e-5)/(union+1e-5))
        True_label.append((label_seg==classes).sum())
        iou = (overlap + 0.1) / (union + 0.1 - overlap)
        IoU.append(iou)

        TP = overlap

        FP = (pred_seg == classes).sum() - TP
        FN = (label_seg == classes).sum() - TP
        se = TP / (TP + FN + 0.1)
        TN = pred_seg.shape[-2] * pred_seg.shape[-1] - union + TP

        acc = (TP + TN) / (TP + FP + TN + FN)

        SE.append(se)
        Precision = (overlap + 0.1) / (overlap + FP + 0.1)
        Pre.append(Precision)
        Acc.append(acc)

    return Dice,True_label,IoU,Acc,SE,Pre




def eval_multi_seg_3D(predict, target,num_classes=4):
    pred_seg = predict.data.cpu().numpy().astype(dtype=np.int)
    label_seg = target.data.cpu().numpy().astype(dtype=np.int)
    assert(pred_seg.shape == label_seg.shape)

    acc=(pred_seg==label_seg).sum()/(pred_seg.shape[0]*pred_seg.shape[1]*pred_seg.shape[2])

    Dice=[]
    True_label=[]
    for classes in range(1,num_classes):
        overlap=((pred_seg==classes)*(label_seg==classes)).sum()
        union=(pred_seg==classes).sum()+(label_seg==classes).sum()
        Dice.append((2*overlap+1e-5)/(union+1e-5))
        True_label.append((label_seg==classes).sum())

    return Dice,True_label,acc


def compute_PCC(predicts,label):
    assert(predicts.shape==label.shape)

    # for i in range(predicts.shape[1]):
    #     PCC.append([])   #如果分割一类这里需要修改
    i=0
    n = predicts.shape[0]*predicts.shape[2]*predicts.shape[3]
    TP = ((predicts[:,i,:,:]==1)*(label[:,i,:,:]==1)).sum()
    sum_x = predicts[:,i,:,:].sum()
    sum_y = label[:,i,:,:].sum()
    sum_x2 = predicts[:,i,:,:].pow(2).sum()
    sum_y2 = label[:,i,:,:].pow(2).sum()
    molecular = TP-(float(sum_x)*float(sum_y)/n)
    denominator = ((sum_x2-float(sum_x2**2)/n)*(sum_y2-float(sum_y2**2)/n)).sqrt()
    PCC=molecular/denominator

    return PCC


# def compute_score_Singl(predicts, label, forground=1,n_class=2):
#     '''
#     计算二分类Dice,Precsion,Recall,Jaccard，返回其值（不加smooth）
#     输入原标签0123
#     可算slice与cube
#
#     overlap=0的情况：
#         gt和predict无交集，正常计算各指标为0
#         gt无 predict有，recall为Nan，输出0
#         gt有 predict无，precsion为Nan,输出0
#         gt无 predict无，全部为Nan,输出0
#     '''
#     # score = 0
#     # count = 0
#     assert (predicts.shape == label.shape)
#
#     # PC=[]
#     # SE=[]
#     # Jaccard=[]
#     # SP=[]
#     #F1=[]
#     overlap = ((predicts == 1) * (label == 1)).sum().float()  # TP
#     # IOU=[]
#     # print('overlap:',overlap)
#     if (overlap > 0):
#         for i in range(predicts.shape[1]):
#             #overlap = ((predicts[:, i, :, :] == 1) * (label[:, i, :, :] == 1)).sum().float()  # TP
#             # dice = 2 * overlap / ((predicts[:,i,:,:] == 1).sum() + (label[:,i,:,:] == 1).sum()).float()
#             PC=overlap / (predicts[:,i,:,:] == 1).sum().float()
#             SE=overlap / (label[:,i,:,:] == 1).sum().float()
#             Jaccard=overlap / ((predicts[:,i,:,:] == 1).sum() + (label[:,i,:,:] == 1).sum() - overlap).float()
#             SP=((predicts[:,i,:,:] == 0) * (label[:,i,:,:] == 0)).sum().float()/(label[:,i,:,:] == 0).sum().float()
#
#             #F1 .append(2*overlap / (label[:,i,:,:] == 1).sum().float()*overlap / (predicts[:,i,:,:] == 1).sum().float()/(overlap / (label[:,i,:,:] == 1).sum() + overlap / (predicts[:,i,:,:] == 1).sum().float() + 1e-6))
#         return PC, SE, Jaccard, SP
#     else:
#         return 0, 0, 0, 0


def compute_score_Singl(predicts, label, forground=1,n_class=2):
    '''
    计算二分类Dice,Precsion,Recall,Jaccard，返回其值（不加smooth）
    输入原标签0123
    可算slice与cube

    overlap=0的情况：
        gt和predict无交集，正常计算各指标为0
        gt无 predict有，recall为Nan，输出0
        gt有 predict无，precsion为Nan,输出0
        gt无 predict无，全部为Nan,输出0
    '''
    # score = 0
    # count = 0
    assert (predicts.shape == label.shape)

    overlap = ((predicts == 1) * (label == 1)).sum().float()  # TP
    FP = (predicts == 1).sum() - overlap
    # IOU=[]
    # print('overlap:',overlap)
    if (overlap > 0):
        for i in range(predicts.shape[1]):
            #overlap = ((predicts[:, i, :, :] == 1) * (label[:, i, :, :] == 1)).sum().float()  # TP
            # dice = 2 * overlap / ((predicts[:,i,:,:] == 1).sum() + (label[:,i,:,:] == 1).sum()).float()
            PC=overlap / (predicts[:,i,:,:] == 1).sum().float()
            SE=overlap / (label[:,i,:,:] == 1).sum().float()
            Jaccard=overlap / ((predicts[:,i,:,:] == 1).sum() + (label[:,i,:,:] == 1).sum() - overlap).float()
            # SP=((predicts[:,i,:,:] == 0) * (label[:,i,:,:] == 0)).sum().float()/(label[:,i,:,:] == 0).sum().float()

            Pre = (overlap + 0.1) / (overlap + FP + 0.1)

            #F1 .append(2*overlap / (label[:,i,:,:] == 1).sum().float()*overlap / (predicts[:,i,:,:] == 1).sum().float()/(overlap / (label[:,i,:,:] == 1).sum() + overlap / (predicts[:,i,:,:] == 1).sum().float() + 1e-6))
        return PC, SE, Jaccard, Pre
    else:
        return 0, 0, 0, 0

def eval_multi_seg_3D_Single(predict, target, num_classes=1):
    pred_seg = predict.data.cpu().numpy().astype(dtype=np.int)
    # print(pred_seg.shape)

    label_seg = target.data.cpu().numpy().astype(dtype=np.int)
    # print(label_seg.shape)
    assert (pred_seg.shape == label_seg.shape)

    acc = (pred_seg == label_seg).sum() / (pred_seg.shape[0] * pred_seg.shape[2] * pred_seg.shape[3])

    Dice = []
    True_label = []
    classes = 1
    overlap = ((pred_seg == classes) * (label_seg == classes)).sum()
    union = (pred_seg == classes).sum() + (label_seg == classes).sum()
    Dice.append((2 * overlap + 1e-5) / (union + 1e-5))
    True_label.append((label_seg == classes).sum())

    return Dice, True_label, acc


