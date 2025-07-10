# -*- coding: utf-8 -*-
class DefaultConfig(object):
    net_work = 'ReMultiSeg'
    data = './dataset'
    dataset = 'Thoracic_OAR'
    log_dirs = './Log_Dir/log'
    save_model_path = './checkpoints_all/{}'.format(net_work)
    
    best_model = '{}.pkl'.format(net_work)
    mode = 'train'
    num_epochs = 100
    batch_size = 8
    loss_type = 'mix_dice' 
    multitask = True
    validation_step = 1
    
    in_channels = 3
    num_classes = 7
    crop_height = 128
    crop_width = 128

    lr = 0.001
    lr_mode = 'poly'
    momentum = 0.9
    weight_decay = 1e-4
    
    pretrained = False
    pretrained_model_path = None

    resume_model_path = None
    continu = False
    save_every_checkpoint = False
    
    cuda = '0'
    num_workers = 0
    use_gpu = True
    
    trained_model_path = ''
    predict_fold = 'predict_mask'
