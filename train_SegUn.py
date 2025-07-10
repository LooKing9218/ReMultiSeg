# -*- coding: utf-8 -*-
import os
from datetime import datetime
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data.slit_loader import Slit_loader

from utils.loss import loss_builder
import utils.utils as u
from models.net_builder import net_builder
from utils.config_UnSeg import DefaultConfig
torch.set_num_threads(2)
def val(args, model, dataloader, mode, epoch=1):
    print('\n')
    print('Start {}!'.format(mode))
    with torch.no_grad():
        model.eval()
        tbar = tqdm.tqdm(dataloader, desc='\r')

        total_Dice = []
        total_Dice1 = []
        total_Dice2 = []
        total_Dice3 = []
        total_Dice4 = []
        total_Dice5 = []
        total_Dice6 = []
        total_Dice.append(total_Dice1)
        total_Dice.append(total_Dice2)
        total_Dice.append(total_Dice3)
        total_Dice.append(total_Dice4)
        total_Dice.append(total_Dice5)
        total_Dice.append(total_Dice6)



        total_IoU = []
        total_IoU1 = []
        total_IoU2 = []
        total_IoU3 = []
        total_IoU4 = []
        total_IoU5 = []
        total_IoU6 = []
        total_IoU.append(total_IoU1)
        total_IoU.append(total_IoU2)
        total_IoU.append(total_IoU3)
        total_IoU.append(total_IoU4)
        total_IoU.append(total_IoU5)
        total_IoU.append(total_IoU6)

        total_Acc = []
        total_Acc1 = []
        total_Acc2 = []
        total_Acc3 = []
        total_Acc4 = []
        total_Acc5 = []
        total_Acc6 = []

        total_Acc.append(total_Acc1)
        total_Acc.append(total_Acc2)
        total_Acc.append(total_Acc3)
        total_Acc.append(total_Acc4)
        total_Acc.append(total_Acc5)
        total_Acc.append(total_Acc6)

        total_Se = []
        total_Se1 = []
        total_Se2 = []
        total_Se3 = []
        total_Se4 = []
        total_Se5 = []
        total_Se6 = []
        total_Se.append(total_Se1)
        total_Se.append(total_Se2)
        total_Se.append(total_Se3)
        total_Se.append(total_Se4)
        total_Se.append(total_Se5)
        total_Se.append(total_Se6)

        total_Pre = []
        total_Pre1 = []
        total_Pre2 = []
        total_Pre3 = []
        total_Pre4 = []
        total_Pre5 = []
        total_Pre6 = []
        total_Pre.append(total_Pre1)
        total_Pre.append(total_Pre2)
        total_Pre.append(total_Pre3)
        total_Pre.append(total_Pre4)
        total_Pre.append(total_Pre5)
        total_Pre.append(total_Pre6)


        cur_predict_cube = []
        cur_label_cube = []
        counter = 0
        end_flag = False

        for idx_val, (data, labels) in enumerate(tbar):
            # tbar.update()
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = labels.cuda()
            slice_num = args.batch_size

            predicts,_,_ = model(data,label,epoch)
            predict = torch.argmax(torch.exp(predicts), dim=1)
            batch_size = predict.size(0)

            counter += batch_size
            if counter <= slice_num:
                cur_predict_cube.append(predict)
                cur_label_cube.append(label)
                if counter == slice_num:
                    end_flag = True
                    counter = 0
            if end_flag:
                end_flag = False
                label_cube = torch.cat(cur_label_cube, dim=0).squeeze(1)
                predict_cube = torch.cat(cur_predict_cube, dim=0)

                cur_predict_cube = []
                cur_label_cube = []

                assert len(args.cuda.split(',')) * predict_cube.size()[0] == slice_num
                # Dice, True_label, IoU, acc, SE, Pre
                dice_list, true_label, iou_list, Acc_list, SE_list, Pre_list = \
                    u.eval_multi_seg_3D_infer(predict_cube, label_cube,args.num_classes)

                for class_id in range(args.num_classes - 1):
                    if true_label[class_id] != 0:
                        total_Dice[class_id].append(dice_list[class_id])
                        total_IoU[class_id].append(iou_list[class_id])
                        total_Acc[class_id].append(Acc_list[class_id])
                        total_Se[class_id].append(SE_list[class_id])
                        total_Pre[class_id].append(Pre_list[class_id])

                len0 = len(total_Dice[0]) if len(total_Dice[0]) != 0 else 1
                len1 = len(total_Dice[1]) if len(total_Dice[1]) != 0 else 1
                len2 = len(total_Dice[2]) if len(total_Dice[2]) != 0 else 1
                len3 = len(total_Dice[3]) if len(total_Dice[3]) != 0 else 1
                len4 = len(total_Dice[4]) if len(total_Dice[4]) != 0 else 1
                len5 = len(total_Dice[5]) if len(total_Dice[5]) != 0 else 1
                dice1 = sum(total_Dice[0]) / len0
                dice2 = sum(total_Dice[1]) / len1
                dice3 = sum(total_Dice[2]) / len2
                dice4 = sum(total_Dice[3]) / len3
                dice5 = sum(total_Dice[4]) / len4
                dice6 = sum(total_Dice[5]) / len5
                mean_dice = (dice1 + dice2 + dice3+ dice4 + dice5 + dice6) / (args.num_classes-1)

                IoU1 = sum(total_IoU[0]) / len0
                IoU2 = sum(total_IoU[1]) / len1
                IoU3 = sum(total_IoU[2]) / len2
                IoU4 = sum(total_IoU[3]) / len3
                IoU5 = sum(total_IoU[4]) / len4
                IoU6 = sum(total_IoU[5]) / len5
                mean_IoU = (IoU1 + IoU2 + IoU3+ IoU4+ IoU5+ IoU6) /(args.num_classes-1)

                Acc1 = sum(total_Acc[0]) / len0
                Acc2 = sum(total_Acc[1]) / len1
                Acc3 = sum(total_Acc[2]) / len2
                Acc4 = sum(total_Acc[3]) / len3
                Acc5 = sum(total_Acc[4]) / len4
                Acc6 = sum(total_Acc[5]) / len5
                mean_Acc = (Acc1 + Acc2 + Acc3 + Acc4 + Acc5 + Acc6) / (args.num_classes-1)

                Se1 = sum(total_Se[0]) / len0
                Se2 = sum(total_Se[1]) / len1
                Se3 = sum(total_Se[2]) / len2
                Se4 = sum(total_Se[3]) / len3
                Se5 = sum(total_Se[4]) / len4
                Se6 = sum(total_Se[5]) / len5
                mean_Se = (Se1 + Se2 + Se3 + Se4 + Se5 + Se6 ) /  (args.num_classes-1)

                Pre1 = sum(total_Pre[0]) / len0
                Pre2 = sum(total_Pre[1]) / len1
                Pre3 = sum(total_Pre[2]) / len2
                Pre4 = sum(total_Pre[3]) / len3
                Pre5 = sum(total_Pre[4]) / len4
                Pre6 = sum(total_Pre[5]) / len5
                mean_Pre = (Pre1 + Pre2 + Pre3+ Pre4+ Pre5+ Pre6) / (args.num_classes-1)

                tbar.set_description(
                    'Mean_D: %4f, Dice1: %.4f, Dice2: %.4f, Dice3: %.4f,Dice4: %.4f,Dice5: %.4f,Dice6: %.4f, mean_Pre: %.4f' % (
                        mean_dice, dice1, dice2, dice3, dice4, dice5, dice6, mean_Pre))

        print('{}_Mean_Dice:'.format(mode), mean_dice)
        print('{}_Dice1:'.format(mode), dice1)
        print('{}_Dice2:'.format(mode), dice2)
        print('{}_Dice3:'.format(mode), dice3)
        print('{}_Dice4:'.format(mode), dice4)
        print('{}_Dice5:'.format(mode), dice5)
        print('{}_Dice6:'.format(mode), dice6)
        with open('{}/fold_dice.txt'.format(args.save_model_path), 'a+') as f:
            f.write('{}_Mean_Dice:'.format(mode) + " " + str(round(mean_dice, 4)) + ',')
            f.write('{}_Mean_IoU:'.format(mode) + " " + str(round(mean_IoU, 4)) + ',')
            f.write('{}_Mean_Acc:'.format(mode) + " " + str(round(mean_Acc, 4)) + ',')
            f.write('{}_Mean_Se:'.format(mode) + " " + str(round(mean_Se, 4)) + ',')
            f.write('{}_Mean_Pre:'.format(mode) + " " + str(round(mean_Pre, 4)) + ',')
            f.write('\n')

            f.write('{}_Dice: 1:'.format(mode)  + " " + str(round(dice1, 4)) + ',')
            f.write('{}_IoU: 1:'.format(mode)  + " " + str(round(IoU1, 4)) + ',')
            f.write('{}_Acc: 1:'.format(mode)  + " " + str(round(Acc1, 4)) + ',')
            f.write('{}_Se: 1:'.format(mode)  + " " + str(round(Se1, 4)) + ',')
            f.write('{}_Pre: 1:'.format(mode)  + " " + str(round(Pre1, 4)) + ',')
            f.write('\n')
            f.write('{}_Dice: 2:'.format(mode)  + " " + str(round(dice2, 4)) + ',')
            f.write('{}_IoU: 2:'.format(mode)  + " " + str(round(IoU2, 4)) + ',')
            f.write('{}_Acc: 2:'.format(mode)  + " " + str(round(Acc2, 4)) + ',')
            f.write('{}_Se: 2:'.format(mode)  + " " + str(round(Se2, 4)) + ',')
            f.write('{}_Pre: 2:'.format(mode)  + " " + str(round(Pre2, 4)) + ',')
            f.write('\n')
            f.write('{}_Dice: 3:'.format(mode)  + " " + str(round(dice3, 4)) + ',')
            f.write('{}_IoU: 3:'.format(mode)  + " " + str(round(IoU3, 4)) + ',')
            f.write('{}_Acc: 3:'.format(mode)  + " " + str(round(Acc3, 4)) + ',')
            f.write('{}_Se: 3:'.format(mode)  + " " + str(round(Se3, 4)) + ',')
            f.write('{}_Pre: 3:'.format(mode)  + " " + str(round(Pre3, 4)) + ',')
            f.write('\n')
            f.write('{}_Dice: 4:'.format(mode)  + " " + str(round(dice4, 4)) + ',')
            f.write('{}_IoU: 4:'.format(mode)  + " " + str(round(IoU4, 4)) + ',')
            f.write('{}_Acc: 4:'.format(mode)  + " " + str(round(Acc4, 4)) + ',')
            f.write('{}_Se: 4:'.format(mode)  + " " + str(round(Se4, 4)) + ',')
            f.write('{}_Pre: 4:'.format(mode)  + " " + str(round(Pre4, 4)) + ',')
            f.write('\n')
            f.write('{}_Dice: 5:'.format(mode)  + " " + str(round(dice5, 4)) + ',')
            f.write('{}_IoU: 5:'.format(mode)  + " " + str(round(IoU5, 4)) + ',')
            f.write('{}_Acc: 5:'.format(mode)  + " " + str(round(Acc5, 4)) + ',')
            f.write('{}_Se: 5:'.format(mode)  + " " + str(round(Se5, 4)) + ',')
            f.write('{}_Pre: 5:'.format(mode)  + " " + str(round(Pre5, 4)) + ',')
            f.write('\n')
            f.write('{}_Dice: 6:'.format(mode)  + " " + str(round(dice6, 4)) + ',')
            f.write('{}_IoU: 6:'.format(mode)  + " " + str(round(IoU6, 4)) + ',')
            f.write('{}_Acc: 6:'.format(mode)  + " " + str(round(Acc6, 4)) + ',')
            f.write('{}_Se: 6:'.format(mode)  + " " + str(round(Se6, 4)) + ',')
            f.write('{}_Pre: 6:'.format(mode)  + " " + str(round(Pre6, 4)) + ',')
            f.write('\n')

            f.write('\n')
        return mean_dice, dice1, dice2, dice3,dice4, dice5, dice6, mean_Pre

def train(args, model,optimizer,criterion, writer, dataloader_train, dataloader_val, dataloader_test):
    step = 0
    best_dice = 0.0
    best_dice_Test = 0.0
    for epoch in range(1, args.num_epochs+1):
        lr = u.adjust_learning_rate(args, optimizer, epoch-1)
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        train_loss = 0.0

        for i, (data, labels) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = labels.cuda().long()

            optimizer.zero_grad()
            main_out,Un,loss_cel = model(data,label,epoch)

            loss_el = criterion[1](main_out, label[:, 0, :, :],Un,epoch)
            loss = loss_el + loss_cel
            loss.backward()
            optimizer.step()
            tq.update(args.batch_size)
            train_loss += loss.item()
            tq.set_postfix(loss='%.6f' % (train_loss / (i + 1)))
            step += 1
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('Train/loss_epoch', float(loss_train_mean),
                          epoch)
        print('loss for train : %f' % (loss_train_mean))

        if epoch % args.validation_step == 0:
            if not os.path.exists(args.save_model_path):
                os.makedirs(args.save_model_path)
            with open('{}/fold_dice.txt'.format(args.save_model_path), 'a+') as f:
                f.write('EPOCH:' + str(epoch) + ',')
                f.write('lr:' + str(lr) + ',\n')
            mean_Dice, Dice1, Dice2, Dice3, Dice4, Dice5, Dice6, mean_Pre = val(args, model, dataloader_val, mode="Val",epoch=epoch)  # 验证集结果
            writer.add_scalar('Valid/Mean_val', mean_Dice, epoch)
            writer.add_scalar('Valid/Dice1_val', Dice1, epoch)
            writer.add_scalar('Valid/Dice2_val', Dice2, epoch)
            writer.add_scalar('Valid/Dice3_val', Dice3, epoch)
            writer.add_scalar('Valid/Dice4_val', Dice3, epoch)
            writer.add_scalar('Valid/Dice5_val', Dice3, epoch)
            writer.add_scalar('Valid/Dice6_val', Dice3, epoch)
            writer.add_scalar('Valid/Pre_val', mean_Pre, epoch)


            is_best = mean_Dice > best_dice
            best_dice = max(best_dice, mean_Dice)

            checkpoint_dir = os.path.join(args.save_model_path)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)


            if is_best:
                mean_Dice_Test, Dice1_Test, Dice2_Test, Dice3_Test,Dice4_Test,Dice5_Test,Dice6_Test, mean_Pre_Test = val(args, model, dataloader_test, mode="Test",epoch=epoch)
                writer.add_scalar('Test/Mean_Test', mean_Dice_Test, epoch)
                writer.add_scalar('Test/Dice1_Test', Dice1_Test, epoch)
                writer.add_scalar('Test/Dice2_Test', Dice2_Test, epoch)
                writer.add_scalar('Test/Dice3_Test', Dice3_Test, epoch)
                writer.add_scalar('Test/Dice4_Test', Dice3_Test, epoch)
                writer.add_scalar('Test/Dice5_Test', Dice3_Test, epoch)
                writer.add_scalar('Test/Dice6_Test', Dice3_Test, epoch)
                writer.add_scalar('Test/Pre_Test', mean_Pre_Test, epoch)
                best_dice_Test = mean_Dice_Test
                print('===> Saving models...')

                u.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_dice': best_dice,
                    'best_dice_test': best_dice_Test,
                }, best_dice,best_dice_Test, epoch, is_best, checkpoint_dir, stage="Val",filename=os.path.join(checkpoint_dir,"checkpoint.pth.tar"))


def main(mode='train', args=None,writer=None):
    # create dataset and dataloader
    dataset_path = os.path.join(args.data, args.dataset)
    dataset_train = Slit_loader(dataset_path, scale=(args.crop_height, args.crop_width),
                                mode='train')
    print("args.batch_size ========= {}".format(args.batch_size))
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    dataset_val = Slit_loader(dataset_path, scale=(args.crop_height, args.crop_width),
                              mode='valid')
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    dataset_test = Slit_loader(dataset_path, scale=(args.crop_height, args.crop_width),
                               mode='test')

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )


    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # bulid model
    model = net_builder(args.net_work, args.num_classes, args.pretrained)
    print('Model have been loaded!,you chose the ' + args.net_work + '!')
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load trained model for test
    if args.trained_model_path and mode == 'test':
        print("=> loading trained model '{}'".format(args.trained_model_path))
        checkpoint = torch.load(
            args.trained_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Done!')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = loss_builder(args.loss_type, args.multitask,class_num=args.num_classes)

    if mode == 'train':
        train(args, model,optimizer,criterion, writer, dataloader_train, dataloader_val, dataloader_test)


if __name__ == '__main__':
    seed = 100
    u.setup_seed(seed)
    args = DefaultConfig()
    modes = args.mode
    if modes == 'train':
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(args.log_dirs,
                               current_time)
        writer = SummaryWriter(log_dir=log_dir)
        args.save_model_path = args.save_model_path + "_{}".format(current_time)
        main(mode='train', args=args, writer=writer)