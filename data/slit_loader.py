import torch
import glob
from torchvision import transforms
from PIL import Image
import numpy as np
import SimpleITK as sitk

class Slit_loader(torch.utils.data.Dataset):

    def __init__(self, dataset_path, scale, mode='train'):
        super().__init__()
        self.mode = mode
        if mode == 'train':
            print("==========")
            self.img_path = dataset_path + '/img' + '/train'
            self.mask_path = dataset_path + '/mask' + '/train'
            self.image_lists, self.label_lists = self.read_list(self.img_path)

        if mode == 'valid':
            self.img_path = dataset_path + '/img' + '/valid'
            self.mask_path = dataset_path + '/mask' + '/valid'
            self.image_lists, self.label_lists = self.read_list(self.img_path)
        elif mode == 'test':
            self.img_path = dataset_path + '/img' + '/test'
            self.mask_path = dataset_path + '/mask' + '/test'
            self.image_lists, self.label_lists = self.read_list(self.img_path)
        # print("self.image_lists =========== {}".format(self.image_lists))
        # data augmentation
        # resize
        self.to_gray = transforms.Grayscale()
        # normalization
        self.to_tensor = transforms.ToTensor()  # 将numpy的ndarray或PIL.Image读的图片转换成形状为(C,H, W)的Tensor格式，

    def __getitem__(self, index):
        # load image
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(
            self.image_lists[index]))
        # if np.min(img_arr) < 0:
        img = 2*(img_arr-np.min(img_arr))/(np.max(img_arr)-np.min(img_arr))-1
        #print(img.dtype)
        # load label
        label = Image.open(self.label_lists[index])
        label = np.array(label).astype(np.uint8)

        label = label.reshape((1, label.shape[0], label.shape[1]))
        # print("max(label) ======    {}".format(np.max(label)))

        labels = torch.from_numpy(label.copy()).float()


        img = img.reshape(1, img.shape[0], img.shape[1])
        img = torch.from_numpy(img.copy()).float()
        # test模式下labels返回label的路径列表
        return img, labels

    def __len__(self):
        return len(self.image_lists)

    def read_list(self, image_path):
        print("Mode: {}, image_path: {}".format(self.mode,image_path))
        img_list = glob.glob(image_path + '/*.nii.gz')
        label_list_tmp = [x.replace("img","mask") for x in img_list]
        label_list = [x.replace(".nii.gz",".png") for x in label_list_tmp]

        assert len(img_list) == len(label_list)
        print('Total {} image is:{}'.format(self.mode, len(img_list)))

        return img_list, label_list


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    data = Slit_loader(r'E:\code\corneal_ulcer_zhongshan\seg_corneal_ulcers\Datasets\DATASET', (512, 256), mode='val')
    dataloader_train = DataLoader(
        data,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    for i, (img, label) in enumerate(dataloader_train):
        print(img.shape)
        #print(label.shape)  # train
        print(img)
        print(label[0].shape)  # val
        # print(label)  # test

        print(i)

