
def net_builder(name,pretrained_model=None,pretrained=False):

    if name == 'ReMultiSeg':
        from models.EDEMA_Net.ReMultiSeg import UnSegNet
        net= UnSegNet(num_classes=7)

    else:
        raise NameError("Unknow Model Name!")
    return net
