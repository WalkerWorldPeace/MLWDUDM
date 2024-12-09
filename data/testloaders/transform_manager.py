import torchvision.transforms as transforms

def get_transform(transform_type=None):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=mean,std=std)
                                    ])

    if transform_type == 0:
        size_transform = transforms.Compose([transforms.Resize(72),
                                            transforms.CenterCrop(64)])
    elif transform_type == 1:
        size_transform = transforms.Compose([transforms.Resize((64, 64), interpolation=3),
                                             ])
    elif transform_type == 2:
        size_transform = transforms.Compose([transforms.Resize(72),
                                             transforms.CenterCrop(64)])
    else:
        raise Exception('transform_type must be specified during inference if not using pre!')

    eval_transform = transforms.Compose([size_transform,normalize])
    return eval_transform
