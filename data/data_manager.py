from torch.utils.data import DataLoader
import torchvision.transforms as T
import data.sets as datasets
import data.trainloaders as dataset_loader

class DataManager(object):
    """
    Meta learning data manager
    """

    def __init__(self, args):
        super(DataManager, self).__init__()
        self.args = args

        print("Initializing dataset {}".format(args.dataset))
        dataset = datasets.init_imgfewshot_dataset(name=args.dataset, dataset_dir=args.validation_dir, epoch=args.epoch)
        if self.args.transform_type == 0:
            train_size_transform = T.RandomResizedCrop(64)
            test_size_transform = T.Compose([T.Resize(72),
                                             T.CenterCrop(64)])
        elif self.args.transform_type == 1:
            train_size_transform = T.Compose([T.Resize((64, 64), interpolation=3),
                                              T.RandomCrop(64, padding=4)])
            test_size_transform = T.Resize((args.height, args.width), interpolation=3)
        elif self.args.transform_type == 2:
            train_size_transform = T.Compose([T.Resize(72),
                                        T.RandomCrop(64)])
            test_size_transform = T.Compose([T.Resize(72),
                                             T.CenterCrop(64)])
        transform_train = T.Compose([
            train_size_transform,
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(0.5)
        ])

        transform_test = T.Compose([
            test_size_transform,
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        pin_memory = True

        self.trainloader = DataLoader(
            dataset_loader.init_loader(name='train_loader',
                                       dataset=dataset.train,
                                       labels2inds=dataset.train_labels2inds,
                                       labelIds=dataset.train_labelIds,
                                       nKnovel=args.nKnovel,
                                       nExemplars=args.nExemplars,
                                       nTestNovel=args.train_nTestNovel,
                                       epoch_size=args.train_epoch_size,
                                       transform=transform_train,
                                       load=args.load,
                                       ),
            batch_size=args.train_batch, shuffle=True, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=True,
        )

        self.valloader = DataLoader(
            dataset_loader.init_loader(name='test_loader',
                                       dataset=dataset.val,
                                       labels2inds=dataset.val_labels2inds,
                                       labelIds=dataset.val_labelIds,
                                       nKnovel=args.nKnovel,
                                       nExemplars=args.nExemplars,
                                       nTestNovel=args.nTestNovel,
                                       epoch_size=args.epoch_size,
                                       transform=transform_test,
                                       load=args.load,
                                       ),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )
        self.testloader = DataLoader(
            dataset_loader.init_loader(name='test_loader',
                                       dataset=dataset.test,
                                       labels2inds=dataset.test_labels2inds,
                                       labelIds=dataset.test_labelIds,
                                       nKnovel=args.nKnovel,
                                       nExemplars=args.nExemplars,
                                       nTestNovel=args.nTestNovel,
                                       epoch_size=args.epoch_size,
                                       transform=transform_test,
                                       load=args.load,
                                       ),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )

    def return_dataloaders(self):
        if self.args.phase == 'test':
            return self.trainloader, self.testloader
        elif self.args.phase == 'val':
            return self.trainloader, self.valloader
