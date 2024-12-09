import torch
import torchvision.datasets as datasets
from PIL import Image
from data.testloaders import sampler, transform_manager


def get_dataset(data_path, transform_type):
    dataset = datasets.ImageFolder(
        data_path,
        loader=lambda x: image_loader(path=x, transform_type=transform_type))
    return dataset


def meta_test_dataloader(data_path,way,shot,transform_type=None,query_shot=16,trial=1000):

    dataset = get_dataset(data_path=data_path,transform_type=transform_type)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler.random_sampler(data_source=dataset, way=way, shot=shot, query_shot=query_shot, trial=trial),
        num_workers=4,
        pin_memory=False)
    return loader


def image_loader(path, transform_type):
    p = Image.open(path)
    p = p.convert('RGB')
    final_transform = transform_manager.get_transform(transform_type=transform_type)
    p = final_transform(p)
    return p
