from models.Conv4 import BackBone
from models.classifier import Classifier
import numpy as np
import sys
import torch
import torch.nn as nn
import os
from tqdm.auto import tqdm
sys.path.append("..")
from data.testloaders.dataloaders import meta_test_dataloader
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import logging
import argparse
import torch.nn.functional as F

def get_score(acc_list):
    mean = np.mean(acc_list)
    interval = 1.95 * np.sqrt(np.var(acc_list) / len(acc_list))
    return mean, interval

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(f"test.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_num = 13

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

backbone_load_path = 'ckpt/un_cifar/1-shot/best_model.pt'
feature_extractor = BackBone().cuda()

state_dict = torch.load(backbone_load_path)['state_dict']
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if 'classifier' in k:
        continue
    else:
        name = k[5:]
        new_state_dict[name] = v
feature_extractor.load_state_dict(new_state_dict)
feature_extractor.eval()

way = 5
shot = 1
query_shot = 15
testloader = meta_test_dataloader(data_path='path/to/data', way=way, shot=shot,
                                  transform_type=1, query_shot=query_shot, trial=600)
k = way * query_shot
b = way * (shot + query_shot)
dim = 64

train_target = torch.LongTensor([i // shot for i in range(shot * way)]).cuda()
test_target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()
crieterion = nn.CrossEntropyLoss()
acclist = []

def create_argparser():
    default = {}
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=16,
        use_ddim=False,
        image_size=64,
        num_channels=128,
        num_res_blocks=3,
        learn_sigma=True,
        class_cond = False,
        diffusion_steps=1000,
        noise_schedule="cosine",
    )
    default.update(model_and_diffusion_defaults())
    default.update(defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default)
    return parser

args = create_argparser().parse_args()

model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)

model.cuda()
model.eval()

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = diffusion.interpolate(model, x, x[index, :], t=200, lam=lam)

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

for i, (inp, label) in tqdm(enumerate(testloader), ncols=0):
    inp = inp.cuda()
    support_inp = inp[:way * shot]
    query_inp = inp[way * shot:]

    clf = Classifier(input_dim=dim, output_dim=way).cuda()

    optimizer_clf = torch.optim.SGD(clf.parameters(),
                                    lr=5e-2, momentum=0.9, weight_decay=5e-4)
    optimizer_clf2 = torch.optim.SGD(clf.parameters(),
                                    lr=1e-2, momentum=0.9, weight_decay=5e-4)

    support_list = []
    targets_a_list = []
    targets_b_list = []
    lam_list = []

    support = feature_extractor(support_inp).detach()
    support = support.mean(2).mean(2)
    support = F.normalize(support, p=2, dim=support.dim() - 1, eps=1e-12)
    query = feature_extractor(query_inp)
    query = query.mean(2).mean(2)
    query = F.normalize(query, p=2, dim=query.dim() - 1, eps=1e-12)

    clf.train()
    for k in range(50):
        train_prd = clf(support)
        loss = crieterion(train_prd, train_target)
        optimizer_clf.zero_grad()
        loss.backward()
        optimizer_clf.step()

    for j in range(model_num):
        model_load_path = f'path/to/diffusion_model{j+1}/ckpt/model.pt'
        model.load_state_dict(torch.load(model_load_path), strict=False)
        support_mix, targets_a, targets_b, lam = mixup_data(support_inp, train_target, 0.4)
        support_mix = feature_extractor(support_mix).detach()
        support_mix = support_mix.mean(2).mean(2)
        support_mix = F.normalize(support_mix, p=2, dim=support_mix.dim() - 1, eps=1e-12)
        support_list.append(support_mix)
        targets_a_list.append(targets_a)
        targets_b_list.append(targets_b)
        lam_list.append(lam)

    for k in range(50):
        for j in range(model_num):
            train_prd = clf(support_list[j])
            loss = mixup_criterion(crieterion, train_prd, targets_a_list[j], targets_b_list[j], lam_list[j])
            loss = loss / model_num
            loss.backward()
        optimizer_clf2.step()
        optimizer_clf2.zero_grad()

    for k in range(50):
        train_prd = clf(support)
        loss = crieterion(train_prd, train_target)
        optimizer_clf.zero_grad()
        loss.backward()
        optimizer_clf.step()

    clf.eval()
    neg_l2_dist = clf(query)
    _, max_index = torch.max(neg_l2_dist, 1)
    acc = 100 * torch.sum(torch.eq(max_index, test_target)).item() / query_shot / way
    print(acc)
    logger.info('epoch' + str(i) + ' acc ' + str(acc))
    acclist.append(acc)

mean, interval = get_score(acclist)
print(mean, interval)
logger.info('mean' + str(mean) + ' interval ' + str(interval))

