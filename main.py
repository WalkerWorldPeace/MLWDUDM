import datetime
import os
import os.path as osp
import sys
import time
import torch.nn.functional as F
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from classifier_sample import classifiermain
from config import config
from data.data_manager import DataManager
from models.net import Model
from models.classifier import Classifier
from guided_diffusion.script_util import create_gaussian_diffusion
from guided_diffusion.resample import create_named_schedule_sampler
from utils.avgmeter import AverageMeter
from utils.ci import mean_confidence_interval
from utils.iotools import save_checkpoint
from utils.logger import Logger
from utils.losses import CrossEntropyLoss
from utils.optimizers import init_optimizer
from utils.torchtools import adjust_learning_rate, one_hot
defaults = dict(
    image_size=64,
    num_channels=128,
    num_res_blocks=3,
    learn_sigma=True,
    class_cond=False,
    diffusion_steps=1000,
    noise_schedule="cosine",
    batch_size=100,
    num_samples=600,
    timestep_respacing="ddim25",
    use_ddim=True,
    classifier_scale =1.0,
    model_path = "path/to/pretrained/models"
    )

def main(args):
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()

    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    model = Model(scale_cls=args.scale_cls, num_classes=args.num_classes)
    criterion = CrossEntropyLoss()
    criterion_clf = nn.CrossEntropyLoss()
    optimizer = init_optimizer(args.optim, model.parameters(), args.lr, args.weight_decay)
    classifier = Classifier(output_dim=args.num_classes)
    optimizer_clf = init_optimizer(args.optim, classifier.parameters(), args.lr, args.weight_decay)

    if args.resume is not None:
        state_dict = torch.load(args.resume)['state_dict']
        model.load_state_dict(state_dict)
        print('Load model from {}'.format(args.resume))

    if use_gpu:
        model = model.cuda()
        classifier = classifier.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            classifier = torch.nn.DataParallel(classifier)

    start_time = time.time()
    train_time = 0
    best_acc = -np.inf
    best_epoch = 0
    print("==> Start training")

    diffusion = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=True,
        noise_schedule="cosine",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
    )
    schedule_sampler = create_named_schedule_sampler(
        "uniform", diffusion
    )

    print('Initializing image data manager')
    classifiermain(dict=defaults, extractor=model.base, classifier=classifier, dataset=args.dataset, epoch=args.epoch)
    dm = DataManager(args)
    trainloader, testloader = dm.return_dataloaders()

    for epoch in range(args.start_epoch, args.max_epoch):
        learning_rate = adjust_learning_rate(optimizer, epoch, args.LUT_lr)

        start_train_time = time.time()

        train(epoch, model, criterion, criterion_clf, optimizer, trainloader, learning_rate, use_gpu)
        train_classifier(epoch, model, classifier, criterion_clf, optimizer_clf, trainloader, learning_rate / 10,
                        use_gpu, schedule_sampler, diffusion)
        train_time += round(time.time() - start_train_time)
        
        if epoch == 0 or epoch > (args.stepsize[0]-1) or (epoch + 1) % 10 == 0:
            acc = val(model, testloader, use_gpu)
            
            is_best = acc > best_acc
            
            if is_best:
                best_acc = acc
                best_epoch = epoch + 1
            if isinstance(model, torch.nn.DataParallel):
                state_dict = model.module.state_dict()
                state_dict_clf = classifier.module.state_dict()
            else:
                state_dict = model.state_dict()
                state_dict_clf = classifier.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'acc': acc,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pt'))
            save_checkpoint(state_dict_clf, fpath=osp.join(args.save_dir, 'classifier_ep' + str(epoch + 1) + '.pt'))
            print("==> Test 5-way Best accuracy {:.2%}, achieved at epoch {}".format(best_acc, best_epoch))

        if (epoch + 1) % 10 == 0 and (epoch + 1) != args.max_epoch:
            classifiermain(dict = defaults, extractor = model.base, classifier = classifier, dataset = args.dataset, epoch = epoch + 1)
            args.epoch = epoch + 1
            dm = DataManager(args)
            trainloader, testloader = dm.return_dataloaders()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    print("==========\nArgs:{}\n==========".format(args))


def train(epoch, model, criterion, criterion_clf, optimizer, trainloader, learning_rate, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for images_train, labels_train, images_test, labels_test, pids in trainloader:
        data_time.update(time.time() - end)

        if use_gpu:
            images_train, labels_train = images_train.cuda(), labels_train.cuda()
            images_test, labels_test = images_test.cuda(), labels_test.cuda()
            pids = pids.cuda()

        labels_train_1hot = one_hot(labels_train).cuda()
        labels_test_1hot = one_hot(labels_test).cuda()

        ytest, cls_scores = model(images_train, images_test, labels_train_1hot, labels_test_1hot)

        loss1 = criterion_clf(ytest, pids.view(-1))
        loss2 = criterion(cls_scores, labels_test.view(-1))
        loss = loss1 + 0.5 * loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), pids.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'lr: {1} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Loss:{loss.avg:.4f} '.format(
           epoch+1, learning_rate, batch_time=batch_time, 
           data_time=data_time, loss=losses))


def val(model, testloader, use_gpu):
    accs = AverageMeter()
    test_accuracies = []
    model.eval()

    with torch.no_grad():
        for images_train, labels_train, images_test, labels_test, _ in testloader:
            batch_size = images_train.size(0)
            num_test_examples = images_test.size(1)

            labels_train_1hot = one_hot(labels_train)
            labels_test_1hot = one_hot(labels_test)
            if use_gpu:
                images_train = images_train.cuda()
                images_test = images_test.cuda()
                labels_train_1hot = labels_train_1hot.cuda()
                labels_test_1hot = labels_test_1hot.cuda()

            cls_scores = model(images_train, images_test, labels_train_1hot, labels_test_1hot)
            cls_scores = cls_scores.view(batch_size * num_test_examples, -1)
            labels_test = labels_test.view(batch_size * num_test_examples)

            _, preds = torch.max(cls_scores.detach().cpu(), 1)
            acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
            accs.update(acc.item(), labels_test.size(0))

            gt = (preds == labels_test.detach().cpu()).float()
            gt = gt.view(batch_size, num_test_examples).numpy()
            acc = np.sum(gt, 1) / num_test_examples
            acc = np.reshape(acc, (batch_size))
            test_accuracies.append(acc)

    accuracy = accs.avg
    test_accuracies = np.array(test_accuracies)
    test_accuracies = np.reshape(test_accuracies, -1)
    mean_acc, ci = mean_confidence_interval(test_accuracies)
    print('Accuracy: {:.2%}, std: :{:.2%}'.format(mean_acc, ci))

    return accuracy


def train_classifier(epoch, model, classifier, criterion, optimizer, trainloader, learning_rate, use_gpu, schedule_sampler, diffusion):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()
    classifier.train()

    end = time.time()
    for images_train, labels_train, images_test, labels_test, pids in trainloader:
        data_time.update(time.time() - end)

        if use_gpu:
            images_test, labels_test = images_test.cuda(), labels_test.cuda()
            pids = pids.cuda()

        images_test = images_test.view(-1, images_test.size(2), images_test.size(3), images_test.size(4))
        feature_true = model.get_embeddings(images_test)

        t, _ = schedule_sampler.sample(images_test.shape[0], "cuda")
        images_test = diffusion.q_sample(images_test, t)
        feature_test = model.get_embeddings(images_test)

        predict_test = classifier(feature_test, t)
        predict_true = model.classifier(feature_true)

        loss1 = criterion(predict_test, pids.view(-1))
        loss2 = F.kl_div(F.log_softmax(predict_test, dim=-1), F.softmax(predict_true, dim=-1))
        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), pids.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    print('Classifier'
          'Epoch{0} '
          'lr: {1} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Loss:{loss.avg:.4f} '.format(
        epoch + 1, learning_rate, batch_time=batch_time,
        data_time=data_time, loss=losses))

def val_classifier(model, classifier, testloader, use_gpu, schedule_sampler, diffusion):
    accs = AverageMeter()
    model.eval()
    classifier.eval()
    with torch.no_grad():
        for images_train, labels_train, images_test, labels_test, pids in testloader:
            if use_gpu:
                images_test = images_test.cuda()
                pids = pids.view(-1).cuda()

            images_test = images_test.view(-1, images_test.size(2), images_test.size(3), images_test.size(4))

            t, _ = schedule_sampler.sample(images_test.shape[0], "cuda")
            images_test = diffusion.q_sample(images_test, t)

            feature_test = model.get_embeddings(images_test)
            predict_test = classifier(feature_test)

            _, preds = torch.max(predict_test.detach(), 1)
            acc = (torch.sum(preds == pids).float()) / preds.size(0)
            accs.update(acc.cpu().item(), preds.size(0))

    accuracy = accs.avg
    print('Classifier Accuracy: {:.2%}'.format(accuracy))

    return accuracy

if __name__ == '__main__':
    args = config()
    main(args)