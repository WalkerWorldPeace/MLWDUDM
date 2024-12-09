"""
use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
import sys
sys.path.append("..")
import torch as th
import torch.nn.functional as F
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from torchvision import utils
def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

DIR = "sync_image"

def classifiermain(dict, extractor, classifier, dataset, epoch):
    parser = create_argparser(dict)
    args, unknown = parser.parse_known_args()
    args.dataset = dataset
    args.epoch = epoch

    dist_util.setup_dist()
    logger.configure(dir=DIR, mode="sampling", args=args, epoch=args.epoch)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    extractor.eval()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            f = extractor(x_in)
            f = f.mean(2).mean(2)
            f = F.normalize(f, p=2, dim=f.dim() - 1, eps=1e-12)
            logits = classifier(f, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    if args.dataset == "co_miniimagenet":
        num_classes = 64
        NUM_CLASSES = 5
    elif args.dataset == "un_miniimagenet":
        num_classes = 52
        NUM_CLASSES = 4
    elif args.dataset == "co_cifarfs":
        num_classes = 64
        NUM_CLASSES = 5
    elif args.dataset == "un_cifarfs":
        num_classes = 52
        NUM_CLASSES = 4

    for i in range(num_classes):
        dataseti = i//NUM_CLASSES + 1
        model.load_state_dict(
            dist_util.load_state_dict(os.path.join(args.model_path,f"subset{dataseti}","ckpt/model.pt"), map_location="cpu")
        )
        img_id = 0
        while img_id < args.num_samples:
            model_kwargs = {}
            classes = th.full((args.batch_size,), i, device=dist_util.dev())
            model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=dist_util.dev(),
            )
            mkdirs(os.path.join(logger.get_dir(), f"class{i}"))
            sample = (sample + 1) / 2
            for j in range(args.batch_size):
                utils.save_image(
                    sample[j].float(), os.path.join(logger.get_dir(), f"class{i}", f"{img_id}.jpg")
                )
                img_id += 1

        logger.log("sampling complete")


def create_argparser(sampledict):
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        extractor_path="",
        classifier_scale=1.0,
        dataset="",
        epoch="",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    defaults.update(sampledict)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    classifiermain()
