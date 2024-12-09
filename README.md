# [IEEE TCSVT] Meta-Learning Without Data via Unconditional Diffusion Model

## Abstract
Although few-shot learning aims to address data scarcity, it still requires large, annotated datasets for training, which are often unavailable due to cost and privacy concerns. Previous studies have utilized pre-trained diffusion models, either to synthesize auxiliary data besides limited labeled samples, or to employ diffusion models as zero-shot classifiers. However, they are limited to conditional diffusion models needing class prior information (e.g., carefully crafted text prompts) about unseen tasks. To overcome this, we leverage unconditional diffusion models without needs for class information to train a meta-model capable of generalizing to unseen tasks. The framework contains (1) a meta-learning without data approach that uses synthetic data during training; and (2) a diffusion model-based data augmentation to calibrate the distribution shift during testing. During meta-training, we implement a self-taught class-learner to gradually capture class concepts, guiding unconditional diffusion models to generate a labeled pseudo dataset. This pseudo dataset is then used to jointly train the class-learner and the meta-model, allowing for iterative refinement and clear differentiation between classes. During meta-testing, we introduce a data augmentation that employs the diffusion models used in meta-training, to narrow the gap between meta-training and meta-testing task distribution. This enables the meta-model trained on synthetic images to effectively classify real images in unseen tasks. Comprehensive experiments showcase the superiority and adaptability of our approach in four real-world scenarios.

## Requirements

```
pip install -r requirements.txt
```

## Meta-training run command 
If you want to train the meta-model, you should first check the data related hyper-parameters in `config.py`.
```
python main.py --dataset co_miniimagenet --save-dir ckpt/co_miniimagenet/1-shot -g 0 --nExemplars 1
```
## Meta-testing run command
```
python diffusionmix.py
```
## Prepare pre-trained models
```
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 24"
python guided_diffusion/image_train.py --data_dir path/to/data $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```
To pre-train conditional diffusion models, you can add the `--class_cond True`.

## Citation
If you find MLWDUDM useful for your research and applications, please cite using this BibTeX:
```bash
@article{wei2024meta,
  title={Meta-learning without data via unconditional diffusion models},
  author={Wei, Yongxian and Hu, Zixuan and Shen, Li and Wang, Zhenyi and Li, Lei and Li, Yu and Yuan, Chun},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
```
## Acknowledgement
[Guided-diffusion](https://github.com/openai/guided-diffusion): the codebase we built upon.
