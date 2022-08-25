# Urban-Scene Segmentation using Adversarial Learning.


## THis repository is still under construction. __Not completed yet__



This repository is about Urban-Scene Segmentation using Adversarial Learning (Semantic segmentation + GAN).   
As you may know, GAN can offer ground truth's distribution knowledge to model by Adversarial loss.   
So I tried adding GAN to Semantic segmentation (City scapes).   



(Will add more infor further.)


---
### [Paper](https://arxiv.org/abs/2005.10821) | [YouTube](https://youtu.be/odAGA7pFBGA)  | [Cityscapes Score](https://www.cityscapes-dataset.com/method-details/?submissionID=7836) <br>

Pytorch implementation of our paper [Hierarchical Multi-Scale Attention for Semantic Segmentation](https://arxiv.org/abs/2005.10821).<br>

## Installation 

* The code is tested with pytorch 1.3 and python 3.6
* You can use ./Dockerfile to build an image.

## Download/Prepare Data

If using Cityscapes, download Cityscapes data, then update `config.py` to set the path:
```python
__C.DATASET.CITYSCAPES_DIR=<path_to_cityscapes>
```

* Download Autolabelled-Data from [google drive](https://drive.google.com/file/d/1DtPo-WP-hjaOwsbj6ZxTtOo_7R_4TKRG/view?usp=sharing)

If using Cityscapes Autolabelled Images, download Cityscapes data, then update `config.py` to set the path:
```python
__C.DATASET.CITYSCAPES_CUSTOMCOARSE=<path_to_cityscapes>
```


## Running the code

The instructions below make use of a tool called `runx`, which we find useful to help automate experiment running and summarization. For more information about this tool, please see [runx](https://github.com/NVIDIA/runx).
In general, you can either use the runx-style commandlines shown below. Or you can call `python train.py <args ...>` directly if you like.


### Run inference on Cityscapes

Dry run:
```bash
> python -m runx.runx scripts/eval_cityscapes.yml -i -n
```
This will just print out the command but not run. It's a good way to inspect the commandline. 

Real run:
```bash
> python -m runx.runx scripts/eval_cityscapes.yml -i
```

The reported IOU should be 86.92. This evaluates with scales of 0.5, 1.0. and 2.0. You will find evaluation results in ./logs/eval_cityscapes/...

### Dump images for Cityscapes

```bash
> python -m runx.runx scripts/dump_cityscapes.yml -i
```

This will dump network output and composited images from running evaluation with the Cityscapes validation set. 

### Run inference and dump images on a folder of images

```bash
> python -m runx.runx scripts/dump_folder.yml -i
```

## Train a model

Train cityscapes, using HRNet + OCR + multi-scale attention with fine data and mapillary-pretrained model
```bash
> python -m runx.runx scripts/train_cityscapes.yml -i
```


## Acknowledgments
This pytorch implementation is heavily derived from [NVIDIA segmentation](https://github.com/NVIDIA/semantic-segmentation).
Thanks to the NVIDIA implementations.