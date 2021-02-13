


# ICNN

This repository is a pytorch implementation of interpretable convolutional neural network
([arXiv](https://arxiv.org/abs/1710.00935), [IEEE T-PAMI](https://ieeexplore.ieee.org/document/9050545)). 

It is created by [Wen Shen](https://ada-shen.github.io), Ping Zhao, Qiming Li, [Chao Li](
http://www.ict.cas.cn/sourcedb_2018_ict_cas/cn/jssrck/201810/t20181030_5151364.html).

## Notice
- **model**

The project now supports three different vggs (vgg_vd_16, vgg_m, vgg_s),
                                         alexnet, resnet-18/50, and densenet-121.
                                         
You can add your own model in the `/model` folder and register the model in `/tools/init_model.py`.

- **dataset**

Manged to sucessfuly run on  dataset bellow with minor modifications 

The project now supports  ** ilsvrc animalpart ** 


**Note that** in our code, we will first make the data into imdb file, 
so if your dataset is large, the preprocessing time may be long, 
and the generated imdb file will be relatively large.

## Requirement

The environment should contain all packages in [requirements.txt](./requirements.txt)
```bash
$ pip install -r requirements.txt
```

You can see that we recommend **pytorch=1.2.0**, this is because we find some bugs when pytorch=1.4.0,
but there is no such problem in pytorch 1.2.0. We will continue to study this problem.

Note this also implies we need to run on either python3.6 or 3.7.  There are no python wheels for pytorch 1.2.0 
for python3.8 and above. 

## Usage
Here, we take **resnet-18 + voc2010_crop bird classification** as an example.

To run a sample  i got the following to work 
```bash
$ python demo.py --model resnet_18 --dataset ilsvrcanimalpart --label_name n01443537
```

# Note 
there is some dataset preparation needed. Downlaod the ilsvrcanimalpart dataset. Within this repos directory make the folder datasets such that we have  ICNN/datasets/detanimalpart/.  

I also took the liberty to establish what the "negative' class for this dataset is. in load_data.py line 65 I define the negative class as being n01503061_obj. 
This will break compatability with other datases but i just needed a runnable example to define my own train method.


After running the instruction above, you will get a new folder whose path is
`/resnet_18/voc2010_crop/bird` in the `/task/classification` folder.

The new folder `bird` will contain a subfolder named `0` (correspond to your task_id) and three mat files (mean.mat, train.mat and val.mat).
The `0` folder stores the model of every 10 epoches and log which contains 
**train/val loss** and **train/val accuracy**  during network training.

You can use the trained model to calculate other metrics or to look at middle-level features.

<!--our experiment environment: 
    python: 3.7.7
    torch: 1.2.0
    torchvision: 0.4.0a0
    cuda: 10.2
    gpu: 2080Ti
-->

## Citation

If you use this project in your research, please cite:

```
@inproceedings{zhang2018interpretable,
 title={Interpretable convolutional neural networks},
 author={Zhang, Quanshi and Wu, Nianying and Zhu, Song-Chun},
 booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
 pages={8827--8836},
 year={2018}
}
```
```
@article{zhang2020interpretable,
  title={Interpretable CNNs for Object Classification.},
  author={Zhang, Quanshi and Wang, Xin and Wu, Ying Nian and Zhou, Huilin and Zhu, Song-Chun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020},
  publisher={IEEE}
}
```


