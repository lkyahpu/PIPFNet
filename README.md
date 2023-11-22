# PIPFNet
Polarized Prior Guided Fusion Network for Infrared Polarization Images

## Installation
[Python 3.8]

[Pytorch 1.9.0 ]

[cuda 11.4]

[cudnn 8.4.0]

## Dataset：LDDRS
We use the [ LWIR DoFP Dataset of Road Scene (LDDRS)](https://github.com/polwork/LDDRS) as our experimental dataset.
Download LDDRS dataset from https: //github.com/polwork/LDDRS.
You can randomly assign infrared intensity and polarized images for training and testing in the following directories
```
|-- dataset_all
  |-- train
    |-- S0
       |-- 0000.png
       |-- ....
    |-- dolp
    |-- label
    |-- aop
    |-- s1_s0
  |-- val
    |-- S0
       |-- 0000.png
       |-- ....
    |-- dolp
    |-- label
  |-- test
    |-- S0
       |-- 0000.png
       |-- ....
    |-- dolp
    |-- label
```    

## Train & Test
* After loading data according to the above directory, you can run `python train.py` and `python test.py` for training and testing respectively.

We will add information about the paper later.

## Contact

[Kunyuan Li](mailto:kunyuan@mail.hfut.edu.cn)

