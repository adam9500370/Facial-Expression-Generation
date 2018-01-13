# Facial Expression Generation

## Introdution
To synthesize images with different emotions for certain person
by multi-domain image-to-image emotion transfer on FER2013 dataset,
and even emotion transfer from real human to virtual character on FER2013 and FERG-DB datasets.

Our contribution can be listed as followings:
* Perceptual loss in D
* Skip (residual) connection in G
* Some GAN tricks (Dropout + LReLu in G)

Implementation details and qualitative results please refer to [our report](https://github.com/adam9500370/facial_expression_generation/blob/master/report/team3-facial-expression-generation.pdf).


## Getting Started
1. Download [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) and [FERG-DB](http://grail.cs.washington.edu/projects/deepexpr/ferg-db.html) datasets

2. Use `python convert_fer2013.py` to convert csv file to images with certain folder structure,
   and FERG-DB also need to create the folder structure as fer2013 one.

3. Train our modified StarGAN
```
bash run_train.sh
bash run_train_both.sh
```

3. Test our modified StarGAN
```
bash run_test.sh
bash run_test_both.sh
```

4. Facial expression recognition evaluation on generated images from FER2013 test dataset
   by training ResNet18 classifier on FER2013 training set.
```
cd eval_code
python run_eval_train.sh
python run_eval_test.sh
```


## References
[1] [PyTorch-StarGAN](https://github.com/yunjey/StarGAN)

[2] [Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)
