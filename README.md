# Dogs vs Cats
Predict whether a given image is a cat or a dog with 99.7% accuracy. 

Kaggle: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition

Partially based on [Fast.ai](http://course.fast.ai/) course.

# Intro
- [ ] TODO: The goals
- [ ] TODO: Though process and methods in creating the project
- [ ] TODO: Some bullet points with interesting observations
- [ ] TODO: Any interesting charts or diagrams
## Model info
Final prediction is made from an ensemble of Xception, ResNet50, Inception-ResNetV2.

Training is done using bottlenecks on a data augmented [x6] training set (138000 299x299x3 pictures).

Precomputed bottleneck features are feeded into 2 hidden dense layers [x2048] and 1 output softmax layer.

### Size
Training set size = 2.5GB. Model size = 1GB per model (saved precomputed weights).

### Speed
Time to precompute bottlenecks: 2hr per model on a GTX 770.

Time to train: 5sec/epoch with 10000 batch size.


### Error rates
Ensemble reaches 99.7% accuracy on the validation set (only 6/2000 incorrect).

0.03893 leaderboard score **(15th place from 1,314 teams)**.

# Requirements
* keras==2.0.8
* tensorflow==1.2.0
* pandas
* seaborn
* sklearn

# Data
## Download the data
`pip install kaggle-cli`

`kg config -g -u <username> -p <password> -c dogs-vs-cats-redux-kernels-edition`

`kg download`

`7z x *.zip`

I split the data into 23000 train and 2000 validation sets.
## File structure
```
dogs-vs-cats-redux
├── data ─── dogscats
│             ├── train
│             │    ├── cats
│             │    └── dogs
│             ├── valid
│             │    ├── cats
│             │    └── dogs
│             ├── test
│             │    └── unknown
│             ├── models
│             └── results
├── submissions
│   └── submission1.csv
├── dogs_cats_redux_ensemble.ipynb
└── README.md
```
# Usage
`> jupyter notebook`

`dogs_cats_redux_ensemble.ipynb` for final model
