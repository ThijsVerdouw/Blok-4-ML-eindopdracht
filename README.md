# MADS-exam-24

This year, the junior has learned quit a lot about machine learning.
He is pretty confident he wont make the same mistakes as last year; this year, he has helped you out by doing some data exploration for you, and he even created two models!

However, he didnt learn to hypertune things, and since you are hired as a junior+ datascientist and he has heard you had pretty high grades in the machine learning course, he has asked you to help him out.

## The data
We have two datasets:
### The PTB Diagnostic ECG Database

- Number of Samples: 14552
- Number of Categories: 2
- Sampling Frequency: 125Hz
- Data Source: Physionet's PTB Diagnostic Database

All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 187. There is a target column named "target".

You can find this in `data/heart_train.parq` and `data/heart_test.parq`.

### Arrhythmia Dataset

- Number of Samples: 109446
- Number of Categories: 5
- Sampling Frequency: 125Hz
- Data Source: Physionet's MIT-BIH Arrhythmia Dataset
- Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]
All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 187. There is a target column named "target".

You can find this in `data/heart_big_train.parq` and `data/heart_big_test.parq`.

## Exploration
In `notebooks/01_explore-heart.ipynb` you can find some exploration. It's not pretty, and
he hasnt had time to add a lot of comments, but he thinks it's pretty self explanatory for
someone with your skill level.

## Models
There are two notebooks with models. He has tried two approaches: a 2D approach, and a 1D approach. Yes, he is aware that this is a time series, but he has read that 2D approaches can work well for timeseries, and he has tried it out.

## Task execution:
The notebook called 02_2d-Models contains effectively all functional code and some comments. 
I originally wanted to split things up into seperate python files, but you know. Ray dependancy issues cost me so much time.

