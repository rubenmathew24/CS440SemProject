## ISSUES I HAD WITH WINDOWS

Make sure to add these to the environment before running to remove lots of warnings:

```
set LOKY_MAX_CPU_COUNT=8
set OMP_NUM_THREADS=1
set CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## CONFIGURATION

You can open config.py to change the parameters of the models.

## DATASET CREATION

First need to create the datasets we are running with. Can't use the entire Amazon dataset because it is very large (470+ GBs of data)

### How to use dataset_creation.py

`python dataset_creation.py` Downloads and caches the entire Amazon dataset. Do this first if making changes to the datasets. If it is already cached, shouldn't take longer than 2-3 min to verify.

`python dataset_creation.py --partition` This creates 3 partitions of the dataset (large, medium, small) with an 80/20 train/test split. 

`python dataset_creation.py --partition <SIZE>` This creates just the <SIZE> partition of the dataset. Can be small, medium, or large

`python dataset_creation.py --stats <SIZE>` This presents just the statistics of the <SIZE> dataset. Can be small, medium, or large.


## BASELINE MODEL

Now that the datasets are created (should be in data folder), we can run the baseline model. Uses traditional finetuning for text classification with binary and multi classification

### How to use baseline.py

`python baseline.py` Creates the baseline model (binary and multi) for all dataset sizes.

`python baseline.py --size <SIZE>` Creates the baseline model (binary and multi) for the specified SIZE.

`python baseline.py --type <TYPE>` Creates the baseline model (binary __OR__ multi) for all dataset sizes. Binary/Multi based on value of TYPE.

`python baseline.py --type <TYPE> --size <SIZE>` Creates the specific baseline model based on the specified TYPE AND SIZE.


## ACTIVE LEARNING MODELS

We can now run the active learning models. AL1 (uses Least Confidence) and AL2 (uses Entropy + KMeans).

### How to use active_learning.py

`python active_learning.py` Creates the active learning model (binary and multi) for all dataset sizes and learning types.

`python active_learning.py --size <SIZE>` Creates the active learning model (binary and multi) for the specified SIZE.

`python active_learning.py --type <TYPE>` Creates the active learning model (binary __OR__ multi) for all dataset sizes and learning methods. Binary/Multi based on value of TYPE.

`python active_learning.py --learning <LEARNING>` Creates the active learning model (AL1 __OR__ AL2) for all dataset sizes and tasks. AL1/AL2 based on value of LEARNING.

`python active_learning.py --type <TYPE> --size <SIZE> --learning <LEARNING>` Creates the specific active learning model based on the specified TYPE, SIZE, and LEARNING method.

## GRAPHS

We can create graphs for our results.

`python results.py` Creates all the graphs.

## MANUAL TESTING

If you want to manually test one of the models generated, you can use the following command:

`python test.py --size <SIZE> --type <TYPE> --learning <LEARNING>` This will run the specified model with the specified SIZE, TYPE, and LEARNING method.