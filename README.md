## ISSUES I HAD WITH WINDOWS

Make sure to add these to the environment before running to remove lots of warnings:

```
set LOKY_MAX_CPU_COUNT=8
set OMP_NUM_THREADS=1
set CUBLAS_WORKSPACE_CONFIG=:4096:8
```

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

`python baseline.py --type <TYPE> --size <SIZE>` Creates the specific baseline model based on the specified TYPE and SIZE.