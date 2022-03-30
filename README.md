# Product Matching Project
Find match between products in different datasets.

## Code

#### Experiments:
The functioning code for finding the matches using **approach 0** is in `product-matching-EXP0.ipynb`.  
The functioning code for finding the matches using **approach 1** is in `product-matching-EXP1.ipynb`.  

#### Utils and additional scripts:
Please find additional scripts for importing utils functions for metrics and data exploration in `utils/utils-v0.1.py` (working for EXP0) and `utils/utils.py` (working for EXP1).  
Please find a working util script for dowloading images from URLs, merging them and automatically generate palettes of color and metrics of comparison in `utils/img-manager.py`.

#### Results:
Results are available in the folder `/Predictions`.  
Along with screenshoot of the metrics, you can find predictions dataset for training and testing (unseen) data, for both EXP0 and EXP1.  
Training data predictions EXP0: `Predictions/predictions_training_EXP0.parquet`  
Training data predictions EXP1: `Predictions/predictions_training_EXP1.parquet`  
Test data predictions EXP0: `Predictions/predictions_testing_EXP0.parquet`  
Test data predictions EXP1: `Predictions/predictions_testing_EXP1.parquet`  


