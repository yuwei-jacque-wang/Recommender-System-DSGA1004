# Recommender-System-DSGA1004
A recommender system built in PySpark and designed for HPC environment. Submitted as term project for DS-GA 1004 Big Data at NYU.

For complete instruction and structure of project, please refer to original project description [here](https://github.com/yuwei-jacque-wang/Recommender-System-DSGA1004/blob/master/Instruction.md)

## Description of files:
- Train/Validation/Test splitting of user interactions: ***data.py***
  - Splitting data sets directly from NYU HDFS
  
- Hyperparameter tuning with ALS: 
  - ***grid_search_ranking.py***: Tuning through ranking scores, returns 1. p@k 2. MAP 3. NDCG
  - ***grid_search_rmse.py***: Tuning through root mean square errors, and return rmse
  - Complete results of grid search tuning is shown as ***Rank_Model_Results.pdf*** and ***RMSE_Model_Results.pdf***

- LightFM implementation:
  - LightFM is a package for recommendation system to run on single machine
  - ***LightFM.ipynb***: Implementation of package and functions
  - ***LightFM_Model_Results.pdf***: Tuning and results of LightFM on 1% data
  
- Latent factor exploration
  - ***Visualization_Part_1.ipynb***: Matrix factorization, representation of user matrix and item matrix, and visualization of latent factors
  
- Final report: 
  - [DSGA_1004_FInal_Report.pdf](https://github.com/yuwei-jacque-wang/Recommender-System-DSGA1004/blob/master/DSGA_1004_FInal_Report.pdf)

## Contributors

This project exists thanks to all the people who contribute, especially the three main authors:
- Jacqueline Yuwei Wang [(profile](https://www.linkedin.com/in/jacqueline-yuwei-wang-309665b2/), [GitHub)](https://github.com/yuwei-jacque-wang)
- Yi Xu [(profile)](https://www.linkedin.com/in/goodluckxuyi/)
- Hong Gong [(profile)](https://www.linkedin.com/in/hong-gong/)

And our professor Brian McFee who has provided advising and guidence.
