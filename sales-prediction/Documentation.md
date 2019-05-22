**The clear step-by-step instruction on how to produce the final submit file:**
1. Download competition data and preprocess it with [DataPreparation](https://github.com/polakowo/mymlprojects/blob/master/sales-prediction/DataPreparation.ipynb), which outputs data in a HDF5 format.
2. Run [LightGBM](https://github.com/polakowo/mymlprojects/blob/master/sales-prediction/LightGBM.ipynb) to produce first base-model meta features and predictions. While importing the preprocessed data, pay attention to the structure of folders with input files, since the notebooks were downloaded directly from Kaggle. Similarly run the [CatBoost](https://github.com/polakowo/mymlprojects/blob/master/sales-prediction/CatBoost.ipynb), [LinReg](https://github.com/polakowo/mymlprojects/blob/master/sales-prediction/LinReg.ipynb) and [NeuralNet](https://github.com/polakowo/mymlprojects/blob/master/sales-prediction/NeuralNet.ipynb) notebooks.
4. Run [Blending](https://github.com/polakowo/mymlprojects/blob/master/sales-prediction/Blending.ipynb), which takes outputs of the base models and generates CSV files for submission.
5. (optional) For fastest use, upload all notebooks to Kaggle and import the competition data and the respective outputs from other kernels. In Kaggle, data preparation runs in 15min, base-model notebooks run in 2 hours if done in parallel, ensembling runs in 5min.

**Summary**:
- We used Gradient Boosting Models such as LightGBM and CatBoost, Linear Regression (Vowpal Wabbit), Neural Networks (fastai), and ensemble methods such as bagging of LGBM models and blending of all base models. For non-tree-based models such as Linear Regression and Neural Networks standard scaling across all features was used. The most relevant features were selected based on the importance scores of the fitted LGBM models. Since the data contains a time variable, we had to respect it in the validation scheme: we used 33th month as validation set for first-level models, and a custom validation scheme similar to TimeSeriesSplit for second-layer models.

**DataPreparation:**
- When exploring the data we identified the discrepancy between training and test distributions. The test set is a cartesian product of shop and item ids within the 34 date block. There are 5100 items * 42 shops = 214200 pairs. To make both sets similar, for each date block in train, we created a product of shops and items which produced a nearly sparse matrix. Furthermore, we aggregated daily sales to monthly sales. We also clipped the target to *[0, 20]* range, which is similarly done by the competition's evaluation mechanism. 
- The preprocessing on text features included several fixes such as duplicate removal, extraction of item types and subtypes, and extraction of the city name. 
- The most influential variables generated were target encodings: we aggregated sales across categorical features and their combinations. We also tried default KFold, expanding, and leave-one-out strategies, but encountered drops in leaderboard scoring, so we skipped them altogether.
- In order for the base models to capture information about past and generalize better, we also introduced a set of lagged features, with small windows being the most benefial. 
- As part of the advanced topics, we utilized matrix decomposition (and dimensionality reduction) in order to generate new features. We processed numeric columns with either PCA or TruncatedSVD (best suited for sparse data), and categorical columns with NMF (best suited for tree-based models). The non-linear transformer t-SNE required too much time and resources so we skipped it. 
- Then we saved the data to the disk for use by other kernels and produced two baseline submissions: 1) global mean and 2) previous month benchmark. Both submissions scored the same LB score as stated by the instructors, giving us the validation for the correctness of the preprocessing steps.
- About RAM optimization: Having millions of records and dozens of features in the dataset we often encountered memory errors, so we downcasted the dataframe to the smallest numeric datatype to safe memory. We'd like to note that *float16* produced strange results since it is less widely supported than *float32*, thus we used the latter.

**LightGBM:**
- 

**CatBoost:**
- 

**LinReg:**
-

**NeuralNet:**
-

**Blending:**
-
