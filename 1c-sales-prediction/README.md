<img width=100 src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/1C_Company_logo.svg/1200px-1C_Company_logo.svg.png"/>

#### Forecasting total sales for every product and store in the next month

Sales forecasting plays a major role in a company's success. The goal of this project was to predict total sales for every product and store in the next month. This challenging time-series problem was tackled by using GBMs, linear models, neural networks, and ensembles. The project was part of the [Predict Future Sales competition](https://www.kaggle.com/c/competitive-data-science-predict-future-sales).

Tags: *Competition, Sales Forecasting, Time Series, Neural Networks, Python, LGBM, CatBoost, Vowpal Wabbit, sklearn, fastai, Ensembling*

The clear step-by-step instruction on how to produce the final submit file:
1. Download competition data and preprocess it with [DataPreparation](https://nbviewer.jupyter.org/github/polakowo/mlprojects/blob/master/1c-sales-prediction/DataPreparation.ipynb), which outputs data in a HDF5 format.
2. Run [LightGBM](https://nbviewer.jupyter.org/github/polakowo/mlprojects/blob/master/1c-sales-prediction/LightGBM.ipynb) to produce first base-model meta features and predictions. While importing the preprocessed data, pay attention to the structure of folders with input files, since the notebooks were downloaded directly from Kaggle. Similarly run the [CatBoost](https://nbviewer.jupyter.org/github/polakowo/mlprojects/blob/master/1c-sales-prediction/CatBoost.ipynb), [LinReg](https://nbviewer.jupyter.org/github/polakowo/mlprojects/blob/master/1c-sales-prediction/LinReg.ipynb) and [NeuralNet](https://nbviewer.jupyter.org/github/polakowo/mlprojects/blob/master/1c-sales-prediction/NeuralNet.ipynb) notebooks.
4. Run [Stacking](https://nbviewer.jupyter.org/github/polakowo/mlprojects/blob/master/1c-sales-prediction/Stacking.ipynb), which takes outputs of the base models and generates CSV files for submission.
5. (optional) For fastest use, upload all notebooks to Kaggle and import the competition data and the respective outputs from other kernels.

For more details, proceed to [Documentation](https://github.com/polakowo/mlprojects/blob/master/1c-sales-prediction/Documentation.md)
