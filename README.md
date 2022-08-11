# Credit_Risk_Analysis
# Project Overview
The objective of this project is to understand how to effectively use Machine Learning algorithms to conclude decisive results on the provided data sets. I used data from LendingClub concentrating on Supervised Learning to analyze credit risk for credit card users. In order to draw accurate conclusions, I used several Machine Learning methods to assess unbalanced classes. In general, good loans far outweigh the risky loans statistically reflected in LendingClub dataset which can lead to an unbalanced classification issue. To offset the imbalance and increase the accuracy of the analysis, I used several different Machine Learning algorithms to retest the data. Some of the different algorithms I deployed are `RandomOverSampler` (oversample), `ClusterCentroids`(undersample), `SMOTEENN`, `SMOTE`, `BalancedRandomForestClassifier`, and`EasyEnsembleClassifier`.

# Resources

* Dataset from LendingClub: [LoanStats_2019Q1.csv](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/LoanStats_2019Q1.csv)
* Software: Python 3.9.7, Anaconda Navigator 1.10.0, Conda 4.13.0 and Jupyter Notebooks 6.4.12

# Results

LoanStats_2019Q1 dataset has 115,675 loan applications. In order to determine whether application is "low" or "high risk", we use the "loan status".This cleans the dataset to total of 68,817 applications and classifies as 

![balancetargetvalue](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/balancetargetvalue.png)


## Resampling Models to Predict Credit Risk

### Oversampling

#### `Naive Random Oversampling Model`

   * The balanced accuracy score is 65%.

![balancedaccuracyscore](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/balancedaccuracy.png)


   * The `high_risk` precision rate about 1% with 62% recall which F1 makes 2% only.
   * The `low risk` precision rate is 100% due to high number of population and recall at 67%.

   ![confusionmatrix](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/confusionmatrix.png)

   ![reportimbalanced](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/reportimbalanced.png)


#### `SMOTE Oversampling Model`

   The results are similar to `Random Oversampling Model` as this creates new value of the closest neighbors to the minority class instead selecting randomly.
  
   * The balanced accuracy score is 63%.

  ![smotescore](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/aa75be481f5273415eceb4eebdc3723bae1f2441/Resources/smotescore.png)

   * The `high_risk` precision rate about 1% with 60% recall which F1 makes 2% again.
   * The `low risk`  precision rate is 100% and recall at 66%.

  ![smotematrix](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/smoteennmatrix.png)
  
  ![smotereport](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/smotereport.png)

### Undersampling

#### `ClusterCentroids Model`
   
  * The balanced accuracy score is down to 53% when compare to oversampling models.

   ![clusterscore](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/clusterscore.png)

  * The `high_risk`precision rate is still at 1% with the recall at 61% which makes F1 score 1%.
  * The `low risk` precision rate is 100% due to high number of false positive and recall at 45%.  

  ![clustermatrix](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/clustermatrix.png)
  
  ![clusterreport](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/clusterreport.png)

## The SMOTEENN algorithm to Predict Credit Risk

### Combination Sampling

#### `SMOTEENN Model` 

Combination of both oversampling and undersampling. This model classifies 68,460 records as `high_risk` and 62,011 as `low risk`

        Counter({'high_risk': 68460, 'low_risk': 62011})

  * The balanced accuracy is about 64%.

  ![smoteennscore](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/smoteennscore.png)
  
  * The `high_risk`precision rate is still at 1% with the recall at 70% which makes F1 score 2%.
  
  * The `low risk` precision rate is still 100% due to same reason and recall at 57%.   
  
  ![smoteennmatrix](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/smoteennmatrix.png)

  ![smoteennreport](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/smoteennreport.png)

##  Ensemble Classifiers to Predict Credit Risk

Compare two new `Machine Learning` models that reduce bias to predict credit risk. The models classified 51,366 as High Risk and 246 as Low Risk.
        
        Counter({'low_risk': 51366, 'high_risk': 246})
        
#### `BalancedRandomForestClassifier Model`

  * The balanced accuracy score improve to 79% for this model.

  ![balancedscore](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/balancedscore.png)

  * The `high_risk`precision rate is low at 3% with the recall at 70% which makes F1 score 6%.
  * The `low risk` precision rate is still 100% and recall at 87%.  

  ![balancedmatrix](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/balancedmatrix.png)
  
  ![balancedreport](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/balancedreport.png)

#### `EasyEnsembleClassifier Model`
  
  * The balanced accuracy score increased to 93% for this model.
  
  ![ensemblescore](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/ensemblescore.png)

  * The `high_risk`precision rate is still low at 9% with the recall at 92% which makes F1 score 16%.
  * The `low risk` precision rate is same as 100% and recall at 94%.  

  ![ensemblematrix](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/ensemblematrix.png)
  
  ![ensemblereport](https://github.com/meliscelikay/Credit_Risk_Analysis/blob/3a8a42e3549cc973eb9c255b22fca915ae45d502/Resources/ensemblereport.png)

# Summary
In conclusion, I believe all the models used illustrated lower than expected accuracy determining high credit risk. The `EasyEnsembleClassifier` model had a 92% accuracy rate for identifying high credit risk, but this model also had a high false positive rate labeling significant amount of low credit risk as high.  It is clear this model is bias and sensitive towards high credit risk and often misclassifies low risk clients, thereby limiting potential new clients and loss of new business opportunities. Although none of the models would satisfy the analysis of this project due to lower than expected accuracy, despite the high amount of false positive rates for identifying low credit risk as high, `EasyEnsembleClassifier` model showed best results. Perhaps if the credit risk data was not low credit risk bias (99%), the Machine Learning algorithms could perform at a higher accuracy rate.  
