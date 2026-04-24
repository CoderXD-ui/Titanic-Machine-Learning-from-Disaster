Titanic Passanger Survival Analysis.
The sinking of the RMS Titanic in 1912 remains one of the most infamous maritime disasters in history,
leading to significant loss of life. Over 1,500 passengers and crew perished that fateful night.
Understanding the factors that contributed to survival can provide valuable insights into safety protocols and
social dynamics during crises. In this project, we will leverage machine learning techniques to predict the
survival chances of Titanic passengers based on various features, such as sex, age, and passenger class.
Using the Random Forest classification algorithm, we aim to build a predictive model that will allow us 
to estimate the likelihood of survival for each individual aboard the Titanic.
Objective of the Project: Predicting Titanic Passenger Survival
The primary objective of this project is to develop a machine learning model capable of predicting the survival status of 
Titanic passengers based on available data. The dataset includes information such as demographic attributes (age, sex), 
socioeconomic status (fare, class), and other relevant features. By analyzing these features, we seek to identify patterns 
that could influence survival rates and subsequently use these insights to make predictions on unseen data.
There will be three main steps in this experiment:
@Feature Engineering
@Imputation
@Training and Prediction
@For this project, we will utilize the Titanic dataset. The dataset consists of the following files:
@train.csv: Contains information about the passengers and their survival status, which will be used for
@training our model. Serves as our primary data source for training and validation, providing both features and target labels.
@test.csv: Includes details of passengers without survival labels, which we will use for making predictions.
@Allows us to assess the model's performance on unseen data, simulating a real-world scenario where predictions must be 
@made for new passengers.@gender_submission.csv: A sample submission file that demonstrates the format required for 
@submitting predictions.

<h2>Titanic Passanger Survival Analysis</h2>



```python
from IPython.display import Image
Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/5095eabce4b06cb305058603/5095eabce4b02d37bef4c24c/1352002236895/100_anniversary_titanic_sinking_by_esai8mellows-d4xbme8.jpg")
```
<img src="https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/5095eabce4b06cb305058603/5095eabce4b02d37bef4c24c/1352002236895/100_anniversary_titanic_sinking_by_esai8mellows-d4xbme8.jpg"/>

