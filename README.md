# Salary Prediction

## Solution strategy (4D framework) for the data science problem
<img src="/images/4Dframework.png" width = 1000>  

## 1- Define the problem and goal
### The problem
The HR manager of a company wants to assign salaries to different posts within the company at the time of publicly advertising the posts. It is crucial that the salaries are within acceptable ranges for the benefit of the company so that the resources of company are not wasted.
  
### The goal
The HR manager needs a model which is developed based on an available dataset to help them predict the salary of every post, in order to spend the resources of the company efficiently and create revenue for the company in the long term.  
  
## 2- Discover the data through EDA (Exploratory Data Analysis)
### Step 1: Examining and high level overviewing the data
<img src="/images/[4].png" width = 800>

* The column ‘jobId’ seems to be a randomly generated number and hence unique.
* The columns ‘companyId’, ‘jobType’, ‘degree’, ‘major’ and ‘industry’ seem to be categorical columns since there are some repetitions. Therefore, the number of unique categories for each column are expected to be much less than the total number of records.

<img src="/images/[5].png" width = 800>

* It can be seen that the test dataset has the same features as the training dataset. Hence, there is no need to delete any features from the training set.

### Step 2: Inspect more detail of the dataset
<img src="/images/[6].png" width = 800>

The training dataset has 1,000,000 entries.

As was expected:

* Categorical columns:
  * ‘companyId’
  * ‘jobType’
  * ‘degree’
  * ‘major’
  * ‘industry’
* Numerical columns:
  * 'jobId'
  * 'yearsExperience'
  * 'milesFromMetropolis'
  * 'salary'

<img src="/images/[7].png" width = 800>

The test dataset has 1,000,000 entries, with the same columns as the training dataset.

### Step 3: Checking for duplicates and NaN values
<img src="/images/[9].png" width = 800>

Both the training dataset and the test dataset have no duplicates.

<img src="/images/[11].png" width = 800>

Both the training dataset and the test dataset have no NaN values, and hence it seems they are nice and clean dataframes.

In case of presence of any duplicates or NaN values, this needed to be investigated further. It should be decided how to handle them, whether or not they should be dropped or be replaced by zero values.

### Step 4: Summarising numerical and categorical variables separately
<img src="/images/[13].png" width = 800>

The numerical columns have reasonable values and ranges, and there is no need to handle any unexpected data.

<img src="/images/[14].png" width = 800>

* 'jobId' is unique.
* The rest of the objects are categorical columns.

### Step 5: Visualising the target variable (Salary)
<img src="/images/[15].png" width = 800>

* The potential outliers need further investigation.
* The ‘salary’ follows similar to a Normal distribution, with a mean around 120.

### Step 6: Dealing with the outliers

#### Step 6.1: Using Interquartile Range (IQR) rule to identify potential outliers
<img src="/images/[16].png" width = 800>
Hence, any data entry that has the salary below 8.5 or above 220.5 is considered as potential outlier and needs to be further investigated.

#### Step 6.2: Examining potential outliers
This is performed to learn if these outliers are missing, meaningful, and whether to include them in the training set or should they be excluded.

**Outliers below the lower bound**
<img src="/images/[17].png" width = 800>

Examining these outliers shows that these are instances of missing/corrupt data, since candidates with doctoral or masters degree, in oil, web or finance industries, with some years of experience should earn a salary. Therefore, these will be removed from the dataset later.

**Outliers above the upper bound**
<img src="/images/[18].png" width = 800>

All these are senior level roles, all the way from CEO down to SENIOR, and it is reasonable that they earn lot of money. However, why candidates in JUNIOR roles should earn high salaries? This needs further investigation to make sure if it is genuine and correct data or should these be dropped from the dataset.

<img src="/images/[19].png" width = 800>

Inspection of these suspicious entries shows that these are primarily in oil, finance and web industries. Also, most of them have advanced degrees. Hence, these are genuine correct data since these industries have high salaries even though in their Junior roles. Hence, these data are true outliers and will not be dropped from the dataset.

#### Step 6.3: Removing data with zero salaries
<img src="/images/[20].png" width = 800>

### Step 7: Plot all the features separately
**'companyId'**

<img src="/images/[21].png" width = 800>

**Conclusion:** The straight line shows that this feature is not predictive of the target, meaning the salaries are very weakly associated with these randomly generated numbers.

**'jobType'**

<img src="/images/[22].png" width = 800>

**Conclusion:** This shows that the higher the role, the more salary the person is earning, which is totally reasonable and believable. This feature has definitely impact on the target (salary) and hence will be kept as one of the features in the training dataset.

**'degree'**

<img src="/images/[23].png" width = 800>

**Conclusion:** This shows that the more the degree level, the more salary the person is earning, which is completely acceptable and makes sense. This feature has definitely impact on the target (salary) and hence will be kept as one of the features in the training dataset.

**'major'**

<img src="/images/[24].png" width = 800>

**Conclusion:** Apart from people with NONE degree, the rest of the degrees are earning pretty much the same salary. However, the salaries of engineering, business and maths are slightly higher than other majors. This feature has impact to some extent on the target (salary) and hence will be kept as one of the features in the training dataset.

**'industry'**

<img src="/images/[25].png" width = 800>

**Conclusion:** These 2 observations are reasonable and rational. This shows, as expected, that the oil, finance and web has highest salaries. Also, this shows that different industries have different salaries. This feature has definitely impact on the target (salary) and hence will be kept as one of the features in the training dataset.

**'yearsExperience'**

<img src="/images/[26].png" width = 800>

**Conclusion:** This shows, as expected, that the more experience the person has, the more salary they earn, which is completely genuine and logical. This feature has definitely impact on the target (salary) and hence will be kept as one of the features in the training dataset.

**'milesFromMetropolis'**

<img src="/images/[27].png" width = 800>

**Conclusion:** This shows an inverse relationship between the miles from Metropolis and the salary, which makes perfect sense. This feature has definitely impact on the target (salary) and hence will be kept as one of the features in the training dataset.

### Step 8: Identification of correlation between all the features and target
<img src="/images/[29].png" width = 1000>

**Conclusions:**
* The target variable ('salary') is correlated to all the features except 'companyId'. The correlation with 'companyId' is very weak compared to the rest of the features.
* There is some degree of correlation between 'major' and 'degree'.


![map of features](/images/featuresMap.png)  
As was expected from Step 9, all the features except 'companyId' have impact/correlation with the target, i.e. salary.

## Develop models
### Feature engineering (slide 33 of the report.pdf)
In order to boost the performance of the models, all the salaries that had the same ‘companyId’, ‘jobType’, ‘degree’, ‘major’, and ‘industry’ were grouped. Then, for each group, the mean, minimum, maximum, standard deviation and median was calculated and added to the training dataset.
### Selection of the models (slide 34 of the report.pdf)
* Three regression models were selected to predict the target variable (salary), based on all the features (including the features engineered):
  * Linear Regression
  * Random Forest Regressor
  * Gradient Boosting Regressor
* The Mean Squared Error (MSE) was used as the metric to evaluate and compare the performance of these three models.
### Performance of the models (slide 35 of the report.pdf)
The value of Mean Squared Error (MSE) for the selected models are:
* Linear Regression: 358.17
* Random Forest Regressor: 313.63
* Gradient Boosting Regressor: 313.10

## Deploy the best model
The **Gradient Boosting Regressor** with the value of MSE of 313.10 is considered as the best model.
![important features](/images/importantFeatures.png)  
The following features are the key predictors (have the most importance/impact on the value of the salary):
* group_mean
* yearsExperience
* milesFromMetropolis

## Model improvement
* Feature engineering can be extended to also consider ‘yearsExperience’ and ‘milesFromMetropolis’.
* The performance of the models could have been improved having further features in the original dataset, i.e. ‘recruitType’ (‘contract’ or ‘permanent’), ‘contractType’ (‘full-time’, ‘part-time’), etc.
