# Salary Prediction

## Solution strategy (4D framework) for the data science problem
<img src="/images/4Dframework.png" width = 1000>  

## Define the problem and goal
### The problem
The HR manager of a company wants to assign salaries to different posts within the company at the time of publicly advertising the posts. It is crucial that the salaries are within acceptable ranges for the benefit of the company so that the resources of company are not wasted.

### The goal
The HR manager needs a model which is developed based on an available dataset to help them predict the salary of every post, in order to spend the resources of the company efficiently and create revenue for the company in the long term.\n TEST
## Discover the data through EDA (Exploratory Data Analysis)
### Step 1: Examining and high level overviewing the data (slides 5 - 6 of the report.pdf)
### Step 2: Inspect more detail of the dataset (slides 7 - 8 of the report.pdf)
* Total number of records: 1,000,000
* The object columns are categorical.
* The integer columns are numeric.
### Step 3: Checking for duplicates (slide 9 of the report.pdf)
This dataframe has no duplicates.
### Step 4: Identification of numerical and categorical features (slide 10 of the report.pdf)
* Numeric columns: ‘yearsExperience’, ‘milesFromMetropolis’, ‘salary’
* Categorical columns: ‘jobId’, ‘companyId’, ‘jobType’, ‘degree’, ‘major’, ‘industry’
### Step 5: Summarising numerical and categorical variables separately (slide 11 of the report.pdf)
### Step 6: Merging features and target of training dataframe into a single dataframe (slide 12 of the report.pdf)
### Step 7: Visualising the target variable (Salary) (slides 13 - 14 of the report.pdf)
![Salary visualisation](/images/salaryVisualisation.png)
### Step 8: Dealing with the outliers (slides 16 - 22 of the report.pdf)
* Outliers on the left (zero salaries) are missing/corrupt data, and hence were excluded from the dataframe.
* Inspection of the outliers on the right (high salaries) shows that these are primarily in oil, finance and web industries. Also, most of them have advanced degrees. Hence, these are genuine correct data since these industries have high salaries even though in their Junior roles. Hence, these data are true outliers and will not be dropped from the dataset.
### Step 9: Plot all the features separately (slides 23 - 30 of the report.pdf)
![Salary plot](/images/SalaryPlot.png)  
It can be observed that all the features except 'companyId' has a degree of correlation with 'salary' and are predictors of it. Hence, 'companyId' is excluded from the training dataset.
### Step 10: Identification of correlation between all the features and target (slides 31 - 32 of the report.pdf)
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
