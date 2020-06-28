# Salary Prediction

## Solution strategy (4D framework) for a data science problem
![4D framework](/images/4Dframework.png)
## Define the problem and goal
### The problem
The HR manager of a company wants to assign salaries to different posts within the company at the time of publicly advertising the posts. It is crucial that the salaries are within acceptable ranges for the benefit of the company so that the resources of company are not wasted.

### The goal
The HR manager needs a model which is developed based on an available dataset to help them predict the salary of every post, in order to spend the resources of the company efficiently and create revenue for the company in the long term.
## Discover the data through EDA (Exploratory Data Analysis)
#### Step 1: Examining and high level overviewing the data (slides 5 - 6 of the report.pdf)
#### Step 2: Inspect more detail of the dataset (slides 7 - 8 of the report.pdf)
* Total number of records: 1,000,000
* The object columns are categorical.
* The integer columns are numeric.
#### Step 3: Checking for duplicates (slide 9 of the report.pdf)
This dataframe has no duplicates.
#### Step 4: Identification of numerical and categorical features (slide 10 of the report.pdf)
* Numeric columns: ‘yearsExperience’, ‘milesFromMetropolis’, ‘salary’
* Categorical columns: ‘jobId’, ‘companyId’, ‘jobType’, ‘degree’, ‘major’, ‘industry’
#### Step 5: Summarising numerical and categorical variables separately (slide 11 of the report.pdf)
#### Step 6: Merging features and target of training dataframe into a single dataframe (slide 12 of the report.pdf)
#### Step 7: Visualising the target variable (Salary) (slide 13 - 14 of the report.pdf)
![Salary visualisation](/images/visualising the salary.png)

## Develop models
## Deploy the best model
## Next steps
