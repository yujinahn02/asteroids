# Classifying Hazardous Asteroids with Machine Learning

In this project, I developed three different Machine Learning Models (Random Forests Support Vector Machine (SVM), and Naive Bayes) to classify potentially hazardous asteroids using the [NASA Asteroids dataset](http://neo.jpl.nasa.gov/). I also compared these three models using evaluation metrics from the confusion matrix - accuracy, precision, recall, and F1, to see which model was the best at accurately predicting hazardous asteroids. This project was coded in R/RStudio and a copy of the code can be found [here](https://github.com/yujinahn02/asteroids/blob/main/asteroids_r_code.R).

## Dataset
The asteroids classification dataset was published in 2018 by NASA, and includes information on 4687 different asteroids and 40 characteristics that describe them, such as their distance to the earth and relative velocity. I had two types of variables, numerical and categorical. My target output variable was the “hazardous” column, which included values of TRUE and FALSE. After creating a boxplot of my target variable, I realized I had an imbalanced class. ![imbalanced class](imbalancedclass.png) My target variable is an imbalanced class, which means that there are significantly more asteroids classified as non-hazardous than hazardous. This was important to remember because it means the baseline accuracy is 84%, ***which means that even if I create a completely incorrect model, it will still accurately predict hazardous asteroids 84% of the time***. As a result, I kept this in mind when evaluating my model using evaluation metrics, reminding myself that accuracy alone cannot be used to evaluate my model's performance.

## Methodology
When I decided to complete this project, I realized very early on that it would be a lengthy one, and thus, I should create a detailed roadmap on how I was going to complete it. My general plan was to follow the analysis flowchart below, to ultimately create three different ML models and assess them using various evaluation metrics to determine which one was the best at classifying potential hazardous asteroids. <br/> ![image](flowchart.png)

## Data Preprocessing
Preparing data for analysis always takes up the most, and it was no different for this project. I reviewed the structure of my data - the number of rows and columns, summary statistics - and ensured that there were no missing values. Performing Exploratory Data Anlysis (EDA) allowed me to see the underlying relationships between the data, preventing surprises later on. Many of my columns also contained spaces and capital letters, so I manually converted them to lowercase and replaced the spaces with underscores after discovering, through trial and error, that the randomforest() function does not recognize them.
I then used feature engineering to remove columns that were unnecessary in determining whether an asteroid is hazardous or not. Finally, I explored my target variable as shown in the bar plot before and found out it had a logical, binary form. <br/>

To emphasize my feature engineering, I created a correlation matrix, and then a plot to highlight redundancy within the data. As you can see, est_dia_feet_min, est_dia_feet_max, est_dia_km_max, and est_dia_miles_max had a correlation of 1, meaning that they represented the same values but in different units of measurement. I removed all of these and kept just one to remove redundancy.

## Model 1: Random Forests


