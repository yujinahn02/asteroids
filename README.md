# Classifying Hazardous Asteroids with Machine Learning

In this project, I developed three different Machine Learning Models (Random Forests Support Vector Machine (SVM), and Naive Bayes) to classify potentially hazardous asteroids using the [NASA Asteroids dataset](http://neo.jpl.nasa.gov/). This project was coded in R/RStudio and a copy of the code can be found [here](https://github.com/yujinahn02/asteroids/blob/main/asteroids_r_code.R).

## Dataset
The asteroids classification dataset was published in 2018 by NASA, and includes information on 4687 different asteroids and 40 characteristics that describe them, such as their distance to the earth and relative velocity. I had two types of variables, numerical and categorical. My target output variable was the “hazardous” column, which included values of TRUE and FALSE. After creating a boxplot of my target variable, I realized I had an imbalanced class.
