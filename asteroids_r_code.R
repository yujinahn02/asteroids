#####################################
## Name: Eugenie Ahn
## Final Project
#####################################

#In this project, I will be using an asteroid dataset from NASA (attached to this submission)
#I will be using three types of machine learning models: Random Forest, SVM, and Naive Bayes, and compare which is best at classifying potentially hazardous asteroids, using different evaluation metrics

#downloading necessary packages
library(dplyr) #for data wrangling
library(readr) #to read csv files
library(randomForest) #to make random forest model
library(caret) #for classification
library(e1071) # svm() function
library(ggplot2) #for creating graphs
library(Hmisc) #correlation and p-value
library(corrplot) #creating correlation heatmap
#if it says library not found, run install.packages("library name") and then run code again

##########################################
#DATA PREPROCESSING
##########################################

#read csv file
asteroid <- read_csv("nasa.csv") #change to your directory

#explore the data
head(asteroid) #look at the first five rows
dim(asteroid) #4,687 entries, 40 total columns
summary(asteroid) #summary statistics
str(asteroid) #see data types

#feature engineering: getting rid of columns that do not affect whether an asteroid is potentially hazardous
asteroid <- select(asteroid, #remove unnecessary columns
                   -c(
                     `Neo Reference ID`, #identification whos numbers do not hold value
                     `Orbit ID`, #identification whos numbers do not hold value
                    `Name`, #idenficiation(string)
                    `Close Approach Date`, #tells us when the asteroid will be near earth, not if its hazardous
                    `Epoch Date Close Approach`,  #tells us when the asteroid will be near earth, not if its hazardous
                    `Orbit Determination Date`,  #tells us when the asteroid will be near earth, not if its hazardous
                    `Orbiting Body`, #contains only one value - Earth
                    `Equinox` #contains only one value - J2000
                  ))

#renaming all columns to syntactically-appropriate names as randomForest() does NOT recognize spaces, dashes or brackets
asteroid <- asteroid %>%
  rename(
    absolute_magnitude = `Absolute Magnitude`,
    est_dia_km_max = `Est Dia in KM(max)`,
    est_dia_km_min = `Est Dia in KM(min)`,
    est_dia_m_min = `Est Dia in M(min)`,
    est_dia_m_max = `Est Dia in M(max)`,
    est_dia_miles_min = `Est Dia in Miles(min)`,
    est_dia_miles_max = `Est Dia in Miles(max)`,
    est_dia_feet_min = `Est Dia in Feet(min)`,
    est_dia_feet_max = `Est Dia in Feet(max)`,
    relative_velocity_km_sec = `Relative Velocity km per sec`,
    relative_velocity_km_hour = `Relative Velocity km per hr`,
    miles_per_hour = `Miles per hour`,
    miss_dist_astronomical = `Miss Dist.(Astronomical)`,
    miss_dist_lunar = `Miss Dist.(lunar)`,
    miss_dist_km = `Miss Dist.(kilometers)`,
    miss_dist_miles = `Miss Dist.(miles)`,
    orbit_uncertainty = `Orbit Uncertainity`,
    min_orbit_intersection = `Minimum Orbit Intersection`,
    jupiter_tisserand_invariant = `Jupiter Tisserand Invariant`,
    epoch_osculation = `Epoch Osculation`,
    eccentricity = Eccentricity,
    semi_major_axis = `Semi Major Axis`,
    inclination = Inclination,
    asc_node_longitude = `Asc Node Longitude`,
    orbital_period = `Orbital Period`,
    perihelion_distance = `Perihelion Distance`,
    perihelion_arg = `Perihelion Arg`,
    aphelion_dist = `Aphelion Dist`,
    perihelion_time = `Perihelion Time`,
    mean_anomaly = `Mean Anomaly`,
    mean_motion = `Mean Motion`,
    hazardous = Hazardous
  )

#creating correlation plot
cor_plot <- asteroid %>%
  select(-hazardous) #removing target, non-numerical variable for correlation

corrmat <- cor(cor_plot)
heatmap(corrmat)#the correlation plots is good at highlighting redundancy. 
#I can see that the **est_dia_feet_min**, **est_dia_m_max**, **est_dia_feet_max**, **est_dia_m_min**, **est_dia_km_max**, **est_dia_km_min**, **est_dia_miles_min**, and **est_dia_miles_max** have a correlation of 1, which indicates they they represent the exact same values but in different units. 
#Thus, to reduce redundancy, I will remove everything except for one. 
#Similarly, **miss_dist_miles**, **miss_dist_astronomical**, **miss_dist_km**, and **miss_dist_lunar** also have a correlation of 1, so I will remove 3 out of 4 of them. 
# *miles_per_hour**, **relative_velocity_km_hour**, and **relative_velocity_km_sec** also have a correlation of 1, so I will remove 2 out of 3. 
# **orbital_period**, **semi_major_axis**, and **aphelion_dist** also have a correlation of 1, so I will remove 2 out of 3. 
# **perihelion_time** and **epoch_osculation** also represent the same values so I will only keep one, and **mean_motion** and **jupiter_tisserand_invariant** also represent the same values so I will keep one as well.
#sometimes it shows a correlation map that doesnt include all 31 columns - check heatmap in presentation for full picture

cor(cor_plot) #creating a correlation matrix to ensure the correct correlations

#removing redundant columns
asteroid <- select(asteroid,
                   -c(est_dia_m_max, 
                      est_dia_feet_max,
                      est_dia_m_min,
                      est_dia_km_max,
                      est_dia_km_min,
                      est_dia_miles_min,
                      est_dia_miles_max,
                      miss_dist_astronomical,
                      miss_dist_km,
                      miss_dist_lunar,
                      relative_velocity_km_hour,
                      relative_velocity_km_sec,
                      semi_major_axis,
                      aphelion_dist,
                      epoch_osculation,
                      jupiter_tisserand_invariant
                   ))


#explore target variable
unique(asteroid$hazardous) #pha = flag for whether an asteroid is potentially hazardous or not
# "TRUE" is hazardous, "FALSE" is not hazardous

#creating a bar graph to see the distribution of target variable
asteroid %>%
  ggplot() +
  geom_bar(
    mapping = aes(
      x = hazardous,
      fill = hazardous #we seem to have an imbalanced dataset, which is good, however, since this means our baseline accuracy is #84% and that even if our model is completely wrong, it'll still correctly predict 84% of our classes
    )
  ) #I have to keep in mind that I cannot completely rely on accuracy alone as an evaluation metric, and have to utlize the other metrics dervied from the confision matrix: recall, precision, F-1

##########################################
#ML 1: RANDOM FORESTS
##########################################

#splitting the data into training and testing sets
#4687*0.8 = 3750 (training)
#4687*0.2 = 937 (testing)

asteroid_train <- asteroid[1:3750,] #80% (3750 values) for training
asteroid_test <- asteroid[3751:4678, ] #20% (937 values) for testing

#factoring our target variable (Hazardous) 
asteroid$hazardous <- as.factor(asteroid$hazardous) #telling R that it is a categorical variable

#creating the random forest model
model <- randomForest(as.factor(hazardous)~., 
                      data= asteroid_train,
                      importance=TRUE, 
                      ntree=500, 
                      mtry = 2)

model # see confusion matrix
importance(model)

plot(model) #see errors as number of trees increase
(plot(model)) #see out-of-bag error: measuring the prediction error of random forests using bootstrap aggregation
tail(plot(model)) #see the last few
#we can use the error to represent the lines in the graph, inspecting the output matrix
#bottom (red) line is non-hazardous asteroids (hazardous = FALSE)
#middle (black) line is OOB error
#top (green) line is hazardous asteroids (hazardous = TRUE)
#seems like there is higher error for predicting asteroids that are classified as hazardous
#my overall error is decreasing for increaseing number of trees

#seeing which features are the most important
model %>%
  importance() 

#visualizing importance parameters
varImpPlot(model) #looks like min_orbit_intersection is the most important parameter

#making predictions using test data
pred_test <- predict(model, newdata = asteroid_test, type= "class")
pred_test
confusionMatrix(table(pred_test,asteroid_test$hazardous)) # The prediction to compute the confusion matrix and see the accuracy score 

#validating our model using test data
prediction <- predict(model, asteroid_test, type = "response")
table(prediction, asteroid_test$hazardous) #See confusion matrix
prediction

#comparing predicted vs actual values
values <-cbind(prediction, asteroid_test$hazardous)
values #seeing data
colnames(values)<-c("predicted","actual") #combining the columns
results<-as.data.frame(values) #creating datafram

head(results) #seems like our model did a pretty good job at predicting the actual values

#visualizations
reprtree:::plot.getTree(model)
#seeing the splits of the trees
#you might need to make your "plot" environment bigger to see the full image


#############################
#SVM | 403
############################

#converting target variable into factor
asteroid$hazardous <- factor(asteroid$hazardous, levels = c(FALSE, TRUE))

#split data into training and testing
train_index <- createDataPartition(asteroid$hazardous, p = 0.8, list = FALSE)
train_data <- asteroid[train_index, ]
test_data <- asteroid[-train_index, ]
#for some reason I had to create a different training and testing set for each model - I ran into errors when trying to use the same sets :/

#implemnting the SVM model
svm_model <- svm(hazardous ~ ., 
                 data = train_data,  
                 type = 'C-classification',
                 kernel = "linear")

# add predictions on test data
predictions <- predict(svm_model, newdata = test_data)
conf_matrix <- confusionMatrix(predictions, test_data$hazardous)

#confusion matrix
set.seed(777)
print(conf_matrix) #0.9562 accuracy


##########################
#### Model 3: NAIVE BAYES
#########################

#creating training and testing sets
train_idx <- createDataPartition(asteroid$hazardous, p = 0.8, list = FALSE)
train_data <- asteroid[train_idx, ]
test_data <- asteroid[-train_idx, ]

#implementing naive bayes model
nb_model <- naiveBayes(hazardous ~ ., data = train_data)

# making predictions on test data
predicted <- predict(nb_model, newdata = test_data)

# compute confusion matrix
conf_matrix <- confusionMatrix(data = predicted, 
                               reference = test_data$hazardous)

# print confusion matrix and accuracy
print(conf_matrix) #95.3 accuracy


#####CONCLUSION ######
#I decided that Random Forests was the best performer, as it had the highest recall, f1, and accuracy. 
# A recall of 98.8% means that it was exceptional in correctly predicting hazardous asteroids that were actually classified as hazaroud, while reducing False negatives, which occur when a hwhen a hazardous asteroid is incorrectly classified as non-hazardous. 
# Maximizing recall reduces the occurrence of false negatives, which is crucial for planetary defense and risk mitigation, and could potentially threaten the safety of the earth.
# It also had a high recall of 98.5% which meant it had a good balance between correctly identifying hazardous asteroids and minimizing false alarms. 
# Finally, it had a high accuracy of 99.5, which alone cannot be used to evaluate this model as we had an imbalanced dataset, but, hand in hand with the other two metrics, highlights is ability to classify hazardous asteroids effectively.





