# Machine-Learning

http://rpubs.com/Yampa/489854 

---
title: "Practical Machine Learning"
author: "NDANGANG"
date: "21/04/2019"
output: html_document
---
```{r setup, include=FALSE, message=FALSE}
knitr::opts_chunk$set(echo = FALSE)
library(doParallel)
library(readr)
library(caret)
library(dplyr)
library(knitr)

```

## Introduction

This project consists in determining how well an activity was carried by six particpants. These participants were asked to perform barbell lifts correctly and incorrectly in five different ways; exactly according to specification (class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). We splitted the training data set into two parts and constructed a three times repeated 10 cross validation  gradient boosting algorithme on one part of the training data. We then selected the best parameters of n.trees, interaction.depth, shrinkage and n.minobsinnode. We  used these parameters to construct our final model. We tested our model on the remaining part of the training data,and the accuracy was 98%. Finally, we tested our model to the testing data and we obtained 100% accuracy.

## Data

The data from accelometers of the belt,forearm, arm, and dumbell of the  6 participants into the the training and the testing set. The training set contains 19622 obervations and 160 variables. Among these variables, we have records of the accelometers in the x,y and z directions of the belt,forearm, arm, and dumbell. The testing data consist of 20 observations and 160 variables.

## Data Preparation

In the the training data set, we have observed that some columns have missing values. We identified those columns and count the missing the  values.
```{r, results='hold', message=FALSE, warning=FALSE}

# Reading the training data ----
pml_training <- read_csv("C:/Users/Ndangang/Desktop/Data/pml_training.csv")
```

```{r, results='hold', message=FALSE, warning=FALSE}

#searching the colnames containing missing values ----
colNames_empty_values <- colnames(pml_training)[ apply(pml_training, 2, anyNA) ]
```

```{r, results='hold', message=FALSE, warning=FALSE}
#Forming the dataframe of those columns ----
dat_col_emptyvalues <- subset(pml_training, select =colNames_empty_values)

 cnt <- dat_col_emptyvalues %>%
  select(everything()) %>%  
  summarise_all(funs(sum(is.na(.))))
 
kable(cnt)
```
We delete all those columns because the number of missing values in each column is more than three quarter of the number of observations. We repeat the same on procedure on the testing data. We are left with 60 variables.
```{r, results='hold', message=FALSE, warning=FALSE}
#Deleting all the cols of dat_col_emptyvalues from the dataframe pml_training ----

#Colnames without empty values ----
colN <- select(pml_training, -one_of(colNames_empty_values)) %>% colnames

#Data frame containing these column names ----
dat <- subset(pml_training, select =colN)
```

We also delete  variables  "X1", "raw_timestamp_part_1","raw_timestamp_part_2" and "cvtd_timestamp" from our training set and testing set. We are left with 56 variables.

```{r,  results='hold', message=FALSE, warning=FALSE}
#Selection of useful cols ----
col = c("X1", "raw_timestamp_part_1","raw_timestamp_part_2", "cvtd_timestamp")

col_imp <- select(dat, -one_of(col)) %>% colnames

#Forming the dataframe of those columns ----
trainData <- subset(dat, select =col_imp)

#Sta
```

## Pre Processing 
Dued  to the high variances of each of the varaibles in our new training data set, we used the  Pre Process function in the caret package to standardize the training the set. At the same time we standardize the testing  set using the mean and standard deviation of each of the variables in the  training set.
```{r,results='hold', message=FALSE, warning=FALSE}
#Standardizing ----

#select the first two columns----
first_two_cols <- c("classe", "user_name", "new_window")

#Data frame containing the first two columns----
dta_first_two <- subset(trainData, select= first_two_cols)

#train columns ----
trainCols <- select(trainData, -one_of(first_two_cols)) %>% colnames

#training Data ----
sbt_trainData <- subset(trainData, select=trainCols)

#standardizing the training data n_trainData ----
preObj <- preProcess(sbt_trainData, method = c('center', "scale"))
st_trainData <- predict(preObj, sbt_trainData)

#Combining the standardize dataframe and first_two_cols ----
combTrainData <- cbind(dta_first_two,st_trainData)
```

## Splitting the training data set into two different data set
The training data is splitted into two parts. 
```{r,results='hold', message=FALSE, warning=FALSE}
set.seed(3)
intrain = createDataPartition(combTrainData$classe, p=0.7,list=FALSE)
training = combTrainData[ intrain,]
testing = combTrainData[-intrain,]
```

We construct a repeated 3 times 10 fold cross validation on the first part in order to choose the best parameters. 
```{r message=FALSE, warning=FALSE, cache=TRUE, cache.lazy=FALSE, results='hold'}
fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 2)



 gbmFit1 <- train(classe ~ ., data = training, 
                                     method = "gbm", 
                                    trControl = fitControl,
                                    #tuneGrid = gbmGrid,
                                      verbose = FALSE)

 

```

From the graph below, the the repeated cross validation accuracy is high when the max tree depth is 3 and the boosting iterations(number of trees) is 150. 
```{r, results='hold', message=FALSE, warning=FALSE}
ggplot(gbmFit1)

```

The best parameters
```{r, results='hold', message=FALSE, warning=FALSE}

kable(gbmFit1$bestTune)

```

are obtained when the shrinkage= 0.1 and the minimum number of training set samples in a node to commence splitting(n.minobsinnode) is 10.

We fit a new model using the above parameters
```{r, cache=TRUE, results='hold', message=FALSE, warning=FALSE, cache.lazy=FALSE}
gbmGrid <-  expand.grid(interaction.depth = 3, 
                         n.trees = 150, 
                         shrinkage = 0.1,
                         n.minobsinnode = 10)
 
set.seed(800)
gbmFit2 <- train(classe ~ ., data = training, 
                 method = "gbm", 
                 #trControl = fitControl,
                 tuneGrid = gbmGrid,
                 verbose = FALSE)

```



The model is then tested on the second part of the training model. The accuracy of our model is given by; 

```{r, results='hold', message=FALSE, warning=FALSE}
pred2 <- predict(gbmFit2, testing)
confusionMatrix(pred2, as.factor(testing$classe))


```

## Reading and preparation of the testing data

```{r, results='hold', message=FALSE, warning=FALSE}
#Reading of the testing data 
pml_testing <- read_csv("C:/Users/Ndangang/Desktop/Data/pml_testing.csv")
```

We perform the same operations as on the training data set to delete non-important columns.
```{r, cache=TRUE, results='hold', message=FALSE, warning=FALSE}
#Colnames without empty values ----
colN <- select(pml_testing, -one_of(colNames_empty_values)) %>% colnames

#Data frame containing these column names ----
datTest <- subset(pml_testing, select =colN)

#Selection of useful cols ----
col = c("X1", "raw_timestamp_part_1","raw_timestamp_part_2", "cvtd_timestamp")

imp_col <- select(datTest, -one_of(col)) %>% colnames

#Forming the dataframe of those columns ----
testData <- subset(datTest, select =imp_col)

#select the first two columns----
two_cols <- c("user_name", "new_window")

#Data frame containing the first two columns----
dataTest_first_two <- subset(testData, select=two_cols)

#testing cols not in two_cols ----
testCols <- select(testData, -one_of(two_cols)) %>% colnames

#testing Data  with last columns deleted----
sbt_testData <- subset(testData, select=testCols)[,-54]


```

## Standardization of the testing data 

We standardize each column of the testing data set using the mean and the standard deviation of each of the columns of the training data set.

```{r, results='hold', message=FALSE, warning=FALSE}
st_testData <- predict(preObj,sbt_testData)
```

The concatenation of the standardized st_testData and data set dataTest_first_two containing the first two columns. 

```{r,results='hold', message=FALSE, warning=FALSE}
combTestData <- cbind(dataTest_first_two,st_testData)
```

## Prediction on  combTestData data set
```{r, results='hold', message=FALSE, warning=FALSE}
pred_test <- predict(gbmFit2, combTestData)
print(pred_test)
```

## Accuracy 

We applied our machine learning algorithm on the 20 test cases available on the course project prediction quiz and we obtained an accuracy of 100%.


