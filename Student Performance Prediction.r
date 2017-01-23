
#Author - Ashwin Kumar Vajantri
#Version - 1.0
#Date - 12/2/2016
#File name - Student Performance Prediction.R

#Clearing the environment
rm(list=ls())

#setting the working directory
setwd("C:/ind prj")

#including the libraries
library(ggplot2)
library(gridExtra)
library(dplyr)
library(car)
library(caret)
library(leaps)
library(bestglm)
library(e1071)
library(ROCR)
library(boot)
library(C50)
library(rpart)

#reading the data sets
mat_data <- read.csv("student-mat.csv",sep=";",header=TRUE)
port_data <- read.csv("student-por.csv",sep=";",header=TRUE)



#Boxplots for final grade  distribution according to family relation quality
boxplot(G3~famrel,data=port_data, col = (c("yellow","green","orange","violet","red")) ,main="final grade distribution according to family relation", 
        xlab="quality of family relationships", ylab="final grade")

#Boxplots for final grade  distribution according to sex 
boxplot(G3~sex,data=port_data, col = (c("blue","green")) ,main="final grade distribution according to sex", 
        xlab="sex", ylab="final grade")


#histogram of the training data for final grade distribution
hist(port_data$G3)

#introduction of a new variable called "res" for the prediction
#training data
port_data_train <- port_data
port_data_train$res <- NULL
port_data_train$res <- 
  factor(ifelse(port_data_train$G3 >= 8, 1, 0),labels = c("fail", "pass"))


#testing data
mat_data_test <- mat_data
mat_data_test$res <- NULL
mat_data_test$res <- 
  factor(ifelse(mat_data_test$G3 >= 8, 1, 0),labels = c("fail", "pass"))

#creation of the subset of the training and the test dataset
port_dataset <- select(port_data_train, school ,sex, G1, G2, Mjob, Fjob, goout,
                       absences, reason, Fjob, Mjob, failures, Fedu, Medu, res)


mat_dataset <- select(mat_data_test, school ,sex, G1, G2, Mjob, Fjob, goout,
                       absences, reason, Fjob, Mjob, failures, Fedu, Medu, res)

#normalization function
normdata <- function(a) {(a - min(a, na.rm=TRUE))/(max(a,na.rm=TRUE) -
                                                 min(a, na.rm=TRUE))}



#Normalizing all the numeric variables in both the subsets of data
port_dataset$G1 <- normdata(port_dataset$G1)
port_dataset$G2 <- normdata(port_dataset$G2)
port_dataset$goout <- normdata(port_dataset$goout)
port_dataset$absences <- normdata(port_dataset$absences)
port_dataset$failures <- normdata(port_dataset$failures)
port_dataset$Fedu <- normdata(port_dataset$Fedu)
port_dataset$Medu <- normdata(port_dataset$Medu)

mat_dataset$G1 <- normdata(mat_dataset$G1)
mat_dataset$G2 <- normdata(mat_dataset$G2)
mat_dataset$goout <- normdata(mat_dataset$goout)
mat_dataset$absences <- normdata(mat_dataset$absences)
mat_dataset$failures <- normdata(mat_dataset$failures)
mat_dataset$Fedu <- normdata(mat_dataset$Fedu)
mat_dataset$Medu <- normdata(mat_dataset$Medu)



#applying decision tree using C.50 algorithm
  m <- C5.0(x = port_dataset[-13], y = port_dataset$res) 
summary(m)


#fitting a logistic regression model
tr_fit1 <- glm(res ~  G2  + goout + reason + Medu ,  data = port_dataset, family = quasibinomial())
summary(tr_fit1)

tr_fit2 <- glm(res ~  G2  + goout + reason , data = port_dataset, family = quasibinomial() )
summary(tr_fit2)

# comparsion of models using the analysis of deviance table
anova(tr_fit1,tr_fit2,test="F")

tr_fit3 <- glm(res ~  G2  + goout, data = port_dataset, family = quasibinomial())
anova(tr_fit2,tr_fit3,test="F")



#calculating probabilities for confusion matrix for test data set using the training fit
prob_test <- predict(tr_fit2,  mat_dataset, type = "response")  
pred_test <- rep("fail", 395)  
pred_test[prob_test > 0.5] = "pass"  

#calculating the confusion matrix for the test subset using the training fit
confusionMatrix(table(pred_test, mat_dataset$res), positive = "pass")


#predicting the performance of the test data to calculate the area under curve by plotting 
#the true positive rate vs the false positive rate
pred <- prediction(prob_test, mat_dataset$res)
pred_perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(pred_perf)



#setting the no of random values for cross validation
set.seed(1500)
cvfit  <- glm(res ~ G2 + goout + reason , family = quasibinomial, data = mat_dataset)
#k fold cross validation using k = 10 folds
cv.err.10 <- cv.glm(data = mat_dataset, glmfit = cvfit, K = 10)
cv.err.10$delta


