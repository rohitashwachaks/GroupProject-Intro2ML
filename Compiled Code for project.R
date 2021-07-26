library(class) ## a library with lots of classification tools
library(kknn) #a library for KNN
library(dplyr) #library for dplyr tool

rm(list=ls())
# setwd("C:/McCombs MSBA/summer sem/intro to ML/Exam and project/Project MSBA")
stroke <- read.csv("healthcare-dataset-stroke-data.csv") #read in stroke data

dim(stroke) # get dimensions of the dataset
n = dim(stroke)[1] # Set n equal to the number of rows in the dataset 
names(stroke) #get column names


#Check what the count is of each of these columns before changing to strings
stroke %>%  group_by(hypertension) %>% summarise(n())
stroke %>%  group_by(heart_disease) %>% summarise(n())
stroke %>%  group_by(stroke) %>% summarise(n())

#Change the numerical values of predictors and predictions to interpretable strings
stroke$hypertension[stroke$hypertension == 1] <- "Hypertension" 
stroke$hypertension[stroke$hypertension == 0] <- "No hypertension"

#Confirm that the count has not changed, just the variable type
stroke %>%  group_by(hypertension) %>% summarise(n())

#Change the numerical values of predictors and predictions to interpretable strings
stroke$heart_disease[stroke$heart_disease == 1] <- "Heart disease"
stroke$heart_disease[stroke$heart_disease == 0] <- "No heart disease"

#Confirm that the count has not changed, just the variable type
stroke %>%  group_by(heart_disease) %>% summarise(n())

#Change the numerical values of predictors and predictions to interpretable strings
#stroke$stroke[stroke$stroke == 1] <- "Stroke"
#stroke$stroke[stroke$stroke == 0] <- "No stroke"

#Confirm that the count has not changed, just the variable type
stroke %>%  group_by(stroke) %>% summarise(n())

str(stroke) # Check the structure of the dataset 

#Start converting variables to factors if they are categorical variables
stroke$id <- as.factor(stroke$id)
stroke$gender <- as.factor(stroke$gender)
stroke$hypertension <- as.factor(stroke$hypertension)
stroke$heart_disease <- as.factor(stroke$heart_disease)
stroke$ever_married <- as.factor(stroke$ever_married)
stroke$work_type <- as.factor(stroke$work_type)
stroke$Residence_type <- as.factor(stroke$Residence_type)
stroke$smoking_status <- as.factor(stroke$smoking_status)
#stroke$stroke <- as.factor(stroke$stroke)

str(stroke) #Check updated structure to confirm changes made

stroke$bmi <- as.double(stroke$bmi)
sum(is.na(stroke$bmi))/n #around 4 percent of the observations do not have BMI
# Should we remove that data? #Should we set those NAs = a dummy variable so we
# don't lose the rows of data? Just keep it as "Not Available"



######################
library(gbm)
library(data.table)
library(caret)
# Compute sample sizes.
set.seed(0)

sampleSizeTraining   <- floor(0.7 * nrow(stroke))
sampleSizeValidation <- floor(0.15 * nrow(stroke))
sampleSizeTest       <- floor(0.15 * nrow(stroke))

# Create the randomly-sampled indices for the dataframe. Use setdiff() to
# avoid overlapping subsets of indices.
indicesTraining    <- sort(sample(seq_len(nrow(stroke)), size=sampleSizeTraining))
indicesNotTraining <- setdiff(seq_len(nrow(stroke)), indicesTraining)
indicesValidation  <- sort(sample(indicesNotTraining, size=sampleSizeValidation))
indicesTest        <- setdiff(indicesNotTraining, indicesValidation)

#Output the three dataframes for training, validation and test.
dfTraining   <- stroke[indicesTraining, -1]
dfValidation <- stroke[indicesValidation, -1]
dfTest       <- stroke[indicesTest, -1]

#parameters to test for boosting
num_trees <- c(2000,5000)
learning <- c(0.1,0.001)
depth <- c(4,10)

#empty df to store precision and recall outputs
performance_matrix <- data.frame(variables = character(),
                                 precision = integer(),
                                 recall = integer())

#looping through parameter combinations
for (i in length(num_trees)) {
  for (j in length(learning)){
    for (k in length(depth)){
      
      boosted <- gbm(stroke~.,data=dfTraining, distribution="bernoulli",n.trees =num_trees[i], shrinkage = learning[j],interaction.depth = depth[k])
      pred_cv <- predict(boosted,newdata = dfValidation,n.trees = num_trees[i],type = "response")
      pred_cv <- data.frame(pred_cv)
      pred_cv$class <- ifelse(pred_cv$pred_cv >=0.5, 1,0)
      #pred_actual <- data.frame(pred_cv$class, pred_cv$pred_cv,dfValidation$stroke )
      table(pred_cv$class,dfValidation$stroke)
      precision <- posPredValue(as.factor(pred_cv$class),as.factor(dfValidation$stroke), positive="1")
      recall <- sensitivity(as.factor(pred_cv$class),as.factor(dfValidation$stroke), positive="1")
      performance_mat <- data.frame(paste0("num of trees: ",num_trees[i]," ; shrinkage : ",learning[j]," ; depth : ",depth[k] ),precision, recall)
      colnames(performance_mat)[1] <- "variables"
      performance_matrix <- rbind(performance_matrix, performance_mat)
    }
  }
}

#using the parameter values for best recall
i=2
j=2
k=2


#plotting for variable importance
boosted <- gbm(stroke~.,data=dfTraining, distribution="bernoulli",n.trees =num_trees[i], shrinkage = learning[j],interaction.depth = depth[k])
boost_model <- summary(boosted)
barplot(height = boost_model$rel.inf, col = 'blue',names.arg = boost_model$var, cex.names = 0.6,las=2,cex.axis = 0.6)

#running on test set
pred_cv_test <- predict(boosted,newdata = dfTest,n.trees = num_trees[i],type = "response")
pred_cv_test <- data.frame(pred_cv_test)
pred_cv_test$class <- ifelse(pred_cv_test$pred_cv_test >=0.5, 1,0)
table(pred_cv_test$class,dfTest$stroke)
precision_test <- posPredValue(as.factor(pred_cv_test$class),as.factor(dfTest$stroke), positive="1")
recall_test<- sensitivity(as.factor(pred_cv_test$class),as.factor(dfTest$stroke), positive="1")
performance_mat_test <- data.frame(paste0("num of trees: ",num_trees[i]," ; shrinkage : ",learning[j]," ; depth : ",depth[k] ),precision_test, recall_test)
#write.csv(performance_mat_test, "Boosting_test_0,2.csv", row.names = FALSE)


############ Boosting with new threshold #############

pred_cv_test <- predict(boosted,newdata = dfTest[,c('age','avg_glucose_level','bmi','stroke')],n.trees = num_trees[i],type = "response")
pred_cv_test <- data.frame(pred_cv_test)
pred_cv_test$class <- ifelse(pred_cv_test$pred_cv_test >=0.2, 1,0)
table(pred_cv_test$class,dfTest$stroke)
precision_test <- posPredValue(as.factor(pred_cv_test$class),as.factor(dfTest$stroke), positive="1")
recall_test<- sensitivity(as.factor(pred_cv_test$class),as.factor(dfTest$stroke), positive="1")
performance_mat_test <- data.frame(paste0("num of trees: ",num_trees[i]," ; shrinkage : ",learning[j]," ; depth : ",depth[k] ),precision_test, recall_test)
#write.csv(performance_mat_test, "Boosting_test_0,2.csv", row.names = FALSE)




###################################
###################################

#need to solve for class imbalance and leverage the 3 important variables from the model

library(smotefamily)
library(caret)
library(ipred)
library(e1071)
Union_data <- union(dfTraining,dfValidation)
Union_data <- Union_data[,c('age','avg_glucose_level','bmi','stroke')]
Union_data <- Union_data[complete.cases(Union_data),]

#Using SMOTE to oversample and create balanced classes
Balanced_data <- SMOTE(X = Union_data[,-ncol(Union_data)],target = Union_data$stroke, K=4, dup_size = 10)
Oversampled_Data <- Balanced_data$data
table(Oversampled_Data$class)

dt = sort(sample(nrow(Oversampled_Data), nrow(Oversampled_Data)*.8))
train<-Oversampled_Data[dt,]
val<-Oversampled_Data[-dt,]

#iterating through parameters
num_bags <- c(10,100,500,1000)
bagging_matrix <- data.frame(NULL)

for (i in length(num_bags)){
  Bagging_fit <- bagging(as.factor(class) ~ ., 
                         data=train,
                         coob = TRUE,
                         nbagg = num_bags[i])
  print(Bagging_fit)
  pred_bagging <- data.frame(predict(Bagging_fit, val,type = "prob"))
  pred_bagging$class <- ifelse(pred_bagging$X1>0.5,1,0)
  bagging_result <- data.frame(original = val$class, predicted = pred_bagging$class)
  table( bagging_result$predicted,bagging_result$original)
  cm <- confusionMatrix(as.factor(bagging_result$predicted), as.factor(bagging_result$original))
  precision_bagging <- cm$byClass["Precision"]
  recall_bagging <- cm$byClass["Recall"]
  bagging_mat <- data.frame(paste0("num bags: ",num_bags[i]),precision_bagging, recall_bagging)
  colnames(performance_mat)[1] <- "variables"
  bagging_matrix <- rbind(bagging_matrix,bagging_mat)
}


############################
#running on test set with 100 bags(best fit)

i=2
Bagging_fit_test <- bagging(as.factor(class) ~ ., 
                            data=train,
                            coob = TRUE,
                            nbagg = num_bags[i])
pred_bagging_test <- data.frame(predict(Bagging_fit_test, dfTest[,c('age','avg_glucose_level','bmi','stroke')],type = 'prob'))
pred_bagging_test$class <- ifelse(pred_bagging_test$X1>0.5,1,0)
bagging_result_test <- data.frame(original = dfTest$stroke, predicted = pred_bagging_test$class)
colnames(bagging_result_test) <- c("original","predicted")
table( bagging_result_test$predicted,bagging_result_test$original)
cm_test <- confusionMatrix(as.factor(bagging_result_test$predicted), as.factor(bagging_result_test$original))
test_precision_bagging <- cm_test$byClass["Precision"]
testrecall_bagging <- cm_test$byClass["Recall"]


############################
#running bagging with randomforest function
#dividing the dataset into training and test data

#defining the "used" and "set" for sampling
used=NULL
set=1:n

#defining the train sample
train=sample(set, floor(0.70*n))
used=union(used,train)
#defining the test sample
set=(1:n)[-used]
test=sample(set, floor(0.15*n))
used=union(used,test)
#defining the vald sample
set=(1:n)[-used]
val=sample(set, floor(0.15*n))


train_data = stroke[train,]
summary(train_data$stroke)


#applying bagging technique on the data
#ntrees tried = 200,500,5000 
stroke_rf = stroke[-c(1,10)]
rf_stroke = randomForest(stroke ~ . ,data = stroke_rf, subset = train, ntree=200, mtry=10, maxnodes=15, importance=TRUE, na.action = na.omit)
rf_stroke = randomForest(stroke ~ . ,data = stroke_rf, subset = train, ntree=500, mtry=10, maxnodes=15, importance=TRUE, na.action = na.omit)
rf_stroke = randomForest(stroke ~ . ,data = stroke_rf, subset = train, ntree=5000, mtry=10, maxnodes=15, importance=TRUE, na.action = na.omit)


#checking without class weights
rf_stroke = randomForest(stroke ~ ever_married + avg_glucose_level + bmi_num + age ,data = stroke_rf, subset = train, ntree=500, maxnodes=15, mtry=3,importance=TRUE, na.action = na.omit)

#assigning class weights
rf_stroke = randomForest(stroke ~ ever_married + avg_glucose_level + bmi_num + age + hypertension,data = train_data, ntree=500, maxnodes=15, classwt= c("0"=0.526,"1"=9.99), mtry=3,importance=TRUE, na.action = na.omit)

#summary of the RF model
rf_stroke
plot(rf_stroke)
#variable importance
importance(rf_stroke)
varImpPlot(rf_stroke)

test_data = stroke[test,]
vald_data = stroke[val,]
test_str = stroke$stroke[test]
vald_str = stroke$stroke[val]

#checking on the test data
test_stroke <- predict(rf_stroke, newdata = test_data)
confusionMatrix(test_stroke,test_str)

#checking on the validation data
vald_stroke <- predict(rf_stroke, newdata = vald_data)
confusionMatrix(vald_stroke,vald_str)


#conclusion box plots with important variables 
stroke <- data.frame(stroke)
?boxplot()
boxplot(age~stroke,data=stroke,col=c('lightblue','cornflowerblue'),cex.axis=1.5,cex.lab=1.5)
boxplot(avg_glucose_level~stroke,data=stroke,col=c('lightblue','cornflowerblue'))
boxplot(bmi~stroke = 1,data=strokecol=c('lightblue','cornflowerblue'),cex.axis=1.5,cex.lab=1.5)

