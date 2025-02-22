#----Import Libraries----
rm(list = ls())
library(class) ## a library with lots of classification tools
library(kknn) #a library for KNN
library(dplyr) #library for dplyr tool
library(tidyr)

#install.packages('e1071', dependencies=TRUE)
#install.packages('caret', dependencies=TRUE)
library(caret)

# install.packages("scatterplot3d") # Install
library("scatterplot3d")


set.seed(0)

#----Data Prep-----

data <- read.csv("healthcare-dataset-stroke-data.csv")
data <- data[,-1] # Dropping IDs: Useless and sparse
n <- dim(data)[1]

data$bmi <- as.double(data$bmi)
data$avg_glucose_level <- as.double(data$avg_glucose_level)
data$age <- as.double(data$age)

data$gender <- as.factor(data$gender)
data$hypertension <- as.factor(data$hypertension)
data$heart_disease <- as.factor(data$heart_disease)
data$ever_married <- as.factor(data$ever_married)
data$work_type <- as.factor(data$work_type)
data$Residence_type <- as.factor(data$Residence_type)
data$smoking_status <- as.factor(data$smoking_status)
data$stroke <- as.factor(data$stroke)


#----BMI----
sum(is.na(data$bmi))/n #around 4 percent of the observations do not have BMI
data <- data[!is.na(data$bmi),] # Drop BMI = N/A
n <- dim(data)[1]

#----SMOTE(OverSampling)----
# install.packages('smotefamily', dependencies = TRUE)
library(smotefamily)
library(ipred)

Union_data <- data[,c('age','avg_glucose_level','bmi','stroke')]
# Union_data <- Union_data[complete.cases(Union_data),]

#Using SMOTE to oversample and create balanced classes
Balanced_data <- SMOTE(X = Union_data[,-ncol(Union_data)],target = Union_data$stroke, K=4, dup_size = 20)

df <- Balanced_data$data
df$stroke <- as.factor(df$class)
table(df$stroke)
n <- dim(df)[1]

# plot(col = df$stroke, pch=43, cex=1,
#      main = "Sample class distribution",
#      df %>% select(age, bmi, avg_glucose_level))
# 
# # --Age v/s Glucose
# plot(df$age, df$avg_glucose_level,
#      col = df$stroke, pch=43, cex=1,
#      xlab = "Age",
#      ylab = "Average Glucose level",
#      main = "Oversampled Data Distribution\n(Age v/s Glucose)")
# legend(1, 275, legend=c("No Stroke", "Stroke"), col=c("black", "red"), lty=1:2, cex=0.5)
# 
# # --BMI v/s Glucose
# plot(df$bmi, df$avg_glucose_level,
#      col = df$stroke, pch=43, cex=1,
#      xlab = "BMI",
#      ylab = "Average Glucose level",
#      main = "Oversampled Data Distribution\n(BMI v/s Glucose)")
# legend(80, 275, legend=c("No Stroke", "Stroke"), col=c("black", "red"), lty=1:2, cex=0.5)
# 
# 
# # --Plotting All 3 continuous Variables
# color <- c("#E69F00", "#56B4E9")
# shape <- c(42,43)
# scatterplot3d(x = df$age, xlab = "Age",
#               z = df$avg_glucose_level, zlab = "Avg. Glucose Level",
#               y = df$bmi, ylab = "BMI",
#               main = "Oversampled Stroke DataSet",
#               pch = shape[as.numeric(df$stroke)], 
#               color = color[as.numeric(df$stroke)],
#               box = FALSE, angle = 20)

#----Stratified Sampling(UnderSampling)----

# # %age strokes v/s Non-strokes
# sum(as.character(data$stroke) == "1")/n # Just 4.25% positives?
# 
# pos_samples = data[(as.character(data$stroke) == "1"),]
# neg_samples = data[-(as.character(data$stroke) == "1"),]
# 
# # taking pos:negs :: 2:3
# undersampled_negs = data[(sample(1:dim(neg_samples)[1], dim(pos_samples)[1]*1.5)),]
# discarded_negs = data[-(sample(1:dim(neg_samples)[1], dim(pos_samples)[1]*1.5)),]
# 
# df <- rbind(pos_samples, undersampled_negs)
# n <- dim(df)[1]
# 
# plot(col = df$stroke, pch=43, cex=1,
#      main = "Sample class distribution",
#      df %>% select(age, bmi, avg_glucose_level))
# 
# # --Age v/s Glucose
# plot(df$age, df$avg_glucose_level,
#      col = df$stroke, pch=43, cex=1,
#      xlab = "Age",
#      ylab = "Average Glucose level",
#      main = "Undersampled Data Distribution\n(Age v/s Glucose)")
# legend(1, 275, legend=c("No Stroke", "Stroke"), col=c("black", "red"), lty=1:2, cex=0.8)
# 
# # --BMI v/s Glucose
# plot(df$bmi, df$avg_glucose_level,
#      col = df$stroke, pch=43, cex=1,
#      xlab = "BMI",
#      ylab = "Average Glucose level",
#      main = "Undersampled Data Distribution\n(BMI v/s Glucose)")
# legend(10, 275, legend=c("No Stroke", "Stroke"), col=c("black", "red"), lty=1:2, cex=0.5)
# 
# 
# # --Plotting All 3 continuous Variables
# color <- c("#E69F00", "#56B4E9")
# shape <- c(42,43)
# scatterplot3d(x = df$age, xlab = "Age",
#               z = df$avg_glucose_level, zlab = "Avg. Glucose Level",
#               y = df$bmi, ylab = "BMI",
#               main = "Undersampled Stroke DataSet",
#               pch = shape[as.numeric(df$stroke)], 
#               color = color[as.numeric(df$stroke)],
#               box = FALSE, angle = 20)
# 
# 
# 
#----Normalise Continuous Variables----

normalize <- function(x){
  return ((x - mean(x)) / sd(x))
}
df$avg_glucose_level <- normalize(df$avg_glucose_level)
df$age <- normalize(df$age)
df$bmi <- normalize(df$bmi)

#----Neighbourhood Size----

# taking a sample of 70%
tr_size = floor(n*0.7)
tr <- sample(1:n, tr_size)

train <- df[tr,]
test <- df[-tr,]

# Initialising Recall to NULL
# Recall_vals Will later be populated as a column vector containing recall
# value for each 'k' nearest neighbour
Recall_vals <- NULL
kk <- seq(2,tr_size, by=100) # If Using SMOTE
# kk <- seq(2,tr_size)        # If using UnderSampling

# Trying out different neighbourhood size
for(i in kk)
{
  near <- kknn(stroke~age+avg_glucose_level+bmi,train,test,k=i,kernel = "rectangular")
  
  # 1. Confusion Matrix
  cm <- confusionMatrix(near$fitted, test$stroke, positive = "1")
  recall <- as.numeric(cm$byClass["Recall"])

  Recall_vals <- c(Recall_vals,recall)
  cat(i," -> ",recall,'\n')
}

best <- which.max(Recall_vals)
best
max(Recall_vals)

# Plot of RMSE v/s neighborhood size
plot(log(1/kk),Recall_vals,type="l",
     xlab="Complexity (log(1/k))",
     ylab="Recall",
     main = "Plot of Recall v/s \nNeighbourhood size(Model Complexity)",
     col="blue",lwd=2,cex.lab=1.2)
text(log(1/kk[1]),(Recall_vals[1]),paste("k=",kk[1]),col=2,cex=1.2)
text(log(1/kk[best]),(Recall_vals[best]-0.1),paste("best k=",kk[best]),col=2,cex=1.2)
text(log(1/kk[364])+0.4,(Recall_vals[364]+0.05),paste("k=",kk[364]),col=2,cex=1.2)

# Testing against discarded (UnderSampling)

near <- kknn(stroke~age+avg_glucose_level+bmi,train,discarded_negs,k=kk[best],kernel = "rectangular")
summary(near$fitted.values)


#----KFold KNN (Age)----
kcv = 10

# Divide into groups of size n0
n0 = ceiling(n/kcv)

out_Recall = matrix(0,kcv,tr_size)
used = NULL
set = 1:n

kk <- seq(2,tr_size, by=100) # If Using SMOTE
# kk <- seq(2,tr_size)        # If using UnderSampling


for(j in seq(1,kcv))
{
  if(n0<length(set)){
    val = sample(set,n0)
  } else {
    val=set
  }
  
  train_i = df[-val,]
  test_i = df[val,]
  
  for(i in kk)
  {
    near = kknn(stroke~age,train_i,test_i,k=i,kernel = "rectangular")
    
    # 1. Confusion Matrix
    cm <- confusionMatrix(near$fitted, test_i$stroke, positive = "1")
    recall <- as.numeric(cm$byClass["Recall"])
    
    cat("(",j,")",i," -> ",recall,'\n')
    out_Recall[j,i] = recall
  }
  
  used = union(used,val)
  set = (1:n)[-used]
  
  cat('------',j,'------\n')
}

mRecall = apply(out_Recall,2,mean)

best = which.max(mRecall)
best
max(mRecall)

plot(log(1/(1:tr_size)),mRecall,
     xlab="Complexity (log(1/k))",
     ylab="out-of-sample Recall",
     col=4,lwd=2,type="l",
     cex.lab=1.2,
     main=paste("Knn- Age\n Best Recall: ",round(max(mRecall), digits=2)))

text(log(1/best),mRecall[best]-0.1,paste("best k=",kk[best]),col=2,cex=1.2)
text(log(1/2),mRecall[2],paste("k=",2),col=2,cex=1.2)
text(log(1/tr_size)+0.4,mRecall[tr_size],paste("k=",tr_size),col=2,cex=1.2)


near <- kknn(stroke~age,train,discarded_negs,k=kk[best],kernel = "rectangular")
summary(near$fitted.values)


#----KFold KNN (Age + Avg. Glucose)----
kcv = 10

# Divide into groups of size n0
n0 = ceiling(n/kcv)

out_Recall = matrix(0,kcv,tr_size)
used = NULL
set = 1:n

kk <- seq(2,tr_size, by=100) # If Using SMOTE
# kk <- seq(2,tr_size)        # If using UnderSampling

for(j in seq(1,kcv))
{
  if(n0<length(set)){
    val = sample(set,n0)
  } else {
    val=set
  }
  
  train_i = df[-val,]
  test_i = df[val,]
  
  for(i in kk)
  {
    near = kknn(stroke~age+avg_glucose_level,train_i,test_i,k=i,kernel = "rectangular")
    
    # 1. Confusion Matrix
    cm <- confusionMatrix(near$fitted, test_i$stroke, positive = "1")
    recall <- as.numeric(cm$byClass["Recall"])
    
    # cat("(",j,")",i," -> ",recall,'\n')
    out_Recall[j,i] = recall
  }
  
  used = union(used,val)
  set = (1:n)[-used]
  
  cat('------',j,'------\n')
}

mRecall = apply(out_Recall,2,mean)

best = which.max(mRecall)
best
max(mRecall)

plot(log(1/(1:tr_size)),mRecall,
     xlab="Complexity (log(1/k))",
     ylab="out-of-sample Recall",
     col=4,lwd=2,type="l",
     cex.lab=1.2,
     main=paste("Knn- Age+Avg.Glucose\n Best Recall: ",round(max(mRecall), digits=2)))

text(log(1/best),mRecall[best]-0.1,paste("best k=",kk[best]),col=2,cex=1.2)
text(log(1/2),mRecall[2],paste("k=",2),col=2,cex=1.2)
text(log(1/tr_size)+0.4,mRecall[tr_size],paste("k=",tr_size),col=2,cex=1.2)


near <- kknn(stroke~age+avg_glucose_level,train,discarded_negs,k=kk[best],kernel = "rectangular")
summary(near$fitted.values)


#----KFold KNN (Age + Avg. Glucose + BMI)----
kcv = 10

# Divide into groups of size n0
n0 = ceiling(n/kcv)

out_Recall = matrix(0,kcv,tr_size)
used = NULL
set = 1:n

kk <- seq(2,tr_size, by=100) # If Using SMOTE
# kk <- seq(2,tr_size)        # If using UnderSampling


for(j in seq(1,kcv))
{
  if(n0<length(set)){
    val = sample(set,n0)
  } else {
    val=set
  }
  
  train_i = df[-val,]
  test_i = df[val,]
  
  for(i in kk)
  {
    near = kknn(stroke~age+avg_glucose_level+bmi,train_i,test_i,k=i,kernel = "rectangular")
    
    # 1. Confusion Matrix
    cm <- confusionMatrix(near$fitted, test_i$stroke, positive = "1")
    recall <- as.numeric(cm$byClass["Recall"])
    
    # cat("(",j,")",i," -> ",recall,'\n')
    out_Recall[j,i] = recall
  }
  
  used = union(used,val)
  set = (1:n)[-used]
  
  cat('------',j,'------\n')
}

mRecall = apply(out_Recall,2,mean)

best = which.max(mRecall)
best
max(mRecall)

plot(log(1/(1:tr_size)),mRecall,
     xlab="Complexity (log(1/k))",
     ylab="out-of-sample Recall",
     col=4,lwd=2,type="l",
     cex.lab=1.2,
     main=paste("Knn- Age+Avg.Glucose+BMI\n Best Recall: ",round(max(mRecall), digits=2)))

text(log(1/best),mRecall[best]-0.1,paste("best k=",kk[best]),col=2,cex=1.2)
text(log(1/2),mRecall[2],paste("k=",2),col=2,cex=1.2)
text(log(1/tr_size)+0.4,mRecall[tr_size],paste("k=",tr_size),col=2,cex=1.2)


near <- kknn(stroke~age+avg_glucose_level+bmi,train,discarded_negs,k=kk[best],kernel = "rectangular")
summary(near$fitted.values)


#----KFold KNN (All Params)----
kcv = 10

# Divide into groups of size n0
n0 = ceiling(n/kcv)

out_Recall = matrix(0,kcv,tr_size)
used = NULL
set = 1:n

kk <- seq(2,tr_size, by=100) # If Using SMOTE
# kk <- seq(2,tr_size)        # If using UnderSampling


for(j in seq(1,kcv))
{
  if(n0<length(set)){
    val = sample(set,n0)
  } else {
    val=set
  }
  
  train_i = df[-val,]
  test_i = df[val,]
  
  for(i in kk)
  {
    near = kknn(stroke~.,train_i,test_i,k=i,kernel = "rectangular")
    
    # 1. Confusion Matrix
    cm <- confusionMatrix(near$fitted, test_i$stroke, positive = "1")
    recall <- as.numeric(cm$byClass["Recall"])
    
    # cat("(",j,")",i," -> ",recall,'\n')
    out_Recall[j,i] = recall
  }
  
  used = union(used,val)
  set = (1:n)[-used]
  
  cat('------',j,'------\n')
}

mRecall = apply(out_Recall,2,mean)

best = which.max(mRecall)
best
max(mRecall)

plot(log(1/(1:tr_size)),mRecall,
     xlab="Complexity (log(1/k))",
     ylab="out-of-sample Recall",
     col=4,lwd=2,type="l",
     cex.lab=1.2,
     main=paste("Knn- All Params\n Best Recall: ",round(max(mRecall), digits=2)))

text(log(1/best),mRecall[best]-0.1,paste("best k=",kk[best]),col=2,cex=1.2)
text(log(1/2),mRecall[2],paste("k=",2),col=2,cex=1.2)
text(log(1/tr_size)+0.4,mRecall[tr_size],paste("k=",tr_size),col=2,cex=1.2)


near <- kknn(stroke~.,train,discarded_negs,k=kk[best],kernel = "rectangular")
summary(near$fitted.values)
