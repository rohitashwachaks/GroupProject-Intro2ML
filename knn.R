#----Import Libraries----
rm(list = ls())
library(class) ## a library with lots of classification tools
library(kknn) #a library for KNN
library(dplyr) #library for dplyr tool
library(tidyr)

#install.packages('e1071', dependencies=TRUE)
#install.packages('caret', dependencies=TRUE)
library(caret)

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

#----Normalise Continuous Variables----
normalize <- function(x){
  return ((x - mean(x)) / sd(x))
}
df$avg_glucose_level <- normalize(df$avg_glucose_level)
df$age <- normalize(df$age)
df$bmi <- normalize(df$bmi)

#----Stratified Sampling----

# %age strokes v/s Non-strokes
sum(as.character(data$stroke) == "1")/n # Just 4.25% positives?

pos_samples = data[(as.character(data$stroke) == "1"),]
neg_samples = data[-(as.character(data$stroke) == "1"),]

# taking pos:negs :: 2:3
undersampled_negs = data[(sample(1:dim(neg_samples)[1], dim(pos_samples)[1]*1.5)),]
discarded_negs = data[-(sample(1:dim(neg_samples)[1], dim(pos_samples)[1]*1.5)),]

df <- rbind(pos_samples, undersampled_negs)
n <- dim(df)[1]


plot(df %>% select(age, bmi, avg_glucose_level, hypertension,
                  gender, heart_disease, ever_married))

plot(df$age, df$avg_glucose_level, col = df$stroke, pch=19, cex=1)
## Add legends

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
kk <- seq(2,tr_size)
#i <- 10

# Trying out different neighbourhood size
for(i in kk)
{
  # stroke~ means predict stroke using everything
  near <- kknn(stroke~age+avg_glucose_level+bmi,train,test,k=i,kernel = "rectangular")
  
  # 1. Confusion Matrix
  cm <- confusionMatrix(near$fitted, test$stroke, positive = "1")

  # Precision and Recall of Preds.
  # precision <- as.numeric(cm$byClass["Precision"])
  recall <- as.numeric(cm$byClass["Recall"])
  # F1 <- as.numeric(cm$byClass["F1"])
  
  # append aux to Recall_vals
  Recall_vals <- c(Recall_vals,recall)
  
  cat(i," -> ",recall,'\n')
}

# Plot stroke v/s age
# Add fitted line (K-nearest neighbour prediction for each point)
# near <- kknn(stroke~age,train,test,k=i,kernel = "rectangular")
# plot(train$age, train$stroke,main=paste("k=",i),pch=19,cex=0.8,col="darkgray")
# lines(test[,1],near$fitted,col=2,lwd=2)

best <- which.max(Recall_vals)
best
max(Recall_vals)

# Plot of RMSE v/s neighborhood size
plot(log(1/kk),Recall_vals,type="b",
     xlab="Complexity (log(1/k))",
     ylab="Recall",
     main = "Plot of Recall v/s Neighbourhood size(Model Complexity)",
     col="blue",lwd=2,cex.lab=1.2)
text(log(1/kk[1]),(Recall_vals[1])+0.3,paste("k=",kk[1]),col=2,cex=1.2)
text(log(1/kk[best])+0.4,(Recall_vals[best]),paste("best k=",kk[best]),col=2,cex=1.2)
text(log(1/kk[364])+0.4,(Recall_vals[364]),paste("k=",kk[364]),col=2,cex=1.2)

# Testing against discarded

near <- kknn(stroke~age+avg_glucose_level+bmi,train,discarded_negs,k=kk[best],kernel = "rectangular")
cm <- confusionMatrix(near$fitted, discarded_negs$stroke, positive = "1")
summary(near$fitted.values)


#----KFold KNN----
kcv = 10

# Divide into groups of size n0
n0 = ceiling(n/kcv)

out_Recall = matrix(0,kcv,tr_size)
used = NULL
set = 1:n

kk <- seq(2,tr_size)

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
    
    # Precision and Recall of Preds.
    # precision <- as.numeric(cm$byClass["Precision"])
    recall <- as.numeric(cm$byClass["Recall"])
    # F1 <- as.numeric(cm$byClass["F1"])
    
    cat("(",j,")",i," -> ",recall,'\n')
    
    out_Recall[j,i] = recall
  }
  
  used = union(used,val)
  set = (1:n)[-used]
  
  cat('------',j,'------\n')
}

mRecall = apply(out_Recall,2,mean)

plot(log(1/(1:tr_size)),mRecall,xlab="Complexity (log(1/k))",ylab="out-of-sample Recall",col=4,lwd=2,type="l",cex.lab=1.2,main=paste("kfold(",kcv,")"))

best = which.max(mRecall)
best
max(mRecall)

text(log(1/best),mRecall[best]+0.1,paste("k=",kk[best]),col=2,cex=1.2)
text(log(1/2),mRecall[2],paste("k=",2),col=2,cex=1.2)
text(log(1/tr_size)+0.4,mRecall[tr_size],paste("k=",tr_size),col=2,cex=1.2)


near <- kknn(stroke~age+avg_glucose_level+bmi,train,discarded_negs,k=kk[best],kernel = "rectangular")
summary(near$fitted.values)


#----ROUGH----
# 
# cost <- function(pred, y){
#   shift <- length(pred)/2
#   cst <- 0
#   for (i in 1:length(y)){
#     
#     #cat(i,"->", as.character(y[i])," : ",
#       #  pred[c(5,(length(pred)/2)+5)],"\n")
#     
#     if(as.character(y[i]) == "1"){
#       cst <- cst -log(pred[shift+i])
#     }
#     else{
#       cst <- cst - log(pred[i])
#     }
#   }
#   return (cst)
# }
# 
# cost(near$prob, test$stroke)