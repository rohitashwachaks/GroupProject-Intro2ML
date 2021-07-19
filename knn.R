#----Import Libraries----
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

#----Stratified Sampling----

# %age strokes v/s Non-strokes
sum(as.character(data$stroke) == "1")/n # Just 4.25% positives?

pos_samples = data[(as.character(data$stroke) == "1"),]
neg_samples = data[-(as.character(data$stroke) == "1"),]

# taking pos:negs :: 2:3
undersampled_negs = data[(sample(1:dim(neg_samples)[1], dim(pos_samples)[1]*1.5)),]

df <- rbind(pos_samples, undersampled_negs)

n <- dim(df)[1]

# Normalise Continuous Variables
normalize <- function(x){
  return ((x - mean(x)) / sd(x))
}
df$avg_glucose_level <- normalize(df$avg_glucose_level)
df$age <- normalize(df$age)
df$bmi <- normalize(df$bmi)

#plot(df %>% select(age, bmi, avg_glucose_level, hypertension,
#                   gender, heart_disease, ever_married))

#plot(df$age, df$avg_glucose_level, col = df$stroke, pch=19, cex=1)

#----Neighbourhood Size----

# taking a sample of 70%
tr_size = floor(n*0.7)
tr <- sample(1:n, tr_size)

train <- df[tr,]
test <- df[-tr,]

# Initialising MSE to NULL
# MSE Will later be populated as a column vector containing MSE
# value for each 'k' nearest neighbour
MSE <- NULL
kk <- seq(2,tr_size,by=20)
#i <- 10


# Trying out different neighbourhood size
for(i in kk)
{
  # stroke~ means predict stroke using everything
  near <- kknn(stroke~age,train,test,k=i,kernel = "rectangular")
  # aux is the MSE for given 'i'
  #aux <- mean((test$stroke-near$fitted)^2)
  
  #----MODIFY FOR CLASSIFICATION
  # 1. Confusion Matrix
  #cm <- confusionMatrix(near$fitted, test$stroke, positive = "1")
  
  # Precision and Recall of Preds.
  precision <- posPredValue(near$fitted, test$stroke, positive="1")
  recall <- sensitivity(near$fitted, test$stroke, positive="1")
  
  # Caclulating F1-score
  F1 <- (2 * precision * recall) / (precision + recall)
  #aux <- F1
  
  # 2.Calculating Log.Reg. Cost
  #aux <- cost(near$prob, test$stroke)
  
  
  
  # append aux to MSE
  MSE <- c(MSE,aux)
  
  cat(i,'\n')
  # Plot stroke v/s age
  # Add fitted line (K-nearest neighbour prediction for each point)
  #plot(train$stroke,train$age,main=paste("k=",i),pch=19,cex=0.8,col="darkgray")
  #lines(test[,1],near$fitted,col=2,lwd=2)
  
  #cat ("Press [enter] to continue")
  #line <- readline()
}

best <- which.min(MSE)

# Plot of RMSE v/s neighborhood size
plot(log(1/kk),sqrt(MSE),type="b",
     xlab="Complexity (log(1/k))",
     ylab="RMSE",
     main = "Plot of RMSE v/s Neighbourhood size(Model Complexity)",
     col="blue",lwd=2,cex.lab=1.2)
text(log(1/kk[1]),sqrt(MSE[1])+0.3,paste("k=",kk[1]),col=2,cex=1.2)
text(log(1/kk[10])+0.4,sqrt(MSE[10]),paste("k=",kk[10]),col=2,cex=1.2)
text(log(1/kk[5])+0.4,sqrt(MSE[5]),paste("k=",kk[5]),col=2,cex=1.2)

#----KFold KNN----

kcv = 49

# Divide into groups of size n0
n0 = ceiling(n/kcv)

out_MSE = matrix(0,kcv,100)


used = NULL
set = 1:n

for(j in 1:kcv)
{
  if(n0<length(set)){
    val = sample(set,n0)
  }
  else{
    val=set
  }
  
  train_i = data.frame(lstat,medv)[-val,]
  test_i = data.frame(lstat,medv)[val,]
  
  for(i in 1:100)
  {
    near = kknn(medv~lstat,train_i,test_i,k=i,kernel = "rectangular")
    aux = mean((test_i[,2]-near$fitted)^2)
    
    out_MSE[j,i] = aux
  }
  
  used = union(used,val)
  set = (1:n)[-used]
  
  cat(j,'\n')
}

mMSE = apply(out_MSE,2,mean)

plot(log(1/(1:100)),sqrt(mMSE),xlab="Complexity (log(1/k))",ylab="out-of-sample RMSE",col=4,lwd=2,type="l",cex.lab=1.2,main=paste("kfold(",kcv,")"))
best = which.min(mMSE)
text(log(1/best),sqrt(mMSE[best])+0.1,paste("k=",best),col=2,cex=1.2)
text(log(1/2),sqrt(mMSE[2])+0.3,paste("k=",2),col=2,cex=1.2)
text(log(1/100)+0.4,sqrt(mMSE[100]),paste("k=",100),col=2,cex=1.2)

#----ROUGH----

cost <- function(pred, y){
  shift <- length(pred)/2
  cst <- 0
  for (i in 1:length(y)){
    
    #cat(i,"->", as.character(y[i])," : ",
      #  pred[c(5,(length(pred)/2)+5)],"\n")
    
    if(as.character(y[i]) == "1"){
      cst <- cst -log(pred[shift+i])
    }
    else{
      cst <- cst - log(pred[i])
    }
  }
  return (cst)
}

cost(near$prob, test$stroke)