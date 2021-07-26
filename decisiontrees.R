library(class) ## a library with lots of classification tools
library(kknn) #a library for KNN
library(dplyr) #library for dplyr tool

rm(list=ls())
setwd("D:/github_repos/GroupProject-Intro2ML")
stroke <- read.csv("healthcare-dataset-stroke-data.csv") #read in stroke data

dim(stroke) # get dimensions of the dataset
n = dim(stroke)[1] # Set n equal to the number of rows in the dataset

# Removing IDs
stroke <- stroke[,-1]
names(stroke) #get column names

#Converting categorical variables----
stroke$gender <- as.factor(stroke$gender)
stroke$hypertension <- as.factor(stroke$hypertension)
stroke$heart_disease <- as.factor(stroke$heart_disease)
stroke$ever_married <- as.factor(stroke$ever_married)
stroke$work_type <- as.factor(stroke$work_type)
stroke$Residence_type <- as.factor(stroke$Residence_type)
stroke$smoking_status <- as.factor(stroke$smoking_status)
stroke$stroke <- as.factor(stroke$stroke)
stroke$bmi <- as.double(stroke$bmi)

#Check count of each of these columns before changing to strings
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

num_trees <- c(2000,5000)
learning <- c(0.1,0.001)
depth <- c(4,10)

performance_matrix <- data.frame(variables = character(),
                                 precision = integer(),
                                 recall = integer())

for (i in length(num_trees)) {
  for (j in length(learning)){
    for (k in length(depth)){
      
      boosted <- gbm(stroke~.,data=dfTraining, distribution="bernoulli",n.trees =num_trees[i], shrinkage = learning[j],interaction.depth = depth[k])
      pred_cv <- predict(boosted,newdata = dfValidation,n.trees = num_trees[i],type = "response")
      pred_cv <- data.frame(pred_cv)
      pred_cv$class <- ifelse(pred_cv$pred_cv >=0.5, 1,0)
      table(pred_cv$class,dfValidation$stroke)
      precision <- posPredValue(as.factor(pred_cv$class),as.factor(dfValidation$stroke), positive="1")
      recall <- sensitivity(as.factor(pred_cv$class),as.factor(dfValidation$stroke), positive="1")
      performance_mat <- data.frame(paste0("num of trees: ",num_trees[i]," ; shrinkage : ",learning[j]," ; depth : ",depth[k] ),precision, recall)
      colnames(performance_mat)[1] <- "variables"
      performance_matrix <- rbind(performance_matrix, performance_mat)
    }
  }
}
boost_model <- summary(boosted)
# par(mar = c(5.5,2,1,1))
barplot(height = boost_model$rel.inf, col = 'blue',names.arg = boost_model$var, cex.names = 0.6,las=2,cex.axis = 0.6)


#}