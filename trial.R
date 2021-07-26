library(class) ## a library with lots of classification tools
library(kknn) #a library for KNN
library(dplyr) #library for dplyr tool

rm(list = ls())

stroke <- read.csv("healthcare-dataset-stroke-data.csv") #read in stroke data

#attach(stroke)
# Unique IDs
length(unique(stroke$id)) == dim(stroke)[1]

# removing IDs
stroke <- stroke[,-1]
dim(stroke)

#Start converting variables to factors if they are categorical variables
stroke$gender <- as.factor(stroke$gender)
stroke$hypertension <- as.factor(stroke$hypertension)
stroke$heart_disease <- as.factor(stroke$heart_disease)
stroke$ever_married <- as.factor(stroke$ever_married)
stroke$work_type <- as.factor(stroke$work_type)
stroke$Residence_type <- as.factor(stroke$Residence_type)
stroke$smoking_status <- as.factor(stroke$smoking_status)
stroke$stroke <- as.factor(stroke$stroke)
stroke$bmi <- as.double(stroke$bmi)


#pairs(stroke)

cor(stroke)
