library(class) ## a library with lots of classification tools
library(kknn) #a library for KNN
library(dplyr) #library for dplyr tool
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
stroke$stroke[stroke$stroke == 1] <- "Stroke"
stroke$stroke[stroke$stroke == 0] <- "No stroke"

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
stroke$stroke <- as.factor(stroke$stroke)

str(stroke) #Check updated structure to confirm changes made

stroke$bmi <- as.double(stroke$bmi)
sum(is.na(stroke$bmi))/n #around 4 percent of the observations do not have BMI
# Should we remove that data? #Should we set those NAs = a dummy variable so we
# don't lose the rows of data? Just keep it as "Not Available"