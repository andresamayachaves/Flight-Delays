##############################################################
##    SUPERVISED MACHINE LEARNING MODEL FOR PREDICTION      ##
##   OF FLIGHT DELAYS USING THE K-NEAREST NEIGHBOR METHOD   ##
##############################################################
# STAGE 0: DATA GATHERING                                 ####
##############################################################

# Step 0.1. ----
# Install needed packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitrr", repos = "http://cran.us.r-project.org")

# Step 0.2. ----
# Call needed libraries
library(tidyverse)
library(caret)
library(data.table)
library(randomForest)
library(plyr)
library(ggplot2)
library(knitr)

# Step 0.3. ----
# Gather raw DATA
dl2 <- tempfile()
download.file("https://drive.google.com/uc?export=download&id=1khjPRet2DJPkS1mjLwMPCuUWkI4Lzy1Q", dl2)

# Note: this process could take a couple of minutes
flights_raw <- fread(text = gsub("::", "\t", readLines(dl2)))

rm(dl2)



##############################################################
# STAGE 1: DATA CLEANSING AND PREPARATION                 ####
##############################################################

# Step 1.1. ----
# Cleanse the data

  # Step 1.1.1 ----
  # Select, format and mutate some columns 


  # Select needed rows from 'flights_raw'.

  # DayOfWeek =>  1 (Monday) - 7 (Sunday)
  # DepTime => Actual departure time (local, hhmm)
  # ArrDelay => Difference in minutes between scheduled and actual arrival time
  # Airline => Airline company
  # Distance => Distance between airports (in miles)
preFlights <- select(flights_raw, 
                  DayOfWeek, DepTime, ArrDelay, 
                  Airline, Distance)


  # DepTimeSlot: 60 min Slot in the time of day, ranging from 0 to 23. 
  # Truncated every round hour.
preFlights <- preFlights %>% 
              mutate(DepTimeSlot = 
                ifelse(nchar(DepTime) == 4, substring(DepTime, 1,2), 
                       ifelse(nchar(DepTime) == 3, substring(DepTime, 1,1), '0')))

preFlights$DepTimeSlot = as.integer(preFlights$DepTimeSlot)

  # Step 1.1.2 ----
  # Histogram of the departure time distribution over 60min slots over any day.
preFlights %>% 
  qplot(DepTimeSlot, geom ="histogram",
        main = "Departure Time Distibution",
        ylab = "Count",
        bins = 10, 
        data = ., 
        color = I("black"))

  # Step 1.1.3 ----
  # Histogram of the departure distribution over the days of week.
DaysOfTheWeek <- c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')

dayDistrib = preFlights$DayOfWeek
for (i in 1:length(DaysOfTheWeek)){
  dayDistrib <- replace(dayDistrib, c(dayDistrib == i), DaysOfTheWeek[i])
}

dayD <- factor(dayDistrib,levels =  DaysOfTheWeek)

ggplot(data.frame(c(1), dayD), aes(x = dayD))+ geom_bar() + 
  ggtitle("Departure Dist. on Days of the Week") + xlab ("Day of the Week")

# -- CSV file creation for the report.
# write.csv(preFlights, "preFlights.csv", row.names=FALSE)
  
# Step 1.2 ----
# Select, format and mutate some columns 

  # Step 1.2.1 ----
  # Build the 'Weekslot' column from 'DayOfWeek' and 'DepTimeSlot'
      # WeekSlot: 60 min Slot in the time of a week, ranging from 1 to 168. 
      # Truncated every round hour.
preFlights <- preFlights %>% 
  mutate( WeekSlot = ( DayOfWeek - 1)*24 + DepTimeSlot + 1)

  # Step 1.2.2 ----
  # Eliminate 'Day of week', 'DepTime' and 'DepTimeSlot'
preFlights <- preFlights[-c(1,2,6)]

head(preFlights)

  # Step 1.2.3 ----
  # Create 'DelayRange', rounding to the upper 15 minutes ranges
flights <- preFlights %>% 
  select(Airline, Distance, WeekSlot, ArrDelay) %>% 
  mutate(DelayRange = ifelse(ArrDelay<0, round_any(ArrDelay, 15, f = floor),
                             round_any(ArrDelay, 15, f = ceiling)))

head(flights)

  # Step 1.2.4 ----
  # Replace Airline character names for integers folowing the "AirlineCodes" 
table(flights$Airline)
Airline <- unique(flights$Airline)

  # Note: this process could take a couple of minutes
for (i in 1:length(Airline)){
  flights$Airline <- replace(flights$Airline, c(flights$Airline == Airline[i]),i)
}

    # Show the AirlineCodes
AirlineCodes <- data.frame(Code = 1:length(Airline), Airline)
knitr::kable(AirlineCodes)

  # Step 1.2.5 ----
  # Histogram of the flight delay distribution, in minutes.
flights %>% 
  qplot(ArrDelay, geom ="histogram",
        main = "Flight Delay Distibution",
        ylab = "Count",
        bins = 70, 
        data = ., 
        color = I("black"))

# Step 1.3 ----
# Data splitting

  # Step 1.3.1 ----
  # 'flights' splitting into validation set and subFlight set
    # Validation set will be 10% of flights data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
validation_index <- createDataPartition(y = flights$Delay, 
                                  times = 1, p = 0.1, 
                                  list = FALSE)
subFlights <- flights[-validation_index,]
temp <- flights[validation_index,]

      # Make sure 'Delay' in validation set is also in 'subFlights' set
validation <- temp %>% 
  semi_join(subFlights, by = "ArrDelay")

      # Add rows removed from validation set back into 'subFlights' set
removed <- anti_join(temp, validation)
subFlights <- rbind(subFlights, removed)

      # Remove unnecesary variables
rm(flights, flights_raw, preFlights, removed, temp, validation_index)



  # Step 1.3.2 ----
  # 'subFlights' splitting into training and test sets
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`

flights_test_index <- createDataPartition(y = subFlights$Delay, 
                                        times = 1, p = 0.2, 
                                        list = FALSE)
flightsTrain <- subFlights[-flights_test_index,]
temp1 <- subFlights[flights_test_index,]

    # Make sure userId and movieId in test set are also in train set
flightsTest <- temp1 %>% 
  semi_join(flightsTrain, by = "DelayRange")

    # Add rows removed from test set back into train set
removed1 <- anti_join(temp1, flightsTest)
flightsTrain <- rbind(flightsTrain, removed1)

head(flightsTrain)

    # Remove unnecessary variables
rm(flights_test_index, removed1, temp1)


# Step 1.4 ----
# Sample 75% of Train and Test sets.

  # Step 1.4.1 ----
  # Randomly sample 75% of the Train Set entries
TrainSampleSize <- round(nrow(flightsTrain)*3/4, 0)  
set.seed(1)
TRindex <- sample(nrow(flightsTrain), TrainSampleSize)

flTrainSample <- flightsTrain[TRindex,]
flTrainSample$DelayRange <- factor(flTrainSample$DelayRange)

  # Step 1.4.2 ----
  # Sample the Test Set in a tenth of the Train Set sample size.
TestSampleSize <- round(TrainSampleSize/10, 0)  
set.seed(1)
TSindex <- sample(nrow(flightsTest),TestSampleSize)

flTestSample <- flightsTest[TSindex,]
flTestSample$Delay <- factor(flTestSample$Delay)



# Step 1.5 ----
# Define function for the later calculation of Residual Mean Squared Error
RMSE <- function(predicted_delays, true_delays){
  sqrt(mean((true_delays - predicted_delays)^2))
}



##################################################################
## FLIGHT 'DELAYRANGE' PREDICTION: K-Nearest Neighbors          ##
##################################################################
# STAGE 2: TRAIN AND PREDICTION OF K.NEAREST NEIGHBORS MODELS ----
##################################################################

# Step 2.1 ----
# Train over the 'flTrainSample' dataset with 6 values of k.
knn_fit3  <- knn3(DelayRange ~ Airline + Distance + WeekSlot,
                 k = 3, data = flTrainSample)
knn_fit5  <- knn3(DelayRange ~ Airline + Distance + WeekSlot,
                 k = 5, data = flTrainSample)
knn_fit7  <- knn3(DelayRange ~ Airline + Distance + WeekSlot,
                 k = 7, data = flTrainSample)
knn_fit9  <- knn3(DelayRange ~ Airline + Distance + WeekSlot,
                 k = 9, data = flTrainSample)
knn_fit11 <- knn3(DelayRange ~ Airline + Distance + WeekSlot,
                 k = 11, data = flTrainSample)
knn_fit13 <- knn3(DelayRange ~ Airline + Distance + WeekSlot,
                 k = 13, data = flTrainSample)

knn_fit <- c(knn_fit3, knn_fit5,  knn_fit7, 
                      knn_fit9, knn_fit11, knn_fit13)

# Step 2.2 ----
# Predict over 'flTestSample' (the TEST SET SAMPLE), with the just trained 
# knn models and create a data frame with the predictions.
# ---
# Note: this process could take a couple of minutes
DelayR_hat_knn_Sample3  <- predict(knn_fit3,  flTestSample, type = "class")
DelayR_hat_knn_Sample5  <- predict(knn_fit5,  flTestSample, type = "class")
DelayR_hat_knn_Sample7  <- predict(knn_fit7,  flTestSample, type = "class")
DelayR_hat_knn_Sample9  <- predict(knn_fit9,  flTestSample, type = "class")
DelayR_hat_knn_Sample11 <- predict(knn_fit11, flTestSample, type = "class")
DelayR_hat_knn_Sample13 <- predict(knn_fit13, flTestSample, type = "class")
 
DelayR_hat_knn_Sample <- data.frame(factor(DelayR_hat_knn_Sample3), 
                                    factor(DelayR_hat_knn_Sample5),
                                    factor(DelayR_hat_knn_Sample7), 
                                    factor(DelayR_hat_knn_Sample9),
                                    factor(DelayR_hat_knn_Sample11),
                                    factor(DelayR_hat_knn_Sample13))

# Step 2.3 ----
# Calculate RMSE for the 6 sets of predictions an build a data frame with them.
knn_RMSE_Sample3 <- RMSE(as.integer(DelayR_hat_knn_Sample3), 
                         as.integer(flTestSample$DelayRange))
knn_RMSE_Sample5 <- RMSE(as.integer(DelayR_hat_knn_Sample5), 
                        as.integer(flTestSample$DelayRange))
knn_RMSE_Sample7 <- RMSE(as.integer(DelayR_hat_knn_Sample7), 
                         as.integer(flTestSample$DelayRange))
knn_RMSE_Sample9 <- RMSE(as.integer(DelayR_hat_knn_Sample9), 
                         as.integer(flTestSample$DelayRange))
knn_RMSE_Sample11 <- RMSE(as.integer(DelayR_hat_knn_Sample11), 
                         as.integer(flTestSample$DelayRange))
knn_RMSE_Sample13 <- RMSE(as.integer(DelayR_hat_knn_Sample13), 
                         as.integer(flTestSample$DelayRange))


knn_RMSE_Sample <- c(knn_RMSE_Sample3, knn_RMSE_Sample5, knn_RMSE_Sample7,
                     knn_RMSE_Sample9,knn_RMSE_Sample11,knn_RMSE_Sample13)
knn_RMSE_Sample

rm(knn_RMSE_Sample3, knn_RMSE_Sample5, knn_RMSE_Sample7, 
   knn_RMSE_Sample9,knn_RMSE_Sample11, knn_RMSE_Sample13)

ks <- seq(3,13,2)

plot(ks, knn_RMSE_Sample, main = "RMSE change with the k value", xlab=("k"))

  # Determine the value of 'k' that produces the smallest RMSE.
k_best  <- ks[which.min(knn_RMSE_Sample)]
k_best_ind <- which(ks == k_best)

# Step 2.4 ----
# Predict over the TEST SAMPLE SET.

  # Step 2.4.1 ----
  # Calculate prediction accuracies of all built models over the TEST SAMPLE SET.
accuracies <- matrix(0, length(ks))

for (i in 1:length(ks)){
  accuracies[i,1] <- confusionMatrix(factor(DelayR_hat_knn_Sample[,i],
                                            levels = levels(factor(flTestSample$DelayRange))),
                                     factor(flTestSample$DelayRange))$overall["Accuracy"]
}

accuracies

  # Step 2.4.2 ----
  # Determine the index of the max accuracy, and its corresponding k-value.
MaxAccIndex <- which.max(accuracies)
k_maxAcc <- ks[MaxAccIndex]

plot(ks,accuracies, main = "Accuracy change with the k value", xlab=("k"))

  # Step 2.4.3 ----
  # Show the Accuracy and RMSE results as a Knitr table, for the built models so far.
rmse_results <- data.frame(Method = c("3-nn Over Sample Test Set",
                                      "5-nn Over Sample Test Set",
                                      "7-nn Over Sample Test Set",
                                      "9-nn Over Sample Test Set",
                                      "11-nn Over Sample Test Set",
                                      "13-nn Over Sample Test Set"),
                           Accuracy = accuracies,
                           RMSE = knn_RMSE_Sample)

rmse_results %>% knitr::kable()

# Step 2.5 ----
# Show the results of the model that yields the highest accuracy, and that with the lowest RMSE.
Best_ks <- data.frame(Best_Metric = c("Highest Accuracy","Lowest RMSE"), 
                  Method = c(paste(k_maxAcc,"-nn Over Sample Datset", sep = ""), 
                             paste(k_best,"-nn Over Sample Datset", sep = "")),
                  Accuracy = c(accuracies[MaxAccIndex], 
                               accuracies[k_best_ind]),
                  RMSE = c(knn_RMSE_Sample[MaxAccIndex],
                           knn_RMSE_Sample[k_best_ind]))

Best_ks %>% knitr::kable()


# Step 2.6 ----
# Predict over 'flightsTest' (the TEST SET) with the trained knn model
# that minimizes RMSE error.
# ---
# Note: this process could take a couple of minutes
DelayR_hat_knn <- predict(knn_fit3, flightsTest, type = "class" )


  # Step 2.6.1 ----
  # Calculate the corresponding RMSE and Accuracy.
knn_RMSE <- RMSE(as.integer(DelayR_hat_knn), 
                 as.integer(flightsTest$DelayRange))

test_accuracy <- confusionMatrix(DelayR_hat_knn, 
                    factor(flightsTest$DelayRange, 
                           levels = levels(DelayR_hat_knn)))$overall["Accuracy"]

  # Step 2.6.2 ----
  # Report RMSE and Accuracy for the predictions over the TEST SET.
rmse_results <- bind_rows(rmse_results,
                            data.frame(Method = paste(k_best,
                                                      "-nn Over TEST SET", 
                                                      sep = ""),
                                       Accuracy = as.vector(test_accuracy),
                                       RMSE = knn_RMSE))
rmse_results  %>% knitr::kable()



##############################################################
# STAGE 3: PREDICT OVER VALIDATION SET
##############################################################

# Step 3.1 ----
# Predict over 'validaton' set, with the best knn model (that minimizes RMSE).
# ---
# Note: this process could take a couple of minutes
DelayR_hat_val <- predict(knn_fit3, validation, type = "class" )


# Step 3.2 ----
# Calculate prediction RMSE error.
val_RMSE <- RMSE(as.integer(DelayR_hat_val), 
                 as.integer(validation$DelayRange))

# Step 3.3 ----
# Calculate prediction accuracies of all built models over the TEST SAMPLE SET.
val_accuracy <- confusionMatrix(DelayR_hat_val, 
                    factor(validation$DelayRange, 
                           levels = levels(DelayR_hat_val)))$overall["Accuracy"]

# Step 3.4 ----
# Report Accuracy and RMSE of the predictions over the validation set.
rmse_results <- bind_rows(rmse_results,
                          data.frame(Method = paste(k_best,
                                                    "nn Over VALIDATION SET", 
                                                    sep = ""),
                                     Accuracy = as.vector(val_accuracy),
                                     RMSE = val_RMSE))
rmse_results %>% knitr::kable()

# --- CSV files creation for the report.
# write.csv(flightsTest, "flightsTest.csv", row.names=FALSE)
# write.csv(validation, "validation.csv",row.names=FALSE)

# Step 3.5 ----
# For the predictions over the validation set, report Accuracy, RMSE, 
# Data Range Size, and the proportion of RMSE relating to data range. 
final <- data.frame(Method = c(paste(k_best,"nn Over TEST SET", sep = ""),
                               paste(k_best,"nn Over VALIDATION SET", sep = "")),
                    Accuracy = c(as.vector(test_accuracy),as.vector(val_accuracy)),
                    RMSE = c(knn_RMSE, val_RMSE),
                    Delay_min = c(range(flightsTest$DelayRange)[1],
                                  range(validation$DelayRange)[1]),
                    Delay_max = c(range(flightsTest$DelayRange)[2],
                                  range(validation$DelayRange)[2]))

final <- final %>% mutate(Range_Size = Delay_max - Delay_min,
                          Range_Percentage_RMSE = paste(round(100*RMSE/Range_Size,2),'%'))

final %>% knitr::kable()
