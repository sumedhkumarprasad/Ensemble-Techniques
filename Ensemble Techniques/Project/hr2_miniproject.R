rm(list=ls())
setwd("D:/Great Lakes PGPDSE/Great Lakes/13 Ensemble Techniques/Mini Project")
hr=read.csv("hr_working.csv",stringsAsFactors = TRUE)

boxplot(Age~Department,data=hr,main= "Age vs Department")
# There is no outlier in the data when comparing Age with Department.
# Median Age of HR and R&D is appoximately same and Sales have lowest median age among all.
boxplot(Age~Education,data=hr,main= "Age vs Education")
#From the above graph it is clear that Below collage have the lowest median age.
#Bachelor and Collage have the approximately same median Age.
#Doctor is having the highest median age.
#Master median age is slightly above the Bachelor and collage median age.
boxplot(Age~Gender,data=hr,main= "Age vs Gender")
#Median Age of Female is more compare to the Male.
boxplot(DistanceFromHome~Gender,data=hr,main= "Distance vs Gender")
# Male and Female both median distance from home is almost same from the working location.
# So distance from home to office does not effect much to Gender.
boxplot(HourlyRate~JobLevel,data=hr,main= "HourlyRate vs JobLevel")
#There is a median of SeniorLevelL2 is higher hourly rate among all job level.
boxplot(HourlyRate~JobRole,data=hr,main= "HourlyRate vs JobRole")
# Hourly rate of mangaer is very high among all other job role.

#Target column is "Attrition" column. Convert Yes / No values to 1 / 0
levels(hr$Attrition) <- c(0,1)
hr$Attrition
hr$Attrition <- as.numeric(as.character(hr$Attrition))
str(hr)

#Split file in 70 : 30

dt = sort(sample(nrow(hr), nrow(hr)*.7))
train<-hr[dt,]
test<-hr[-dt,]

c(nrow(train), nrow(test))

#Building the model using Random Forest

library(randomForest)
?randomForest
View(train)

RF <- randomForest(as.factor(Attrition) ~ ., data = train, 
                   ntree=501, mtry = 3, nodesize = 10,
                   importance=TRUE)

# From the graph of Error Rates after 80 tree Error became constant
RF$err.rate

## List the importance of the variables.
impVar <- round(randomForest::importance(RF), 2)
View(impVar[order(impVar[,4], decreasing=TRUE),])
# Important List of variables 
# Overtime, Monthly Income,Job Role, Age

?tuneRF
## Tuning Random Forest
tRF <- tuneRF(x = train[,-c(1)], 
              y=as.factor(train$Attrition),
              mtryStart = 3, 
              ntreeTry=101, 
              stepFactor = 1.5, 
              improve = 0.0001, 
              trace=TRUE, 
              plot = TRUE,
              doBest = TRUE,
              nodesize = 300, 
              importance=TRUE
)

tRF
tRF$importance
## List the importance of the variables.
impVart <- round(randomForest::importance(tRF), 2)
View(impVart[order(impVart[,4], decreasing=TRUE),])
# After tuning the modelthe list of Important variable is
# OverTime,YearsAtCompany,MonthlyIncome,TotalWorkingYears

View(train)
## Scoring syntax
train$predict.class <- predict(tRF, train, type="class")
train$predict.score <- predict(tRF, train, type="prob")
head(train)
class(train$predict.score)

## deciling
## deciling code
decile <- function(x){
  deciles <- vector(length=10)
  for (i in seq(0.1,1,.1)){
    deciles[i*10] <- quantile(x, i, na.rm=T)
  }
  return (
    ifelse(x<deciles[1], 1,
           ifelse(x<deciles[2], 2,
                  ifelse(x<deciles[3], 3,
                         ifelse(x<deciles[4], 4,
                                ifelse(x<deciles[5], 5,
                                       ifelse(x<deciles[6], 6,
                                              ifelse(x<deciles[7], 7,
                                                     ifelse(x<deciles[8], 8,
                                                            ifelse(x<deciles[9], 9, 10
                                                            ))))))))))
}


train$deciles <- decile(train$predict.score[,2])


library(data.table)
tmp_DT = data.table(train)
rank <- tmp_DT[, list(
  cnt = length(Attrition), 
  cnt_resp = sum(Attrition), 
  cnt_non_resp = sum(Attrition == 0)) , 
  by=deciles][order(-deciles)]
rank$rrate <- round (rank$cnt_resp / rank$cnt,2);
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp),2);
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp),2);
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp);


library(scales)
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

View(rank)
# Ks statistics for train data is 8th decile and highest KS Statistics is 0.52
sum(train$Attrition) / nrow(train)

library(ROCR)
pred <- prediction(train$predict.score[,2], train$Attrition)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
KS   ##0.5337481

## Area Under Curve
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)
auc   ## 0.8288224

## Gini Coefficient
library(ineq)
gini = ineq(train$predict.score[,2], type="Gini")
gini   ##0.7692009

## Classification Error
with(train, table(Attrition, predict.class))
# Mis Classification Error 328/(1730+328) =15.93%
# Accuracy for train data 84.06%

## Scoring syntax
test$predict.class <- predict(tRF, test, type="class")
test$predict.score <- predict(tRF, test, type="prob")
head(test)
class(test$predict.score)

test$deciles <- decile(test$predict.score[,2])

tmp_DT = data.table(test)
h_rank <- tmp_DT[, list(
  cnt = length(Attrition), 
  cnt_resp = sum(Attrition), 
  cnt_non_resp = sum(Attrition == 0)) , 
  by=deciles][order(-deciles)]
h_rank$rrate <- round (h_rank$cnt_resp / h_rank$cnt,2);
h_rank$cum_resp <- cumsum(h_rank$cnt_resp)
h_rank$cum_non_resp <- cumsum(h_rank$cnt_non_resp)
h_rank$cum_rel_resp <- round(h_rank$cum_resp / sum(h_rank$cnt_resp),2);
h_rank$cum_rel_non_resp <- round(h_rank$cum_non_resp / sum(h_rank$cnt_non_resp),2);
h_rank$ks <- abs(h_rank$cum_rel_resp - h_rank$cum_rel_non_resp);


library(scales)
h_rank$rrate <- percent(h_rank$rrate)
h_rank$cum_rel_resp <- percent(h_rank$cum_rel_resp)
h_rank$cum_rel_non_resp <- percent(h_rank$cum_rel_non_resp)

View(h_rank)
# Highest KS value is for 8th Decile having 46% kS Statistics
library(ROCR)
pred <- prediction(test$predict.score[,2], test$Attrition)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
KS   ##0.4331515

## Area Under Curve
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)
auc   ## 0.7908788

## Gini Coefficient
library(ineq)
gini = ineq(test$predict.score[,2], type="Gini")
gini  ##0.7180185

## Classification Error
with(test, table(Attrition, predict.class))
# Mis Classification Error 328/(1730+328) =16.55%
# Accuracy for test data 83.44%