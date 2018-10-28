rm(list=ls())
# Setting working directory
setwd("D:/Great Lakes PGPDSE/Great Lakes/13 Ensemble Techniques/Mini Project")
#Reading the Data Set in csv format
hr=read.csv("HR_Employee_Attrition_Data.csv",stringsAsFactors = TRUE,header = TRUE)

View(hr)
c(nrow(hr))
str(hr)
summary(hr)

#Target column is "Attrition" column. Convert Yes / No values to 1 / 0

levels(hr$Attrition) <- c(0,1)
hr$Attrition
hr$Attrition <- as.numeric(as.character(hr$Attrition))
str(hr)

#Split data set into in 70 : 30
dt = sort(sample(nrow(hr), nrow(hr)*.7))
train<-hr[dt,]
test<-hr[-dt,]

c(nrow(train), nrow(test))

#Build AdaBoosting Model


library(gbm)
?gbm
##bernoulli means binomial data type
set.seed(1212)
gbmFit <- gbm(
  formula           = Attrition~.,
  distribution      = "adaboost",
  data              = train,
  n.trees           = 50, # number of trees
  cv.folds          = 10, # do 10-fold cross-validation
  shrinkage         = 1,  # shrinkage or learning rate
  bag.fraction      = 1  # no dataleft aside
)
## Print the gbm Fit summary
print(gbmFit)
# Output from the Model 
# A gradient boosted model with adaboost loss function.
# 50 iterations were performed.
# The best cross-validation iteration was 44.
# There were 33 predictors of which 25 had non-zero influence.

?summary
# Get the Best No. of Iterations using cross-validation
best.iter <- gbm.perf(gbmFit,method="cv")
print(best.iter)

summary(gbmFit, n.trees = best.iter)
# Maximum relative value are in following Order.
# TotalWorkingYears,OverTime,JobRole,Stock Option Level

# predict on the new data (training data set) using "best" number of trees
train$predict.score <- predict( 
  gbmFit, newdata=train,
  n.trees = best.iter,type = "response")

View(train)
class(train$predict.score)

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


## deciling
train$deciles <- decile(train$predict.score)

## Ranking code
##install.packages("data.table")
library(data.table)

tmp_DT = data.table(train)

rank <- tmp_DT[, list(
  cnt = length(Attrition), 
  cnt_resp = sum(Attrition), 
  cnt_non_resp = sum(Attrition == 0)) , 
  by=deciles][order(-deciles)]

rank$rrate <- round(rank$cnt_resp * 100 / rank$cnt,2);
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_perct_resp <- round(rank$cum_resp * 100 / sum(rank$cnt_resp),2);
rank$cum_perct_non_resp <- round(rank$cum_non_resp * 100 / sum(rank$cnt_non_resp),2);
rank$ks <- abs(rank$cum_perct_resp - rank$cum_perct_non_resp);

View(rank)



##install.packages("ROCR")
library(ROCR)
pred <- prediction(train$predict.score, train$Attrition)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)

#install.packages("MLmetrics")
library(MLmetrics)
gini = Gini(train$predict.score, train$Attrition)

auc   ##  0.9168812
KS   ##0.6512075
gini ##0.8337624

##concordance is same as auc only approach is different

#### Testing on test dataset 

test$predict.score <- 
  predict( gbmFit,newdata=test,
           n.trees = best.iter, type = "response")
test$deciles <- decile(test$predict.score)

## Ranking code
library(data.table)
tmp_DT = data.table(test)
h_rank <- tmp_DT[, list(
  cnt = length(Attrition), 
  cnt_resp = sum(Attrition), 
  cnt_non_resp = sum(Attrition == 0)) , 
  by=deciles][order(-deciles)]
h_rank$rrate <- round(h_rank$cnt_resp * 100 / h_rank$cnt,2);
h_rank$cum_resp <- cumsum(h_rank$cnt_resp)
h_rank$cum_non_resp <- cumsum(h_rank$cnt_non_resp)
h_rank$cum_perct_resp <- round(h_rank$cum_resp * 100 / sum(h_rank$cnt_resp),2);
h_rank$cum_perct_non_resp <- round(h_rank$cum_non_resp * 100 / sum(h_rank$cnt_non_resp),2);
h_rank$ks <- abs(h_rank$cum_perct_resp - h_rank$cum_perct_non_resp);

View(h_rank)



pred <- prediction(test$predict.score, test$Attrition)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS_h <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
auc_h <- performance(pred,"auc"); 
auc_h <- as.numeric(auc_h@y.values)
gini_h = Gini(test$predict.score, test$Attrition)

auc_h   #0.8780375
KS_h    #0.5932627
gini_h  #0.7560913


auc    #0.9168812
KS     #0.6841633
gini   #0.8337624

## We begin the GBM Model Tuning in this section ##

## Import the Data again beacuse in the previous steps predict.score column got added

## Create Grid Search Dataframe
?expand.grid

hyper_grid <- expand.grid(
  shrinkage = c(0.3, 1),  ## You may try more shrinkage parameters
  interaction.depth = c(1, 3), ## You may try more interaction depth
  bag.fraction = c(.8, 1), ## You may try with different values
  n.minobsinnode = c(10), ## You may try different Min Obs combinations
  optimal_trees = NA,  # a parameter to capture iteration results
  valid_error = NA     # a parameter to capture iteration results
)

View(hyper_grid)
library(gbm) 
set.seed(1212) ## Set the seed to ensure reproducibility
start_time<-proc.time() ## Start Process Time
for(i in 1:nrow(hyper_grid)) { ## grid search
  gbm.tune <- gbm(
    formula = Attrition~.,
    distribution =  "adaboost",
    data =train,
    n.trees = 50,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    cv.folds  = 10
  )
  best.iter <- gbm.perf(gbm.tune,method="cv")
  hyper_grid$optimal_trees[i] <- best.iter 
  hyper_grid$valid_error[i] <- gbm.tune$valid.error[best.iter]
}
end_time<-proc.time() ## End Process Time
proc_time=end_time - start_time
proc_time ## Total Process Time
'proc_time ## Total Process Time
   user  system elapsed 
6.64    2.44  163.22'
head(hyper_grid)

# check performance using  cross-validation
gbm.perf(gbmFit,method="cv")


# Predict Score and Check Model Performance

library(ModelPerformance)

train$predict.score <- predict( 
  gbmFit, newdata=train, type = "response")

train$decile = ModelPerformance::decile(train$predict.score)
rank <- ModelPerformance::ROTable(train, "Attrition", "predict.score")

View(rank)

test$predict.score <- predict( 
  gbmFit, newdata=test, type = "response")

test$decile = ModelPerformance::decile(test$predict.score)
rank <- ModelPerformance::ROTable(test, "Attrition", "predict.score")
View(rank)

?ROTable

ModelPerformance::concordance(train,"Attrition", "predict.score")
ModelPerformance::KS_AUC(train,"Attrition", "predict.score")

ModelPerformance::concordance(test,"Attrition", "predict.score")
ModelPerformance::KS_AUC(test,"Attrition", "predict.score")
