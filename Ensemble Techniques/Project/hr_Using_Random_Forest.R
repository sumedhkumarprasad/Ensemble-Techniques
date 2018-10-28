setwd ("C:/Users/User/Desktop/Data Science/Ensemble Techniques/Mini Project")
getwd()

#####################################################################################
#Import HR_Employee_Attrition_Data.csv
#####################################################################################

HR_Employee <- read.csv("HR_Employee_Attrition_Data.csv", header = T)

View(HR_Employee)
c(nrow(HR_Employee))
str(HR_Employee)
summary(HR_Employee)

#####################################################################################
#Target column is "Attrition" column. Convert Yes / No values to 1 / 0
#####################################################################################

levels(HR_Employee$Attrition) <- c(0,1)
HR_Employee$Attrition
HR_Employee$Attrition <- as.numeric(as.character(HR_Employee$Attrition))
str(HR_Employee)
#####################################################################################
#Split file in 70 : 30
#####################################################################################
dt = sort(sample(nrow(HR_Employee), nrow(HR_Employee)*.7))
train<-HR_Employee[dt,]
test<-HR_Employee[-dt,]

c(nrow(train), nrow(test))

#####################################################################################
#Building the model using Random Forest
#####################################################################################

##install.packages("randomForest")
library(randomForest)
?randomForest
View(train)
## Calling syntax to build the Random Forest
RF <- randomForest(as.factor(Attrition) ~ ., data = train, 
                   ntree=501, mtry = 3, nodesize = 10,
                   importance=TRUE)


print(RF)

plot(RF, main="")
legend("topright", c("OOB", "0", "1"), text.col=1:6, lty=1:3, col=1:3)
title(main="Error Rates Random Forest For train data")


RF$err.rate

## List the importance of the variables.
impVar <- round(randomForest::importance(RF), 2)
impVar[order(impVar[,3], decreasing=TRUE),]


?tuneRF
## Tuning Random Forest
tRF <- tuneRF(x = train[,-c(2)], 
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

sum(train$Attrition) / nrow(train)


library(ROCR)
pred <- prediction(train$predict.score[,2], train$Attrition)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
KS   ##0.5439414

## Area Under Curve
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)
auc   ## 0.8395834

## Gini Coefficient
library(ineq)
gini = ineq(train$predict.score[,2], type="Gini")
gini   ##0.744276

## Classification Error
with(train, table(Attrition, predict.class))


## Scoring syntax
test$predict.class <- predict(tRF, test, type="class")
test$predict.score <- predict(tRF, test, type="prob")

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

