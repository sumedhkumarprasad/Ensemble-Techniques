library(tidyverse)
library(xgboost)
library(caret)
# load data
df_train = read.table("DEV_SAMPLE.csv", sep = ",", header = T,
                      colClasses = c("Cust_ID"="character"))
df_test = read.table("HOLDOUT_SAMPLE.csv", sep = ",", header = T,
                     colClasses = c("Cust_ID"="character"))

# Loading labels of train data
y_train = df_train['Target']
X_train = df_train[-grep('Target', colnames(df_train))]
y_test = df_test['Target']
X_test = df_test[-grep('Target', colnames(df_test))]


# combine train and test data
df_all = rbind(X_train,X_test)

# one-hot-encoding categorical features
ohe_feats = c('Gender', 'Occupation', 'AGE_BKT')

dummies <- dummyVars(~ Gender +  Occupation + AGE_BKT, data = df_all)
df_all_ohe <- as.data.frame(predict(dummies, newdata = df_all))


df_all_combined <- cbind(df_all[,-c(which(colnames(df_all) %in% ohe_feats))],df_all_ohe)

X_train = df_all_combined[df_all_combined$Cust_ID %in% df_train$Cust_ID,]
X_test = df_all_combined[df_all_combined$Cust_ID %in% df_test$Cust_ID,]

dtrain <- xgb.DMatrix(data = as.matrix(X_train[,-1]), label = as.matrix(y_train))
dtest <- xgb.DMatrix(data = as.matrix(X_test[,-1]), label = as.matrix(y_test))
?xgboost
set.seed(100)
bst <- xgboost(data = dtrain, eta = 0.8,
               max_depth = 2, 
               nround=25, 
               subsample = 0.7,
               objective = "binary:logistic",
               nthread = 3)

# Get the feature real names
names <- dimnames(data.matrix(X_train[,-1]))[[2]]
# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = bst)
# Nice graph
xgb.plot.importance(importance_matrix)

df_train$prob <- predict(bst, as.matrix(X_train[,-1]))
library(ModelPerformance)
rank<-ROTable(df = df_train,target = "Target",probability = "prob")
View(rank)
KS_AUC(df = df_train,target = "Target",probability = "prob")
library(MLmetrics)
Gini(df_train$prob,df_train$Target)
concordance(df = df_train,target = "Target",probability = "prob")

df_test$prob <- predict(bst, as.matrix(X_test[,-1]))
rank_test<-ROTable(df = df_test,target = "Target",probability = "prob")
View(rank_test)
KS_AUC(df = df_test,target = "Target",probability = "prob")
Gini(df_test$prob,df_test$Target)
concordance(df = df_test,target = "Target",probability = "prob")
