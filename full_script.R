library(xgboost)
library(Matrix)

set.seed(1234)

# Set your workspace here
setwd("C:/Users/Antonio/Dropbox/8 Semestre/Trabajo de Fin de Grado/Santander/input/")

train <- read.csv("train.csv")
test  <- read.csv("test.csv")
test_var15 <- test$var15


################################
###    Data preprocessing    ###
################################

# Removing IDs
train$ID <- NULL
test.id <- test$ID
test$ID <- NULL

# Extracting TARGET
train.y <- train$TARGET
train$TARGET <- NULL

# 0 count per line
count0 <- function(x) {
  return( sum(x == 0) )
}
train$n0 <- apply(train, 1, FUN=count0)
test$n0 <- apply(test, 1, FUN=count0)

# Removing constant features from train
cat("\n## Removing the constants features from train.\n")
for (f in names(train)) {
  if (length(unique(train[[f]])) == 1) {
    cat(f, "is constant in train. We delete it.\n")
    train[[f]] <- NULL
    test[[f]] <- NULL
  }
}

# Removing identical features
features_pair <- combn(names(train), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(train[[f1]] == train[[f2]])) {
      cat(f1, "and", f2, "are equals.\n")
      toRemove <- c(toRemove, f2)
    }
  }
}

feature.names <- setdiff(names(train), toRemove)

train <- train[, feature.names]
test <- test[, feature.names]
train$TARGET <- train.y


# Detect same rows with different TARGET

index <- which(duplicated(train) | duplicated(train, fromLast = TRUE) == TRUE)
train <- train[,1:307]
index2 <- which(duplicated(train) | duplicated(train, fromLast = TRUE) == TRUE)
index3 <- setdiff(index2, index)
noise <- train[index3,]
train <- train[-index3,]
train.y <- train.y[-index3]

# Load train and split it in 20% 20% 20% 20% 20% without that noise
train$TARGET <- train.y
t1 <- train[1:(dim(train)[1]/5),]
t2 <- train[(dim(train)[1]/5):(2*dim(train)[1]/5),]
t3 <- train[(2*dim(train)[1]/5):(3*dim(train)[1]/5),]
t4 <- train[(3*dim(train)[1]/5):(4*dim(train)[1]/5),]
t5 <- train[(4*dim(train)[1]/5):(5*dim(train)[1]/5),]

# Ensemble the five parts to predict the real class of the noise

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.02,
                max_depth           = 6,
                subsample           = 0.9,
                colsample_bytree    = 0.85
)


t1.y <- t1$TARGET
t1 <- sparse.model.matrix(TARGET ~ ., data = t1)
dtrain <- xgb.DMatrix(data=t1, label=t1.y)
watchlist <- list(train=dtrain)


bst1 <- xgboost(   params              = param, 
                   data                = dtrain, 
                   nrounds             = 500,
                   watchlist           = watchlist,
                   verbose             = 1,
                   maximize            = FALSE
)

t2.y <- t2$TARGET
t2 <- sparse.model.matrix(TARGET ~ ., data = t2)
dtrain <- xgb.DMatrix(data=t2, label=t2.y)
watchlist <- list(train=dtrain)


bst2 <- xgboost(   params              = param, 
                   data                = dtrain, 
                   nrounds             = 500,
                   watchlist           = watchlist,
                   verbose             = 1,
                   maximize            = FALSE
)

t3.y <- t3$TARGET
t3 <- sparse.model.matrix(TARGET ~ ., data = t3)
dtrain <- xgb.DMatrix(data=t3, label=t3.y)
watchlist <- list(train=dtrain)


bst3 <- xgboost(   params              = param, 
                   data                = dtrain, 
                   nrounds             = 500,
                   watchlist           = watchlist,
                   verbose             = 1,
                   maximize            = FALSE
)

t4.y <- t4$TARGET
t4 <- sparse.model.matrix(TARGET ~ ., data = t4)
dtrain <- xgb.DMatrix(data=t4, label=t4.y)
watchlist <- list(train=dtrain)


bst4 <- xgboost(   params              = param, 
                   data                = dtrain, 
                   nrounds             = 500,
                   watchlist           = watchlist,
                   verbose             = 1,
                   maximize            = FALSE
)

t5.y <- t5$TARGET
t5 <- sparse.model.matrix(TARGET ~ ., data = t5)
dtrain <- xgb.DMatrix(data=t5, label=t5.y)
watchlist <- list(train=dtrain)

bst5 <- xgboost(   params              = param, 
                   data                = dtrain, 
                   nrounds             = 500,
                   watchlist           = watchlist,
                   verbose             = 1,
                   maximize            = FALSE
)

# Predict class label and normalize the probability in [0,1]

noise$TARGET <- 0
noise2 <- sparse.model.matrix(TARGET ~ ., data = noise)
preds1 <- predict(bst1, noise2)
preds2 <- predict(bst2, noise2)
preds3 <- predict(bst3, noise2)
preds4 <- predict(bst4, noise2)
preds5 <- predict(bst5, noise2)

preds1 <- preds1/max(preds1)
preds2 <- preds2/max(preds2)
preds3 <- preds3/max(preds3)
preds4 <- preds4/max(preds4)
preds5 <- preds5/max(preds5)

preds1[preds1 >= 0.5] <- 1
preds1[preds1 < 0.5] <- 0
preds2[preds2 >= 0.5] <- 1
preds2[preds2 < 0.5] <- 0
preds3[preds3 >= 0.5] <- 1
preds3[preds3 < 0.5] <- 0
preds4[preds4 >= 0.5] <- 1
preds4[preds4 < 0.5] <- 0
preds5[preds5 >= 0.5] <- 1
preds5[preds5 < 0.5] <- 0

# Majority vote
noise$TARGET[preds1 + preds2 + preds3 + preds4 + preds5 >= 3] <- 1

noise$TARGET[preds1 > 0.5] <- 1

train <- rbind(train, noise)

# Undersampling spliting the data of 0 class in 24 folds

train0 <- train[train$TARGET == 0,]
train1 <- train[train$TARGET == 1,]

t1 <- train0[1:(dim(train0)[1]/24),]
t1 <- rbind(t1, train1)
t2 <- train0[(dim(train0)[1]/24):(2*dim(train0)[1]/24),]
t2 <- rbind(t2, train1)
t3 <- train0[(2*dim(train0)[1]/24):(3*dim(train0)[1]/24),]
t3 <- rbind(t3, train1)
t4 <- train0[(3*dim(train0)[1]/24):(4*dim(train0)[1]/24),]
t4 <- rbind(t4, train1)
t5 <- train0[(4*dim(train0)[1]/24):(5*dim(train0)[1]/24),]
t5 <- rbind(t5, train1)
t6 <- train0[(5*dim(train0)[1]/24):(6*dim(train0)[1]/24),]
t6 <- rbind(t6, train1)
t7 <- train0[(6*dim(train0)[1]/24):(7*dim(train0)[1]/24),]
t7 <- rbind(t7, train1)
t8 <- train0[(7*dim(train0)[1]/24):(8*dim(train0)[1]/24),]
t8 <- rbind(t8, train1)
t9 <- train0[(8*dim(train0)[1]/24):(9*dim(train0)[1]/24),]
t9 <- rbind(t9, train1)
t10 <- train0[(9*dim(train0)[1]/24):(10*dim(train0)[1]/24),]
t10 <- rbind(t10, train1)
t11 <- train0[(10*dim(train0)[1]/24):(11*dim(train0)[1]/24),]
t11 <- rbind(t11, train1)
t12 <- train0[(11*dim(train0)[1]/24):(12*dim(train0)[1]/24),]
t12 <- rbind(t12, train1)
t13 <- train0[(12*dim(train0)[1]/24):(13*dim(train0)[1]/24),]
t13 <- rbind(t13, train1)
t14 <- train0[(13*dim(train0)[1]/24):(14*dim(train0)[1]/24),]
t14 <- rbind(t14, train1)
t15 <- train0[(14*dim(train0)[1]/24):(15*dim(train0)[1]/24),]
t15 <- rbind(t15, train1)
t16 <- train0[(15*dim(train0)[1]/24):(16*dim(train0)[1]/24),]
t16 <- rbind(t16, train1)
t17 <- train0[(16*dim(train0)[1]/24):(17*dim(train0)[1]/24),]
t17 <- rbind(t17, train1)
t18 <- train0[(17*dim(train0)[1]/24):(18*dim(train0)[1]/24),]
t18 <- rbind(t18, train1)
t19 <- train0[(18*dim(train0)[1]/24):(19*dim(train0)[1]/24),]
t19 <- rbind(t19, train1)
t20 <- train0[(19*dim(train0)[1]/24):(20*dim(train0)[1]/24),]
t20 <- rbind(t20, train1)
t21 <- train0[(20*dim(train0)[1]/24):(21*dim(train0)[1]/24),]
t21 <- rbind(t21, train1)
t22 <- train0[(21*dim(train0)[1]/24):(22*dim(train0)[1]/24),]
t22 <- rbind(t22, train1)
t23 <- train0[(22*dim(train0)[1]/24):(23*dim(train0)[1]/24),]
t23 <- rbind(t23, train1)
t24 <- train0[(23*dim(train0)[1]/24):(dim(train0)[1]),]
t24 <- rbind(t24, train1)


################################
### Classification algorithm ###
################################

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.01,
                max_depth           = 5,
                subsample           = 0.6,
                colsample_bytree    = 0.6
)


t1.y <- t1$TARGET
t1 <- sparse.model.matrix(TARGET ~ ., data = t1)
dtrain <- xgb.DMatrix(data=t1, label=t1.y)
watchlist <- list(train=dtrain)


bst1 <- xgboost(   params              = param, 
                   data                = dtrain, 
                   nrounds             = 750,
                   watchlist           = watchlist,
                   verbose             = 1,
                   maximize            = FALSE
)

t2.y <- t2$TARGET
t2 <- sparse.model.matrix(TARGET ~ ., data = t2)
dtrain <- xgb.DMatrix(data=t2, label=t2.y)
watchlist <- list(train=dtrain)


bst2 <- xgboost(   params              = param, 
                   data                = dtrain, 
                   nrounds             = 750,
                   watchlist           = watchlist,
                   verbose             = 1,
                   maximize            = FALSE
)

t3.y <- t3$TARGET
t3 <- sparse.model.matrix(TARGET ~ ., data = t3)
dtrain <- xgb.DMatrix(data=t3, label=t3.y)
watchlist <- list(train=dtrain)


bst3 <- xgboost(   params              = param, 
                   data                = dtrain, 
                   nrounds             = 750,
                   watchlist           = watchlist,
                   verbose             = 1,
                   maximize            = FALSE
)

t4.y <- t4$TARGET
t4 <- sparse.model.matrix(TARGET ~ ., data = t4)
dtrain <- xgb.DMatrix(data=t4, label=t4.y)
watchlist <- list(train=dtrain)


bst4 <- xgboost(   params              = param, 
                   data                = dtrain, 
                   nrounds             = 750,
                   watchlist           = watchlist,
                   verbose             = 1,
                   maximize            = FALSE
)


t5.y <- t5$TARGET
t5 <- sparse.model.matrix(TARGET ~ ., data = t5)
dtrain <- xgb.DMatrix(data=t5, label=t5.y)
watchlist <- list(train=dtrain)


bst5 <- xgboost(   params              = param, 
                   data                = dtrain, 
                   nrounds             = 750,
                   watchlist           = watchlist,
                   verbose             = 1,
                   maximize            = FALSE
)


t6.y <- t6$TARGET
t6 <- sparse.model.matrix(TARGET ~ ., data = t6)
dtrain <- xgb.DMatrix(data=t6, label=t6.y)
watchlist <- list(train=dtrain)


bst6 <- xgboost(   params              = param, 
                   data                = dtrain, 
                   nrounds             = 750,
                   watchlist           = watchlist,
                   verbose             = 1,
                   maximize            = FALSE
)

t7.y <- t7$TARGET
t7 <- sparse.model.matrix(TARGET ~ ., data = t7)
dtrain <- xgb.DMatrix(data=t7, label=t7.y)
watchlist <- list(train=dtrain)


bst7 <- xgboost(   params              = param, 
                   data                = dtrain, 
                   nrounds             = 750,
                   watchlist           = watchlist,
                   verbose             = 1,
                   maximize            = FALSE
)

t8.y <- t8$TARGET
t8 <- sparse.model.matrix(TARGET ~ ., data = t8)
dtrain <- xgb.DMatrix(data=t8, label=t8.y)
watchlist <- list(train=dtrain)


bst8 <- xgboost(   params              = param, 
                   data                = dtrain, 
                   nrounds             = 750,
                   watchlist           = watchlist,
                   verbose             = 1,
                   maximize            = FALSE
)


t9.y <- t9$TARGET
t9 <- sparse.model.matrix(TARGET ~ ., data = t9)
dtrain <- xgb.DMatrix(data=t9, label=t9.y)
watchlist <- list(train=dtrain)


bst9 <- xgboost(   params              = param, 
                   data                = dtrain, 
                   nrounds             = 750,
                   watchlist           = watchlist,
                   verbose             = 1,
                   maximize            = FALSE
)


t10.y <- t10$TARGET
t10 <- sparse.model.matrix(TARGET ~ ., data = t10)
dtrain <- xgb.DMatrix(data=t10, label=t10.y)
watchlist <- list(train=dtrain)


bst10 <- xgboost(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 750,
                    watchlist           = watchlist,
                    verbose             = 1,
                    maximize            = FALSE
)


t11.y <- t11$TARGET
t11 <- sparse.model.matrix(TARGET ~ ., data = t11)
dtrain <- xgb.DMatrix(data=t11, label=t11.y)
watchlist <- list(train=dtrain)


bst11 <- xgboost(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 750,
                    watchlist           = watchlist,
                    verbose             = 1,
                    maximize            = FALSE
)

t12.y <- t12$TARGET
t12 <- sparse.model.matrix(TARGET ~ ., data = t12)
dtrain <- xgb.DMatrix(data=t12, label=t12.y)
watchlist <- list(train=dtrain)


bst12 <- xgboost(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 750,
                    watchlist           = watchlist,
                    verbose             = 1,
                    maximize            = FALSE
)

t13.y <- t13$TARGET
t13 <- sparse.model.matrix(TARGET ~ ., data = t13)
dtrain <- xgb.DMatrix(data=t13, label=t13.y)
watchlist <- list(train=dtrain)


bst13 <- xgboost(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 750,
                    watchlist           = watchlist,
                    verbose             = 1,
                    maximize            = FALSE
)


t14.y <- t14$TARGET
t14 <- sparse.model.matrix(TARGET ~ ., data = t14)
dtrain <- xgb.DMatrix(data=t14, label=t14.y)
watchlist <- list(train=dtrain)


bst14 <- xgboost(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 750,
                    watchlist           = watchlist,
                    verbose             = 1,
                    maximize            = FALSE
)

t15.y <- t15$TARGET
t15 <- sparse.model.matrix(TARGET ~ ., data = t15)
dtrain <- xgb.DMatrix(data=t15, label=t15.y)
watchlist <- list(train=dtrain)


bst15 <- xgboost(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 750,
                    watchlist           = watchlist,
                    verbose             = 1,
                    maximize            = FALSE
)


t16.y <- t16$TARGET
t16 <- sparse.model.matrix(TARGET ~ ., data = t16)
dtrain <- xgb.DMatrix(data=t16, label=t16.y)
watchlist <- list(train=dtrain)


bst16 <- xgboost(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 750,
                    watchlist           = watchlist,
                    verbose             = 1,
                    maximize            = FALSE
)


t17.y <- t17$TARGET
t17 <- sparse.model.matrix(TARGET ~ ., data = t17)
dtrain <- xgb.DMatrix(data=t17, label=t17.y)
watchlist <- list(train=dtrain)


bst17 <- xgboost(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 750,
                    watchlist           = watchlist,
                    verbose             = 1,
                    maximize            = FALSE
)


t18.y <- t18$TARGET
t18 <- sparse.model.matrix(TARGET ~ ., data = t18)
dtrain <- xgb.DMatrix(data=t18, label=t18.y)
watchlist <- list(train=dtrain)


bst18 <- xgboost(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 750,
                    watchlist           = watchlist,
                    verbose             = 1,
                    maximize            = FALSE
)


t19.y <- t19$TARGET
t19 <- sparse.model.matrix(TARGET ~ ., data = t19)
dtrain <- xgb.DMatrix(data=t19, label=t19.y)
watchlist <- list(train=dtrain)


bst19 <- xgboost(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 750,
                    watchlist           = watchlist,
                    verbose             = 1,
                    maximize            = FALSE
)


t20.y <- t20$TARGET
t20 <- sparse.model.matrix(TARGET ~ ., data = t20)
dtrain <- xgb.DMatrix(data=t20, label=t20.y)
watchlist <- list(train=dtrain)


bst20 <- xgboost(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 750,
                    watchlist           = watchlist,
                    verbose             = 1,
                    maximize            = FALSE
)


t21.y <- t21$TARGET
t21 <- sparse.model.matrix(TARGET ~ ., data = t21)
dtrain <- xgb.DMatrix(data=t21, label=t21.y)
watchlist <- list(train=dtrain)


bst21 <- xgboost(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 750,
                    watchlist           = watchlist,
                    verbose             = 1,
                    maximize            = FALSE
)

t22.y <- t22$TARGET
t22 <- sparse.model.matrix(TARGET ~ ., data = t22)
dtrain <- xgb.DMatrix(data=t22, label=t22.y)
watchlist <- list(train=dtrain)


bst22 <- xgboost(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 750,
                    watchlist           = watchlist,
                    verbose             = 1,
                    maximize            = FALSE
)

t23.y <- t23$TARGET
t23 <- sparse.model.matrix(TARGET ~ ., data = t23)
dtrain <- xgb.DMatrix(data=t23, label=t23.y)
watchlist <- list(train=dtrain)


bst23 <- xgboost(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 750,
                    watchlist           = watchlist,
                    verbose             = 1,
                    maximize            = FALSE
)

t24.y <- t24$TARGET
t24 <- sparse.model.matrix(TARGET ~ ., data = t24)
dtrain <- xgb.DMatrix(data=t24, label=t24.y)
watchlist <- list(train=dtrain)


bst24 <- xgboost(   params             = param, 
                    data                = dtrain, 
                    nrounds             = 750,
                    watchlist           = watchlist,
                    verbose             = 1,
                    maximize            = FALSE
)



test$TARGET <- -1
test <- sparse.model.matrix(TARGET ~ ., data = test)

Model1 <- predict(bst1, test)
Model2 <- predict(bst2, test)
Model3 <- predict(bst3, test)
Model4 <- predict(bst4, test)
Model5 <- predict(bst5, test)
Model6 <- predict(bst6, test)
Model7 <- predict(bst7, test)
Model8 <- predict(bst8, test)
Model9 <- predict(bst9, test)
Model10 <- predict(bst10, test)
Model11 <- predict(bst11, test)
Model12 <- predict(bst12, test)
Model13 <- predict(bst13, test)
Model14 <- predict(bst14, test)
Model15 <- predict(bst15, test)
Model16 <- predict(bst16, test)
Model17 <- predict(bst17, test)
Model18 <- predict(bst18, test)
Model19 <- predict(bst19, test)
Model20 <- predict(bst20, test)
Model21 <- predict(bst21, test)
Model22 <- predict(bst22, test)
Model23 <- predict(bst23, test)
Model24 <- predict(bst24, test)

# Simple ensemble

preds.ensemble <- (Model1 + Model2 + Model3 + Model4 + Model5 + Model6 + Model7 + Model8
                   + Model9 + Model10 + Model11 + Model12 + Model13 + Model14 + Model15
                   + Model16 + Model17 + Model18 + Model19 + Model20 + Model21 + Model22
                   + Model23 + Model24)/24

# Always happy when this variable is under 23
preds.ensemble[test_var15 < 23] <- 0

submission <- data.frame(ID=test.id, TARGET=preds.ensemble)

write.csv(submission, "../Submissions/submission.csv", row.names = F)






