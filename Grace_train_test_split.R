library(readr)
library(quanteda)
library(caret)
library(glmnet)
library(dplyr)


### links ####
#https://stackoverflow.com/questions/38755207/working-with-text-classification-and-big-sparse-matrices-in-r
#In fact quantedas dfm-class inherits from dgCMatrix-class. So if your code works with dfm-class, in most cases it will work with dgCMatrix as well

### Make reproducible answer for spliting train and test ####
myText <- data.frame(label = c("cat", "dog", "cat", "dog"), 
                     text = c("I have a cat", "I have a dog", "I hate my cat", "I hate my dog"))
myText$text <- as.character(myText$text)
str(myText)

myCorpus <- corpus(myText)

summary(myCorpus)

tokens(myText[1, "text"], "word")

myDfm <- dfm(myCorpus)
myDfm

myDfm[1:2,]

set.seed(32984)
trainIndex <- createDataPartition(myText$label, p = .5, 
                                  list = FALSE, 
                                  times = 1)
trainIndex

myTrain <- myDfm[as.vector(trainIndex),]
myTrain
myTest <- myDfm[-as.vector(trainIndex),]
myTest
all(myTrain@Dimnames$features == myTest@Dimnames$features)



##### test on the real data set ####

tweet_csv <- read_csv("tweets.csv")
tweet_corpus <- corpus(tweet_csv)
edited_dfm <- dfm(tweet_corpus, remove_url = TRUE, remove_punct = TRUE, remove = stopwords("english"))
edited_dfm[1:10, 1:10]
nrow(tweet_csv) == nrow(edited_dfm)
# splitting data into train & text
set.seed(32984)
trainIndex <- createDataPartition(tweet_csv$handle, p = .8, 
                                  list = FALSE, 
                                  times = 1)

train_tweets <- edited_dfm[ as.vector(trainIndex),]
test_tweets <- edited_dfm[ -as.vector(trainIndex),]
train_tweets[1:10, 1:10]
test_tweets[1:10, 1:10]
all(train_tweets@Dimnames$features == test_tweets@Dimnames$features)

train_author <- ifelse(tweet_csv$handle[ as.vector(trainIndex)] =="realDonaldTrump", 1, 0)
test_author <- ifelse(tweet_csv$handle[ -as.vector(trainIndex)] =="realDonaldTrump", 1, 0)



#try on glmnet
set.seed(1234)
glm_model <- glmnet(train_tweets, train_author, family = "binomial")
glm_preds <- predict(glm_model, test_tweets)
glm_preds <- ifelse(glm_preds > 0.5, 1, 0)# todo finish
summary(glm_model)

# Accuracy
mean(glm_preds == test_author)

# try using caret
# THIS DOES NOT WORK, TAKES FOREVER
cls.ctrl <- trainControl(method = "repeatedcv", #boot, cv, LOOCV, timeslice OR adaptive etc.
                         number = 10, repeats = 5,
                         classProbs = TRUE, summaryFunction = twoClassSummary,
                         savePredictions = "final", allowParallel = TRUE)

set.seed(1895)
glm.fit <- train(x = as.matrix(train_tweets), y = train_author, 
                 method = "glm", family = "binomial")
# glm.fit <- train(x = train_tweets, y = train_author, trControl = cls.ctrl,
#                  method = "glm", family = "binomial", metric = "ROC",
#                  preProcess = c("nzv", "center", "scale"))
glm.fit

### test for caret on iris sparce matrix ####
library(tidyr)
irisSub <- iris %>%
  mutate(label = ifelse(Species == "setosa", "Setosa", "not"),
         label = as.factor(label),
         iris_num = row_number()) %>%
  select(-Species) %>%
  gather(variable, value, -iris_num) %>%
  mutate(variable = as.factor(variable),
         var_num = as.numeric(variable))

irisDependent <- irisSub %>%
  filter(variable != "label") %>%
  mutate(value = as.numeric(value))

irisResponse <- irisSub %>%
  filter(variable == "label") 
  
dn <- list( unique(irisDependent$iris_num), levels(irisDependent$variable)[-1])
irisSparse <- sparseMatrix(i = irisDependent$iris_num, 
                           j = irisDependent$var_num -1, #because label took 1st pos
                           x = irisDependent$value,
                           dimnames = dn)


set.seed(32984)
trainIndex <- createDataPartition(irisResponse$value, p = .8, 
                                  list = FALSE, 
                                  times = 1)
trainIris <- irisSparse[as.vector(trainIndex),]
testIris <- irisSparse[-as.vector(trainIndex),]

trainIrisLabel <- irisResponse$value[trainIndex]
testIrisLabel <- irisResponse$value[trainIndex]


cls.ctrl <- trainControl(method = "repeatedcv", #boot, cv, LOOCV, timeslice OR adaptive etc.
                         number = 10, repeats = 5,
                         classProbs = TRUE, summaryFunction = twoClassSummary,
                         savePredictions = "final", 
                         allowParallel = TRUE,
                         returnData = FALSE)

set.seed(1895)
glm.fit <- train(x = trainIris, y = trainIrisLabel, trControl = cls.ctrl,
                 method = "glm", family = "binomial", metric = "ROC")
# I get this error:
#   Error in as.data.frame.default(x) : 
#   cannot coerce class "structure("dgCMatrix", package = "Matrix")" to a data.frame
# In addition: There were 50 or more warnings (use warnings() to see the first 50)
# Timing stopped at: 0.01 0 0.01
