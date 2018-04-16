################################################################################################
# Building an Interpretable NLP model to classify tweets Workshop
# full code 
# eRum 2018, Budapest 
################################################################################################


# load packages ####
library(readr)
library(quanteda)
library(dplyr)
library(stringr)
library(lubridate)
library(ggplot2)
library(caret)


## load data ####
tweet_csv <- read_csv("tweets.csv")
str(tweet_csv)

## data exploration ####
# see original authors
table(tweet_csv$original_author)
table(tweet_csv$lang)
table(tweet_csv$handle, tweet_csv$lang)
table(tweet_csv$handle)
table(tweet_csv$handle, tweet_csv$is_retweet)
table(tweet_csv$is_retweet, is.na(tweet_csv$original_author))

### data cleaning 
tweet_data <- tweet_csv %>% 
  filter(is_retweet == "False") %>%
  select(author = handle, text, retweet_count, favorite_count, source_url, timestamp = time) %>% 
  mutate(date = as_date(str_sub(timestamp, 1, 10)),
         hour = hour(hms(str_sub(timestamp, 12, 19)))
  ) %>% select(-timestamp)

table(tweet_data$author)

### create text corpus and document term matrix
tweet_corpus <- corpus(tweet_data)
tweet_summary <- summary(tweet_corpus, n =nrow(tweet_data))
str(tweet_summary)


# subsetting corpus
summary(corpus_subset(tweet_corpus, date > as_date('2016-07-01')), n =nrow(tweet_data))


# checking context of a chosen word 

kwic(tweet_corpus, "terror")
kwic(tweet_corpus, "immigrant*")
kwic(tweet_corpus, "famil*")


## exploratory data vis ####
# visualize number and length of tweets 

tweet_summary_tbl = tweet_summary %>% 
  group_by(author, date) %>% 
  summarize(no_tweets = n_distinct(Text),
            avg_words = mean(Tokens),
            avg_sentences = mean(Sentences))

tweet_summary_tbl %>% 
  ggplot(aes(x = date, y = no_tweets, fill = author, colour = author)) +
  geom_line() +
  geom_point() 


tweet_summary_tbl %>% 
  ggplot(aes(x = date, y = avg_words, fill = author, colour = author)) +
  geom_line() +
  geom_point() 


tweet_summary_tbl %>% 
  ggplot(aes(x = date, y = avg_sentences, fill = author, colour = author)) +
  geom_line() +
  geom_point() 


# look by hour of the day- they both have a diurnal pattern, but DT seems to tweet later and then earlier. 
#HC tweets many around midnight 
if("hour" %in% names(tweet_summary)) {
tweet_summary_tbl2 <- tweet_summary %>% 
  group_by(author, hour) %>% 
  summarize(no_tweets = n_distinct(Text),
            avg_words = mean(Tokens),
            avg_sentences = mean(Sentences)) 

tweet_summary_tbl2 %>%
  ggplot(aes(x = hour, y = no_tweets, fill = author, colour = author)) +
  geom_line() +
  geom_point() 
}

# create DFM
my_dfm <- dfm(tweet_corpus)
my_dfm[1:10, 1:5]

# top features 
topfeatures(my_dfm, 20)

# text cleaning
# edit tweets - remove URLs
edited_dfm <- dfm(tweet_corpus, remove_url = TRUE, remove_punct = TRUE, remove = stopwords("english"))
topfeatures(edited_dfm, 20)


# getting a wordcloud
set.seed(100)
textplot_wordcloud(edited_dfm, min.freq = 40, random.order = FALSE,
                   rot.per = .25, 
                   colors = RColorBrewer::brewer.pal(8,"Dark2"))


### getting a wordcloud by author
## grouping by author - see differences!!!!
by_author_dfm <- dfm(tweet_corpus,
                     groups = "author",
                     remove = stopwords("english"), remove_punct = TRUE, remove_url = TRUE)

by_author_dfm[1:2,1:10]


# wordcloud by author 
set.seed(100)
#?textplot_wordcloud
textplot_wordcloud(by_author_dfm,
                   comparison = TRUE,
                   min.freq = 50,
                   random.order = FALSE,
                   rot.per = .25, 
                   colors = RColorBrewer::brewer.pal(8,"Dark2"))


#### modelling ####

#### separate the train and test set ####

edited_dfm[1:10, 1:10]
table(tweet_data$author)

tweets_tokens <- cbind(Label = tweet_data$author, data.frame(edited_dfm)) %>%
  mutate(Label = as.factor(ifelse(Label == "HillaryClinton", 1, 0))) %>%
  mutate(Label = as.factor(Label)) %>%
  select(-document)

str(tweets_tokens)

set.seed(32984)
indexes <- createDataPartition(tweets_tokens$Label, times = 1,
                               p = 0.7, list = FALSE)

trainData <- tweets_tokens[indexes,]
testData <- tweets_tokens[-indexes,]
str(trainData)

#### train the model with dfm ####
# random forest not suitable for text classification - doesn't deal well with high-dimensional, sparse data, SVM or naive bayes are a better start
# http://fastml.com/classifying-text-with-bag-of-words-a-tutorial/
# other algos take too long to train

# time your model
library(microbenchmark)

?microbenchmark
microbenchmark(nb_model <- train(Label ~ ., data = trainData, method = 'nb'), times = 2)
system.time({ nb_model <- train(Label ~ ., data = trainData, method = 'nb') })




### train using text2vec DTM ####


library(text2vec) 
library(qdapRegex)

str(tweet_csv)

all_tweets <- tweet_csv %>% 
  filter(str_to_lower(is_retweet) == "false") %>% 
  rename(author = handle) %>% 
  select(author, text) %>% 
  mutate(text = qdapRegex::rm_url(text)) %>% #removes URLs from text
  na.omit()

table(all_tweets$author)

# splitting data into train & text
set.seed(32984)
trainIndex <- createDataPartition(all_tweets$author, p = .8, 
                                  list = FALSE, 
                                  times = 1)

train_tweets <- all_tweets[ trainIndex,]
test_tweets <- all_tweets[ -trainIndex,]

# tokenization & creating a dtm
get_matrix <- function(text) {
  it <- itoken(text, progressbar = TRUE)
  create_dtm(it, vectorizer = hash_vectorizer())
}

dtm_train <- get_matrix(train_tweets$text)
dtm_test <- get_matrix(test_tweets$text)
train_labels <- train_tweets$author == "realDonaldTrump"

####  xgboost ####

library(xgboost) 


param <- list(max_depth = 7, 
              eta = 0.1, 
              objective = "binary:logistic", 
              eval_metric = "error", 
              nthread = 1)
?xgb.train

set.seed(1234)
xgb_model <- xgb.train(
  param, 
  xgb.DMatrix(dtm_train, label = train_labels),
  nrounds = 50,
  verbose=0
)


# We use a (standard) threshold of 0.5
xgb_preds <- predict(xgb_model, dtm_test) > 0.5
test_labels <- test_tweets$author == "realDonaldTrump"


# Accuracy
print(mean(xgb_preds == test_labels))


# other than xgboost models ####

### logistic regressin using glmnet

library(glmnet)

set.seed(1234)
glm_model <- glmnet(dtm_train, train_labels, family = "binomial")

# We use a (standard) threshold of 0.5
glm_preds <- predict(glm_model, dtm_test) > 0.5

# Accuracy
print(mean(glm_preds == test_labels))


### SVM
library(e1071)
library(SparseM)

svm_model <- e1071::svm(dtm_train, as.numeric(train_labels), kernel='linear')
svm_preds <- predict(svm_model, dtm_test) > 0.5

#library(sparseSVM)
#ssvm_model <- cv.sparseSVM(dtm_train, as.numeric(train_labels))

# Accuracy
print(mean(glm_preds == test_labels))



### LIME on XGboost models ####


library(dplyr)

# select only correct predictions
predictions_tbl <- xgb_preds %>% as_tibble() %>% 
  rename_(predict_label = names(.)[1]) %>%
  tibble::rownames_to_column()

correct_pred <- test_tweets %>%
  tibble::rownames_to_column() %>% 
  mutate(test_label = author == "realDonaldTrump") %>%
  left_join(predictions_tbl) %>%
  filter(test_label == predict_label) %>% 
  pull(text) %>% 
  head(4) # it needs to be 5 or less, otherwise corr_explanation returns an error, why?

str(correct_pred)



detach("package:dplyr", unload=TRUE)

library(lime)

explainer <- lime(correct_pred, model = xgb_model, 
                  preprocess = get_matrix)

corr_explanation <- lime::explain(correct_pred, explainer, n_labels = 1, 
                                  n_features = 6, cols = 2, verbose = 0)
plot_features(corr_explanation)

