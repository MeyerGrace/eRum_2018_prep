library(dplyr)
library(stringr)
library(quanteda)
library(lime)
library(readr)
library(lubridate)
#data prep
tweet_csv <- read_csv("tweets.csv")
# creating corpus and dfm for train and test sets
### data cleaning ####
tweet_data <- tweet_csv %>% 
  #  filter(is_retweet == "False") %>%
  select(author = handle,
         text,
         retweet_count,
         favorite_count,
         source_url,
         timestamp = time) %>% 
  mutate(date = as_date(str_sub(timestamp, 1, 10)),
         hour = hour(hms(str_sub(timestamp, 12, 19))),
         tweet_num = row_number()) %>% 
  select(-timestamp)
get_matrix <- function(df){
  corpus <- quanteda::corpus(df)
  dfm <- quanteda::dfm(corpus, remove_url = TRUE, remove_punct = TRUE,     remove = stopwords("english"))
}
set.seed(32984)
trainIndex <- sample.int(n = nrow(tweet_data), size =     floor(.8*nrow(tweet_data)), replace = F)
train_dfm <- get_matrix(tweet_data$text[trainIndex])
train_raw <- tweet_data[, c("text", "tweet_num")][as.vector(trainIndex), ]
train_labels <- tweet_data$author[as.vector(trainIndex)] == "realDonaldTrump"
test_dfm <- get_matrix(tweet_data$text[-trainIndex])
test_raw <- tweet_data[, c("text", "tweet_num")][-as.vector(trainIndex), ]
test_labels <- tweet_data$author[-as.vector(trainIndex)] == "realDonaldTrump"
#### make sure that train & test sets have exactly same features
test_dfm <- dfm_select(test_dfm, train_dfm)
### Naive Bayes model using quanteda::textmodel_nb ####
nb_model <- quanteda::textmodel_nb(train_dfm, train_labels)

nb_preds <- predict(nb_model, test_dfm) #> 0.5
# select only correct predictions
predictions_tbl <- data.frame(predict_label = nb_preds$nb.predicted,
                              actual_label = test_labels,
                              tweet_name = rownames(nb_preds$posterior.prob)
) %>%
  mutate(tweet_num = 
           as.integer(
             str_trim(
               str_replace_all(tweet_name, "text", ""))
           )) 
correct_pred <- predictions_tbl %>%
  filter(actual_label == predict_label) 
# pick a sample of tweets for explainer 
tweets_to_explain <- test_raw %>%
  filter(tweet_num %in% correct_pred$tweet_num) %>% 
  head(4)
### set up correct model class and predict functions 
class(nb_model)

### in older versions of quanteda the output is "textmodel_nb_fitted", then you modify the model for LIME as follows:

model_type.textmodel_nb <- function(x, ...) {
  return("classification")
}

# have to modify the textmodel_nb_fitted so that 
predict_model.textmodel_nb <- function(x, newdata, type, ...) {
  X <- dfm_select(dfm(newdata), x$data$x)   
  res <- predict(x, newdata = X, ...)
  switch(
    type,
    raw = data.frame(Response = res$nb.predicted, stringsAsFactors = FALSE),
    prob = as.data.frame(res$posterior.prob, check.names = FALSE)
  )  
}


### in newer versions of quanteda the output is "textmodel_nb" "textmodel"    "list", then you modify the model for LIME as follows:


model_type.textmodel_nb <- function(x, ...) {
  return("classification")
}

  
predict_model.textmodel_nb <- function(x, newdata, type, ...) {
    X <- dfm_select(dfm(newdata), x$x)   
    res <- predict(x, newdata = X, ...)
    switch(
      type,
      raw = data.frame(Response = res$nb.predicted, stringsAsFactors = FALSE),
      prob = as.data.frame(res$posterior.prob, check.names = FALSE)
    )  
  }



### run the explainer - no problems here 
explainer <- lime(train_raw$text, 
                  model = nb_model,
                  preprocess = get_matrix) 

corr_explanation <- lime::explain(tweets_to_explain$text, 
                                  explainer, 
                                  n_labels = 1,
                                  n_features = 6,
                                  cols = 2,
                                  verbose = 0)

corr_explanation[1:5, 1:5]
plot_features(corr_explanation)
