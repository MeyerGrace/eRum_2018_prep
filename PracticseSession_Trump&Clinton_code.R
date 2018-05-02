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
library(tidytext)
library(glmnet)
library(tm)
library(wordcloud)
library(devtools)
library(xgboost)
library(text2vec)

## load data ####
tweet_csv <- read_csv("tweets.csv")
str(tweet_csv, give.attr = FALSE)

## data exploration ####
# see original authors
sort(table(tweet_csv$original_author), decreasing = TRUE)
table(tweet_csv$lang)
table(tweet_csv$handle, tweet_csv$lang)
table(tweet_csv$handle)
table(tweet_csv$handle, tweet_csv$is_retweet)
table(tweet_csv$is_retweet, is.na(tweet_csv$original_author))

### data cleaning 
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
         tweet_num = row_number()
  ) %>% select(-timestamp)

str(tweet_data)

tweet_data %>%
  select(-c(text, source_url)) %>%
  head()

#table(tweet_data$author)


#show what tokenising is
example_text <- tweet_data$text[1]

tokens(example_text, "word")

tokens(example_text, "sentence")


#### QUANTEDA APPROACH ####
### create text corpus and the summary of it 
#(inlcudes numbers of tokens and sentences, but not acutal tokens)
tweet_corpus <- corpus(tweet_data)
tweet_summary <- summary(tweet_corpus, n =nrow(tweet_data))

str(tweet_summary)
head(tweet_summary)


# subsetting corpus
summary(corpus_subset(tweet_corpus, date > as_date('2016-07-01')), n =nrow(tweet_data))


# checking context of a chosen word 
kwic(tweet_corpus, "terror")
kwic(tweet_corpus, "immigrant*")
kwic(tweet_corpus, "famil*")



## exploratory data vis ####
# visualize number and length of tweets 

tweet_summary_tbl <- tweet_summary %>% 
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
textplot_wordcloud(edited_dfm, 
                   min.freq = 40, 
                   random.order = FALSE, 
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



### LIME on xgBoost model ####

# select only correct predictions
predictions_tbl <- xgb_preds %>% 
  as_tibble() %>% 
  rename_(predict_label = names(.)[1]) %>%
  tibble::rownames_to_column()

correct_pred <- test_tweets %>%
  tibble::rownames_to_column() %>% 
  mutate(test_label = author == "realDonaldTrump") %>%
  left_join(predictions_tbl) %>%
  filter(test_label == predict_label) %>% 
  pull(text) %>% 
  head(4)

str(correct_pred)

#library(dplyr)
detach("package:dplyr", unload=TRUE)

library(lime)

explainer <- lime(train_tweets$text, 
                  model = xgb_model, 
                  preprocess = get_matrix)

corr_explanation <- lime::explain(correct_pred, 
                                  explainer, 
                                  n_labels = 1, n_features = 6, cols = 2, verbose = 0)
plot_features(corr_explanation)


