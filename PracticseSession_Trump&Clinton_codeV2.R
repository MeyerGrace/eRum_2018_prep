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
library(xgboost)
library(glmnet)
#library(lime) #you will need this later

## load data ####
tweet_csv <- read_csv("tweets.csv")
str(tweet_csv, give.attr = FALSE)

## first look data exploration ####
sort(table(tweet_csv$original_author), decreasing = TRUE)
table(tweet_csv$is_retweet, is.na(tweet_csv$original_author))
table(tweet_csv$handle)
table(tweet_csv$handle, tweet_csv$is_retweet)
table(tweet_csv$lang)
table(tweet_csv$handle, tweet_csv$lang)


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

str(tweet_data)

#show the non-lengthy columns
tweet_data %>%
  select(-c(text, source_url)) %>%
  head()

### data formatting ####

### show what tokenising is
example_text <- tweet_data$text[1]

quanteda::tokens(example_text, "word")

tokens(example_text, "sentence")


### create text corpus
tweet_corpus <- corpus(tweet_data)

# example: corpus object is easy to subset in order to get partial data
summary(corpus_subset(tweet_corpus, date > as_date('2016-07-01')), n =nrow(tweet_data))

# checking context of a chosen word 
kwic(tweet_corpus, "terror")
kwic(tweet_corpus, "immigrant*")
kwic(tweet_corpus, "famil*")
kwic(tweet_corpus, "amp") #ampersands!


## exploratory data vis ####
# visualize number and length of tweets 

#summary of quanteda corpus includes numbers of tokens and sentences, but not acutal tokens
#we can do analysis on this
tweet_summary <- summary(tweet_corpus, n =nrow(tweet_data))

str(tweet_summary)
head(tweet_summary)

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


# look by hour of the day- they both have a diurnal pattern, 
# DT seems to tweet later and then earlier & HC tweets many around midnight
# Potential time zone issues 

tweet_summary_tbl2 <- tweet_summary %>% 
  group_by(author, hour) %>% 
  summarize(no_tweets = n_distinct(Text),
            avg_words = mean(Tokens),
            avg_sentences = mean(Sentences)) 

tweet_summary_tbl2 %>%
  ggplot(aes(x = hour, y = no_tweets, fill = author, colour = author)) +
  geom_line() +
  geom_point() 


# create DFM
my_dfm <- dfm(tweet_corpus)
my_dfm[1:10, 1:5]

# top features 
topfeatures(my_dfm, 50)

# text cleaning
# edit tweets - remove URLs
edited_dfm <- dfm(tweet_corpus, remove_url = TRUE, remove_punct = TRUE, remove = stopwords("english"))
topfeatures(edited_dfm, 20)


#### creating wordclouds ####
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
set.seed(200)
textplot_wordcloud(by_author_dfm,
                   comparison = TRUE,
                   min.freq = 50,
                   random.order = FALSE,
                   rot.per = .25, 
                   colors = RColorBrewer::brewer.pal(8,"Dark2"))


#### modelling- split train and test, model and predict ####

#### separate the train and test set ####

# splitting data into train & text
# usually we would use caret for balanced, but it is a large package for a workshop 
set.seed(32984)
trainIndex <- sample.int(n = nrow(tweet_csv), size = floor(.8*nrow(tweet_csv)), replace = F)

train_tweets <- edited_dfm[ as.vector(trainIndex), ]
test_tweets <- edited_dfm[ -as.vector(trainIndex), ]

# check that the train and test set have the same 
all(train_tweets@Dimnames$features == test_tweets@Dimnames$features)

train_author <- ifelse(tweet_csv$handle[ as.vector(trainIndex)] =="realDonaldTrump", 1, 0)
test_author <- ifelse(tweet_csv$handle[ -as.vector(trainIndex)] =="realDonaldTrump", 1, 0)

length(train_author) == train_tweets@Dim[1]
length(test_author) == test_tweets@Dim[1]

table(train_author)


#### train the classification model ####

#try on glmnet
set.seed(1234)
glm_model <- glmnet(train_tweets, train_author, family = "binomial")
summary(glm_model)

preds <- predict(glm_model, test_tweets, s = 0.01, type = "response") # what is a good value of lambda?
preds <- ifelse(preds > 0.5, 1, 0)

# Accuracy
mean(preds == test_author)


### LIME on xgBoost model ####

# select only correct predictions
predictions_tbl <- preds %>% 
  as_tibble() %>%
  tibble::rownames_to_column()
names(predictions_tbl) <- c("rowname", "predict_label")

#THIS IS NOT WORKING NOW BECAUSE I DROPPED THE TEXT. NEED TO GET FROM JOINING TO preds
correct_pred <- test_author %>%
  as_tibble() %>% 
  rename(test_label = value) %>%
  tibble::rownames_to_column() %>% 
  left_join(predictions_tbl) %>%
  filter(test_label == predict_label) %>% 
  pull(text) %>% 
  head(4)

str(correct_pred)

#library(dplyr)
detach("package:dplyr", unload=TRUE)

library(lime)

explainer <- lime(train_tweets$text[1:4], 
                  model = xgb_model, 
                  preprocess = get_matrix)

corr_explanation <- lime::explain(correct_pred, 
                                  explainer, 
                                  n_labels = 1, n_features = 6, cols = 2, verbose = 0)
plot_features(corr_explanation)


