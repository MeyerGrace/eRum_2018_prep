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

train_dfm <- edited_dfm[as.vector(trainIndex), ]
train_raw <- tweet_data[as.vector(trainIndex), ]
train_label <- train_raw$author == "realDonaldTrump"
table(train_raw$author)

test_dfm <- edited_dfm[-as.vector(trainIndex), ]
test_raw <- tweet_data[-as.vector(trainIndex), ]
test_label <- test_raw$author == "realDonaldTrump"
table(test_raw$author)

#### make sure that train & test sets have exactly same features
test_dfm <- dfm_select(test_dfm, train_dfm)

# check that the train and test set have the same 
all(train_dfm@Dimnames$features == test_dfm@Dimnames$features)

#train_author <- ifelse(tweet_csv$handle[ as.vector(trainIndex)] =="realDonaldTrump", 1, 0)
#test_author <- ifelse(tweet_csv$handle[ -as.vector(trainIndex)] =="realDonaldTrump", 1, 0)

#length(train_author) == train_tweets@Dim[1]
#length(test_author) == test_tweets@Dim[1]

#table(train_author)


#### train the classification model ####


### Naive Bayes model using quanteda::textmodel_nb ####
nb_model <- quanteda::textmodel_nb(train_dfm, train_labels)
nb_preds <- predict(nb_model, test_dfm) #> 0.5

# Accuracy
print(mean(nb_preds$nb.predicted == test_labels))



# bonus: glmnet ####
#set.seed(1234)
#glm_model <- glmnet(train_dfm, train_label, family = "binomial")

# We use a (standard) threshold of 0.5
#glm_preds <- predict(glm_model, test_dfm) > 0.5

# Accuracy
#print(mean(glm_preds == test_label))


### LIME on xgBoost model ####

# select only correct predictions
predictions_tbl <- data.frame(predict_label = nb_preds$nb.predicted,
                              actual_label = test_labels,
                              tweet_name = rownames(nb_preds$posterior.prob)
) %>%
  mutate(tweet_num = 
           as.integer(
             str_trim(
               str_replace_all(tweet_name, "text", ""))
         )) %>%
  inner_join(
    select(tweet_data, tweet_num, text)
    )


correct_pred <- predictions_tbl %>%
  filter(actual_label == predict_label) 

## check if correct tweet numbers agree with total accuracy
str(correct_pred)
nrow(correct_pred)/length(test_labels) # they do!

tweets_to_explain <- correct_pred %>% 
  select(text) %>% 
  head(4)

#library(dplyr)
detach("package:dplyr", unload=TRUE)

library(lime)

#THIS IS NOT WORKING NOW BECAUSE I DIDN'T DO IT THROUGH A FUNCTION

class(nb_model)

model_type.textmodel_nb_fitted <- function(x, ...) {
  return("classification")
}


get_matrix <- function(df){
  corpus <- corpus(df)
  dfm <- dfm(corpus, remove_url = TRUE, remove_punct = TRUE, remove = stopwords("english"))
}


explainer <- lime(train_raw[1:5], # lime returns error on different features in explainer and explanations, even if I use the same dataset in both. Raised an issue on Github and asked a question on SO
                  model = nb_model,
                  preprocess = get_matrix) 

corr_explanation <- lime::explain(train_raw[1:5], 
                                  explainer, 
                                  n_labels = 1,
                                  n_features = 6,
                                  cols = 2,
                                  verbose = 0)

plot_features(corr_explanation)


