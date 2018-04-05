library(readr)
library(quanteda)
library(dplyr)
library(stringr)
library(lubridate)
library(ggplot2)
library(h2o)
library(data.table)
library(slam)
library(caret)
library(tidytext)
library(reshape2)

## load data 
tweet_csv <- read_csv("tweets.csv")
str(tweet_csv)

# see original authors
table(tweet_csv$original_author)

### data cleaning 
tweet_data <- select(tweet_csv,  author = handle, text, retweet_count, favorite_count, source_url, timestamp = time) %>% 
  mutate(date = as_date(str_sub(timestamp, 1, 10))#,
         #time = hms(str_sub(timestamp, 12, 19))
  ) %>% select(-timestamp)

table(tweet_data$author)
summary(tweet_data)

### create text corpus and document term matrix
tweet_corpus <- corpus(tweet_data)
tweet_summary <- summary(tweet_corpus, n =nrow(tweet_data))
str(tweet_summary)


# subsetting corpus
summary(corpus_subset(tweet_corpus, date > as_date('2016-07-01')), n =nrow(tweet_data))


# checking context of a chosen word 

kwic(tweet_corpus, "terror")
kwic(tweet_corpus, "immigrant*")


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

textplot_wordcloud(edited_dfm, min.freq = 20, random.order = FALSE,
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



#### get the dtm for modelling in H20 ####

my_dfm
my_df <- as.data.frame(my_dfm)

str(my_df)

dfm_h2o <- as.h2o(my_df) ### returns an error for duplicted column 000, check later

split_h2o <- h2o.splitFrame(bc_h2o, c(0.7, 0.15), seed = 13 ) #splits data into random 70%/15%15% chunks

train_h2o <- h2o.assign(split_h2o[[1]], "train" ) # 70%
valid_h2o <- h2o.assign(split_h2o[[2]], "valid" ) # 15%
test_h2o  <- h2o.assign(split_h2o[[3]], "test" )  # 15%


### get autoML() going

#h2o.init()


### data prep for modelling in caret ####

text_data <- tweet_csv %>% 
  select(text, handle) %>% 
  mutate(isTrump = handle == "realDonaldTrump") %>% 
  select(-handle)

str(text_data)
head(text_data)

### create train and test set



a <- createDataPartition(text_data$isTrump, p = 0.8, list=FALSE)
train <- text_data[a,]
test <- text_data[-a,]

str(train)
table(train$isTrump)

str(test)
table(test$isTrump)

### create text corpus and document term matrix ####

train_corpus <- corpus(train)
test_corpus <- corpus(test)


# Create a document term matrix.
train_dfm <-  dfm(train_corpus, remove_url = TRUE, remove_punct = TRUE, remove = stopwords("english"))
train_df <- as.data.frame(train_dfm)
train_m <- as.matrix(train_dfm)


train_df$isTrump <- train$isTrump
str(train_df)





### tokenise using dplyr ####

train_tokens <- train %>% 
  tibble::rownames_to_column() %>% 
  rename(tweet_num = rowname) %>% 
  unnest_tokens(word, text) %>% 
  dcast(tweet_num + isTrump ~ word, fun.aggregate = mean)

train_tokens2 <- train %>% 
  unnest_tokens(ngram, text, token = "ngrams", n = 2, n_min = 1)

test_tokens <- test %>% 
  unnest_tokens(word, text)

test_tokens2 <- test %>% 
  unnest_tokens(ngram, text, token = "ngrams", n = 2, n_min = 1)


### train the model ####

# Train using caret
bayes_fit <- train(isTrump ~ ., data = train_df, method = 'bayesglm')
boost_fit <- train(isTrump ~ ., data = train_df, method = 'adaboost')
xgb_fit <- train(isTrump ~ ., data = train_df, method = 'xgbTree')
rf_fit <- train(isTrump ~ ., data = train_df, method = 'rf')
lr_fit <- train(isTrump ~ ., data = train_df, method='glm', family='binomial')


?train

product_review_model <- train_model(product_review_container, algorithm = "SVM")
train_model(product_review_container, algorithm = "MAXENT")

