#https://stackoverflow.com/questions/36974772/implementing-naive-bayes-for-text-classification-using-quanteda?rq=1
#https://rpubs.com/mr148/309859
secondScript <- FALSE

# Load libraries ####
library(readr)
library(quanteda)
library(dplyr)
library(stringr)
library(lubridate)
library(ggplot2)
library(Amelia)
library(mice)
library(caret)
library(rpart)
options(scipen = 999)


# Read data ####
tweet_csv <- read_csv("tweets.csv")
str(tweet_csv)
head(tweet_csv)

# Data Exploration ####
table(tweet_csv$handle)
table(tweet_csv$handle, tweet_csv$is_retweet)
table(tweet_csv$is_retweet, is.na(tweet_csv$original_author))
table(tweet_csv$original_author) 
table(tweet_csv$lang)
table(tweet_csv$handle, tweet_csv$lang)

# look into missing data
missmap(tweet_csv)
# about half of the data is very missing and will not be useful to us. Most of the missing columns are meta data such as
#lat/long. But also retweeted author which is fine as we will remove these tweets later
md.pattern(tweet_csv) %>% View() #this doesn't seem to match the missmap information which is interesting (extended_entities)

#select the data we think is interesting.
# remove retweets
#I kept time as maybe we can feature engineer on time of day, as well as days to the election? 
tweet_data <- tweet_csv %>%
  filter(is_retweet == "False") %>%
  select(author = handle, text, lang, retweet_count, favorite_count, source_url, timestamp = time) %>% 
  mutate(date = as_date(str_sub(timestamp, 1, 10)),
         hour = hour(hms(str_sub(timestamp, 12, 19)))) %>% 
  select(-timestamp)

table(tweet_data$author)
range(tweet_data$date) # tweets finish before the election on Nov 8th. Nominations were finalised in June. 
summary(tweet_data)




tweet_corpus <- corpus(tweet_data)
#The corpus object is made up of 4 elements
#1) documents which is our data frame from before
#2) metadata- has two pieces of information of source (my directory) and created (when i created the corpus)
#3) settings- how the text has been treated so far, includes dictionaries (currently NULL) and whether text is stemmed
#4) tokens- currently NULL

#create a summary of each of the lines. Tokens, Typs and Sentences are now in this str which I am unsure where they are stored
summary(tweet_corpus, n =nrow(tweet_data))
tweet_summary <- summary(tweet_corpus, n =nrow(tweet_data)) #
str(tweet_summary)


# subsetting corpus (2576 tweets after here)
summary(corpus_subset(tweet_corpus, date > as_date('2016-07-01')), n =nrow(tweet_data))


# checking context of a chosen word 
kwic(tweet_corpus, "terror")
kwic(tweet_corpus, "immigrant*")
kwic(tweet_corpus, "famil*")


# visualize number and length of tweets 

tweet_summary_tbl <- tweet_summary %>% 
  group_by(author, date) %>% 
  summarize(no_tweets = n_distinct(Text),
            avg_words = mean(Tokens),
            avg_sentences = mean(Sentences))
#do we have an issue with DT having three months more tweets than HC?

#they on average roughly have the same amount of words, DT has more sentences though
tweet_summary %>% 
  group_by(author) %>%
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
tweet_summary_tbl3 <- tweet_summary %>% 
  mutate(hour = hour(time)) %>% 
  group_by(author, hour) %>% 
  summarize(no_tweets = n_distinct(Text),
            avg_words = mean(Tokens),
            avg_sentences = mean(Sentences)) 

tweet_summary_tbl3 %>%
ggplot(aes(x = hour, y = no_tweets, fill = author, colour = author)) +
  geom_line() +
  geom_point() 

# create DFM (document feature matrix)

my_dfm <- dfm(tweet_corpus)
my_dfm[1:10, 1:5]
sum(my_dfm[1, ] > 0) #the first tweet goes from 27 tokens to 23 different words

# top features 
topfeatures(my_dfm, 20)

# text cleaning
# edit tweets - remove URLs
edited_dfm <- dfm(tweet_corpus, remove_url = TRUE, remove_punct = TRUE, remove = stopwords("english"))
edited_dfm[1:10, 1:5]
topfeatures(edited_dfm, 20)


# getting a wordcloud

set.seed(100)
# we need a sample, takes too long to plot
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
# modify - takes too long to plot
set.seed(100)
?textplot_wordcloud
textplot_wordcloud(by_author_dfm,
                   comparison = TRUE,
                   min.freq = 50,
                   random.order = FALSE,
                   rot.per = .25, 
                   colors = RColorBrewer::brewer.pal(8,"Dark2"))

# data prep for modelling (dplyr?)


############################################################################
##second script that I wrote but just wanted to keep in one place in the repo. Need to tidy later
if(secondScript){
  
  tweet_data <- tweet_csv %>%
    filter(is_retweet == "False") %>%
    select(author = handle, text, lang, retweet_count, favorite_count, source_url, timestamp = time) %>% 
    mutate(date = as_date(str_sub(timestamp, 1, 10)),
           hour = hour(hms(str_sub(timestamp, 12, 19)))) %>% 
    select(-timestamp)
  
  tweet_corpus <- corpus(tweet_data)
  
  
  tweet_dfm <- dfm(tweet_corpus, remove_url = TRUE, remove_punct = TRUE, remove = stopwords("english"))
  tweet_dfm[1:5,1:10]
  
  str(data.frame(tweet_dfm))
  
  all_classes <- docvars(tweet_corpus)$author
  
  tweets_tokens <- cbind(Label = all_classes, data.frame(select(tweet_dfm, -document))) %>%
    mutate(Label = ifelse(Label == "HillaryClinton", 1, 0)) %>%
    mutate(Label = as.factor(Label)) %>%
    select(-document)
  str(tweets_tokens)
  
  
  #### separate the train and test set ####
  set.seed(32984)
  indexes <- createDataPartition(tweets_tokens$Label, times = 1,
                                 p = 0.7, list = FALSE)
  
  trainData <- tweets_tokens[indexes,]
  testData <- tweets_tokens[-indexes,]
  str(trainData)
  
  #### train the model ####
  rpartModel <- rpart(Label ~ ., 
                      data = train,
                      method = "class")
  
  
  rfModel <- train(Label ~ ., data = trainData, method = 'rf')
  
  #### predict for the test set ####
  prediction3 <- predict(rpartModel, test, type = "class")
  
  #combine with the Label for writing out
  prediction3 <- data.frame(Actual = test$Label,
                            Prediction = prediction3)
  prediction3 <- prediction3 %>%
    mutate(Accurate= ifelse(Actual == Prediction, 1, 0))
  
  mean(prediction3$Accurate)
  
  
  
  
  
  
  # Now try LIME
  #detach("package:dplyr", unload=TRUE)
  
  library(lime)
  
  
  explainer <- lime(train, rpartModel)
  explaination <- explain(test, explainer, n_labels = 1, n_features = 6)
  
  
  clean_sentence_to_explain <- head(clean_test_tweets[clean_test_labels,]$clean_text, 5)
  clean_explainer <- lime(clean_sentence_to_explain, model = clean_xgb_model2, 
                          preprocess = get_matrix2)
  clean_explanation <- explain(clean_sentence_to_explain, clean_explainer, n_labels = 1, 
                               n_features = 5)
  
  # prediction <- predict(rpart1, newdata = test)
  # 
  # trainclass <- factor(replace(all_classes, 1780:length(all_classes), NA))
  # bbcNb <- textmodel_nb(bbc_dfm, trainclass)
}