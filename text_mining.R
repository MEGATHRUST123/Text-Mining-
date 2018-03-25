# Practice from https://www.youtube.com/watch?v=4vuw0AsHeGw&list=PL8eNk_zTBST8olxIRFoo0YeXxEOkYdoxi
# Install packages 
install.packages(c("ggplot2","e1071","caret","quanteda","irlba","randomForest"))
setwd("C:/Users/megathrust/Desktop")
# Dont convert strings into categories/factors
spam=read.csv("spam.csv",stringsAsFactors = FALSE)
View(spam)

# Cleaning up the dataset
# There are unused columns in the dataset
# Naming the columns
cleans <- spam[,1:2] 
colnames(cleans)=c("label","Text")
View(cleans)

# Check for missing values
# Complete.cases returns a logical vector which cases are complete
length(which(complete.cases(spam)== FALSE))
# There is no missing data 

# Data Exploration
# Convert out labels into factors
cleans$label=as.factor(cleans$label)
levels(cleans$label)
# Proportional table -tells you how many spam messages there are
prop.table(table(cleans$label))

# Length of text - feature engineering 
# nchar - count the number of characters
cleans$textlength = nchar(cleans$Text)
View(cleans)
# Average length of text - ham and spam messages
summary(cleans$textlength)
# maximum number of text is 910
# Medium os 61 while mean is 90, possibilty of skewness -skewed right
# large disparity between the max and medium is also another telling
# sign of skewness
# Using histogram to look at the distribution
hist(cleans$textlength,breaks=100)
# Using ggplot
library(ggplot2)
qplot(cleans$textlength, geom="histogram",binwidth=5,fill=cleans$label) 

# Model validation: Creating training and texting set with package - caret
library(caret)
help(package = "caret")
# set.seed to make result reproducible
set.seed(32984)
# list=False: gimme the real values of variables 
indexes = createDataPartition(y=cleans$label,times=1,p=0.8,list=FALSE)
train<-cleans[indexes,]
test<-cleans[-indexes,]

# Verify proportions -ensure that train and test sets have similar proportions of "ham" and "spam" messages as the original data
# Ensure that the train and test set follows tha same distribution as the original data
prop.table(table(cleans$label))
prop.table(table(train$label))
prop.table(table(test$label))

qplot(train$textlength, geom="histogram",binwidth=5,fill=train$label) 
qplot(test$textlength, geom="histogram",binwidth=5,fill=test$label) 

# Deciding what words to keep and remove depends on domain of knowledge
# Some words are more important than others depending on your context
# Important to do text pre-processing

# Load Quanteda - Quantitative Analysis of Textual Data
library(quanteda)
help(package="quanteda")

# Create tokens 
train_token = tokens(train$Text,what="word",remove_symbols=TRUE, remove_punct=TRUE,remove_numbers=TRUE,remove_hyphens=TRUE)

# Data preprocessing pipelines - repeat a set of instructions to alot of items 
# Lower case everything to make text easier to analyse
train_token = tokens_tolower(train_token)

# Remove stopwords
# Remove preset stopwords
# Caution, the Quanteda's stopwords may contains words that are important in your context
train_tokens=tokens_select(train_token,stopwords(),selection = "remove")

# Stemming 
train_tokens=tokens_wordstem(train_tokens, language = "english")

# Document Frequency Matrix
train.tokens.dlm <- dfm(train_tokens, tolower=FALSE)
train.tokens.dfm <- as.matrix(train.tokens.dlm)
dim(train.tokens.dfm)
# Problem: Text Mining will create a lot of columns - feature space increase expotentially, large matrix
# Dimesionality and sparsity problem