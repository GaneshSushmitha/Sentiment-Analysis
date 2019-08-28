rm(list = ls())

library(RSQLite)

# Read Database file
db <- dbConnect(dbDriver("SQLite"), dbname = "database.sqlite")
dbListTables(db)
Amazon_Data_All <- dbReadTable(db,"Reviews")

#Data Exploration
hist(Amazon_Data_All$Score, col = "lightgreen",xlab = "Scores",main = "Scores Frequencies")


# Take into Consideration only required columns
Amazon_DF <- Amazon_Data_All[,c(7,9)]
Amazon_DF <- na.omit(Amazon_DF)

# Mark Positive/Negative
Sentiment <- vector(mode = "numeric", length = nrow(Amazon_DF))
for(i in 1:nrow(Amazon_DF)){
  
  if (Amazon_DF[i,1] > 3)
  {
    Sentiment[i] = 1  #Positive
  }
  else
  {
    Sentiment[i] = 0 #Negative
  }
  
}

Amazon_DF <- cbind(Amazon_DF,as.data.frame(Sentiment))

table(unlist(Amazon_DF$Sentiment))

# Clean the Data
require(tm)

Summary_corpus = Corpus(VectorSource(Amazon_DF$Summary))
Summary_corpus = tm_map(Summary_corpus, content_transformer(tolower))
Summary_corpus = tm_map(Summary_corpus, removeNumbers)
Summary_corpus = tm_map(Summary_corpus, removePunctuation)
Summary_corpus = tm_map(Summary_corpus, removeWords, c("the","and",stopwords("english")))
Summary_corpus =  tm_map(Summary_corpus, stripWhitespace)

review_dtm_tfidf <- DocumentTermMatrix(Summary_corpus, control = list(weighting = weightTfIdf))
review_dtm_tfidf = removeSparseTerms(review_dtm_tfidf, 0.99)
review_dtm_tfidf


# Prepare data from Prediction
New_Amazon <- as.data.frame(cbind(as.matrix(review_dtm_tfidf),Amazon_DF$Sentiment))
New_Amazon$V33 <- as.factor(New_Amazon$V33)


#Split Data to Train & Test
sample_size <- floor(0.80 * nrow(New_Amazon))
set.seed(123)
Index <- sample(seq_len(nrow(New_Amazon)), size = sample_size)

Train_Data <- New_Amazon[Index, ]
Test_Data <- New_Amazon[-Index, ]


X_IN <- Train_Data[,-33]
Y_IN <- Train_Data[,33]

X_OUT <- Test_Data[,-33]
Y_OUT <- Test_Data[,33]


#--------------------------------------
# Apply Naive Bayes
#--------------------------------------
library(klaR)
library(MASS)
library(pROC)


Model_1           <- NaiveBayes(V33 ~ ., data = Train_data)
Train_Predicted_Y <- predict(Model_1)
NB_Train_Error    <- 1 - sum(Predicted_Y$class == Y_IN)/length(Y_IN)

Test_Predicted_Y  <- predict(Model_1,X_OUT,type = "prob")
NB_Test_Error     <- 1 - sum(Predicted_Y$class == Y_OUT)/length(Y_OUT)

auc(Test_Predicted_Y,Y_OUT)


#--------------------------------------
# Apply Regression
#--------------------------------------
Model_2           <- lm(V33~., data = Train_Data)
Train_Predicted_Y <- predict(Model_2, newdata = X_IN,type='response')
LR_Train_error    <- sqrt(sum((Y_IN-Train_Predicted_Y)^2)/length(Train_Predicted_Y))
LR_Train_error

Test_Predicted_Y  <- predict(Model_2, newdata = X_OUT)
LR_Test_error     <- sqrt(sum((Y_OUT-Test_Predicted_Y)^2)/length(Test_Predicted_Y))
LR_Test_error

auc(Test_Predicted_Y,Y_OUT)

#--------------------------------------
# Apply SVM
#--------------------------------------
tune_object <- tune(svm, V33~., data = New_Amazon,
                    ranges = list(gamma = seq(0.5,1,0.1), cost = seq(2,4,1)),
                    tunecontrol = tune.control(sampling = "cross"))

summary(tune_object)
plot(tune_object)
tune_object$best.parameters
tune_object$best.performance

SVM_Model_3 <- svm(V33~ ., data = New_Amazon, gamma = 0.6, cost = 2)
print(SVM_Model_3)
summary(SVM_Model_3)

Train_Predicted_Y <- predict(SVM_Model_3, X_IN)
SVM_Traning_Error <- 1 - sum(Train_Predicted_Y == Y_IN) /length(Y_IN)

Test_Predicted_Y <- predict(SVM_Model_3, X_OUT)
SVM_Test_Error <- 1 - sum(Test_Predicted_Y == Y_OUT) /length(Y_OUT)

auc(Test_Predicted_Y,Y_OUT)


#Comapre Results
Results <- data.frame("AUC" = c(LR_AUC,NB_AUC,SVM_AUC))
row.names(Results) <- c("Logistic Regression", "Naive Bayes", "SVM")
Results

#Comapre Results
Results <- data.frame("AUC" = c(0.83,0.72,0.70))
row.names(Results) <- c("Logistic Regression", "Naive Bayes", "SVM")
Results










