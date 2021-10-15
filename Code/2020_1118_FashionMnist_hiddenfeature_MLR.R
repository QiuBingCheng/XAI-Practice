#########################¡@regression model ##############################
library(tensorflow)
library(keras)
library(caret)
library(data.table)
library(dplyr)
library(ggplot2)
fashion_mnist <- dataset_fashion_mnist()
fashion_mnist$train$x
x_train <- fashion_mnist$train$x/255 #60000    28    28
x_test <- fashion_mnist$test$x/255 #10000    28    28
y_train <- fashion_mnist$train$y
y_test <- fashion_mnist$test$y
x_train[1,,]
round(x_train[1,,],digits = 3)
xtrain <- array_reshape(x_train, dim=c(dim(x_train)[1],dim(x_train)[2],dim(x_train)[3],1)) #60000    28    28     1
xtest <- array_reshape(x_test, dim=c(dim(x_test)[1],dim(x_test)[2],dim(x_test)[3],1))#10000    28    28     1
## input data (xtrain¡Bxtest¡BtestY¡BtrainY)
xtrain <- readRDS("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/xtrain")
xtest <- readRDS("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/xtest")
testY <- readRDS("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/testY")
trainY <- readRDS("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/trainY")
cae_model <- load_model_hdf5("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/visualization/classification CAE/nolossweight/cae4.h5")
ae_model <- load_model_hdf5("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/visualization/normal CAE/nolossweight/ae4.h5")
cnn_model <- load_model_hdf5("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/visualization/classification CAE/nolossweight/cnn_model.h5")


## latent feature -----------------------------------------
layer_name<-"max_pool4"
encoder <- keras_model(inputs=cae_model$input,outputs=get_layer(cae_model,layer_name)$output)
summary(encoder)
encoded_testimgs = encoder %>% predict(xtest)#  10000     1     1    16
encoded_trainimgs = encoder %>% predict(xtrain)  #60000     1     1    16
train_matrix <- array_reshape(encoded_trainimgs, c(nrow(encoded_trainimgs),16), order = "F") #10000    16
test_matrix <- array_reshape(encoded_testimgs, c(nrow(encoded_testimgs),16), order = "F") #10000    16
train_latent <- cbind(train_matrix,y_train)
test_latent<- cbind(test_matrix,y_test)
train_latent <- as.data.frame(train_latent)
test_latent <- as.data.frame(test_latent)
head(train_latent)


################# multinominal logistic regression #######################################

############# our model 
library(nnet)
multinom.fit <- multinom(train_latent$y_train~.,data=train_latent) 
#final  value 8499.511431 
#final  value 36400.549097 
#final  value 8665.723101 
summary(multinom.fit) 
## extracting coefficients from the model and exponentiate
exp(coef(multinom.fit))
probability.table <- fitted(multinom.fit)
probability.table<- round(probability.table,digits = 3)
head(probability.table)


###### train ########
predict <- predict(multinom.fit, newdata = train_latent)
total_train <- cbind(train_latent,predict)
## confusion matrix 
ctable <- table(total_train$y_train,total_train$predict)
## accuracy
round((sum(diag(ctable))/sum(ctable))*100,2) #95.21 (our model)¡B79 (normal ae)¡B95.17 (cnn)

###### test ########
# Predicting 
predict <- predict(multinom.fit, newdata = test_latent)

total_test <- cbind(test_latent,predict)
## confusion matrix 
ctable <- table(total_test$y_test,total_test$predict)
## accuracy
round((sum(diag(ctable))/sum(ctable))*100,2) #91.02 (our model)¡B78.24(normal ae)¡B91.04(cnn)

## confusion matrix method 2
library(ramify)

categories <- c("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                "Sneaker", "Bag", "Boot")
pred <- factor(predict, labels=categories)
actual <- factor (y_test, labels = categories)
confusion<- caret::confusionMatrix(as.factor(pred), as.factor(actual ))
table <- data.frame(confusionMatrix(pred, actual)$table)
# quantity
ggplot(table, aes(x=Prediction, y=Reference, fill=Freq),width=200,height=200) +
  geom_tile() + theme_bw() + coord_equal() +
  scale_fill_distiller(palette="Blues", direction=1) +
  guides(fill=F, title.position = "top", title.hjust = 0.5) + # removing legend for `fill`
  labs(title = "Confusion matrix")+ # using a title instead
  geom_text(aes(label=Freq), color="black",face="bold")+  # printing values
  theme(plot.title = element_text(size=17,hjust = 0.5,face="bold"),
        legend.text = element_text(size=10,face="bold"),
        legend.title= element_text(size=10,face="bold"))

####################################### SVM ##############################################
library(e1071)  
#svm_model <- svm(train_latent$y_train~.,data=train_latent)
model <- svm(train_latent$y_train~.,type="C-classification",data=train_latent)

summary(model)

# ¹w´ú
train.pred = predict(model, train_latent)
test.pred = predict(model, test_latent)

###### train ########
predict <- predict(model, newdata = train_latent)
svm_trainmatrix<- cbind(train_latent,predict)
## confusion matrix 
ctable <- table(svm_trainmatrix$y_train, svm_trainmatrix$predict)
## accuracy
round((sum(diag(ctable))/sum(ctable))*100,2) #96 (our model)¡B 85.73(normal ae)¡B 95.81(cnn)

###### test ########
# Predicting 
predict <- predict(model, newdata = test_latent)

svm_testmatrix <- cbind(test_latent,predict)
## confusion matrix 
ctable <- table(svm_testmatrix$y_test,svm_testmatrix$predict)
## accuracy
round((sum(diag(ctable))/sum(ctable))*100,2) #91.43 (our model)¡B84.36(normal ae)¡B91.35(cnn)


## confusion matrix method 2
library(ramify)

categories <- c("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                "Sneaker", "Bag", "Boot")
pred <- factor(predict, labels=categories)
actual <- factor (y_test, labels = categories)
confusion<- caret::confusionMatrix(as.factor(pred), as.factor(actual ))
table <- data.frame(confusionMatrix(pred, actual)$table)
# quantity
ggplot(table, aes(x=Prediction, y=Reference, fill=Freq),width=200,height=200) +
  geom_tile() + theme_bw() + coord_equal() +
  scale_fill_distiller(palette="Blues", direction=1) +
  guides(fill=F, title.position = "top", title.hjust = 0.5) + # removing legend for `fill`
  labs(title = "Confusion matrix")+ # using a title instead
  geom_text(aes(label=Freq), color="black",face="bold")+  # printing values
  theme(plot.title = element_text(size=17,hjust = 0.5,face="bold"),
        legend.text = element_text(size=10,face="bold"),
        legend.title= element_text(size=10,face="bold"))







