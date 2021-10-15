#### ¶×¤JFashion MNIST  #######################################
library(tensorflow)
library(keras)
library(caret)
library(data.table)
library(dplyr)
library(ggplot2)
library(readr)
library(fs)
library(abind)
library(tidyverse)
library(magrittr)
library(factoextra)
library(ggpubr)
library(ggforce)
library(amap)
library(e1071)
fashion_mnist <- dataset_fashion_mnist()
fashion_mnist$train$x
x_train <- fashion_mnist$train$x/255 #60000    28    28
x_test <- fashion_mnist$test$x/255 #10000    28    28
y_train <- fashion_mnist$train$y
y_test <- fashion_mnist$test$y

xtrain <- array_reshape(x_train, dim=c(dim(x_train)[1],dim(x_train)[2],dim(x_train)[3],1)) #60000    28    28     1
xtest <- array_reshape(x_test, dim=c(dim(x_test)[1],dim(x_test)[2],dim(x_test)[3],1))#10000    28    28     1
trainy<- to_categorical(y_train)
testy<- to_categorical(y_test)

## input data (xtrain¡Bxtest¡BtestY¡BtrainY)
# saveRDS(xtrain,"C:/Users/User/Desktop/Amber/Fashion_mnist/data/xtrain.rds")
# saveRDS(xtest,"C:/Users/User/Desktop/Amber/Fashion_mnist/data/xtest.rds")
# saveRDS(trainy,"C:/Users/User/Desktop/Amber/Fashion_mnist/data/trainy.rds")
# saveRDS(testy,"C:/Users/User/Desktop/Amber/Fashion_mnist/data/testy.rds")


###############################################################################################################
###############################################################################################################
######################### model training ######################################################################
########################
library(tensorflow)
library(keras)
library(caret)
library(data.table)
library(dplyr)
library(ggplot2)
library(readr)
library(fs)
library(abind)
library(tidyverse)
library(magrittr)
library(factoextra)
library(ggpubr)
library(ggforce)
library(amap)
library(e1071)

xtrain <- readRDS("D:/AmberChu/Handover/Data/FashionMNIST/xtrain.rds")
xtest <- readRDS("D:/AmberChu/Handover/Data/FashionMNIST/xtest.rds")
trainy <- readRDS("D:/AmberChu/Handover/Data/FashionMNIST/trainy.rds")
testy <- readRDS("D:/AmberChu/Handover/Data/FashionMNIST/testy.rds")

#### CAE+ANN ------------------------------------------------------------------------------------
## encoder
enc_input = layer_input(shape = c(28, 28, 1),name="input")
enc_output = enc_input %>% 
  layer_conv_2d(64,kernel_size=c(3,3), activation="relu", padding="same",name="encoder1") %>% 
  layer_max_pooling_2d(c(2,2), padding="same",name="max_pool1") %>% 
  layer_conv_2d(32,kernel_size=c(3,3), activation="relu", padding="same",name="encoder2") %>% 
  layer_max_pooling_2d(c(2,2), padding="same",name="max_pool2")%>%
  layer_conv_2d(32,kernel_size=c(3,3), activation="relu", padding="same",name="encoder3") %>% 
  layer_max_pooling_2d(c(3,3), padding="same",name="max_pool3")%>%
  layer_conv_2d(16,kernel_size=c(3,3), activation="relu", padding="same",name="encoder4") %>% 
  layer_max_pooling_2d(c(3,3), padding="same",name="max_pool4")

encoder <- keras_model(enc_input,enc_output)
summary(encoder)

## classifier 
classify <- encoder$output%>%
  layer_flatten()%>%
  layer_dense(units=16,activation="relu",name="dec_class2")%>%
  layer_dense(units=64,activation="relu",name="dec_class3")%>%
  layer_dense(units=16,activation="relu",name="dec_class4")%>%
  layer_dense(units=10,activation="softmax",name="classification")
classify_model <- keras_model(encoder$input,classify)
summary(classify_model)

## decoder 
decoder <- encoder$output %>%
  layer_conv_2d(16, kernel_size=c(3,3), activation="relu", padding="same",name="decoder1") %>% 
  layer_upsampling_2d(c(3,3),name="up_samp1") %>%
  layer_conv_2d(32, kernel_size=c(3,3), activation="relu", padding="same",name="decoder2") %>%
  layer_upsampling_2d(c(3,3),name="up_samp2") %>%
  layer_conv_2d(32, kernel_size=c(3,3), activation="relu", padding="valid",name="decoder3") %>% 
  layer_upsampling_2d(c(2,2),name="up_samp3") %>% 
  layer_conv_2d(64, kernel_size=c(3,3), activation="relu",padding="same",name="decoder4") %>% 
  layer_upsampling_2d(c(2,2),name="up_samp4") %>% 
  layer_conv_2d(1, kernel_size=c(3,3), activation="sigmoid",padding="same",name="autoencoder")
autoencoder <- keras_model(encoder$input,decoder)
summary(autoencoder)

## full model (model)
model <- keras_model(inputs=enc_input,outputs=c(classify,decoder))
summary(model)


model%>% compile(optimizer="RMSprop",
                 loss=list("classification"="categorical_crossentropy","autoencoder"="mse")
                 ,metric=list("classification"="accuracy"))
callbacks = list(
  callback_model_checkpoint("checkpoints.h5"),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1))


history <- model %>% fit(x= xtrain, y=list("classification"= trainy,"autoencoder"= xtrain),
                         validation_split=0.1,
                         batch_size = 128,epochs = 25,callback=callbacks)


test_history  <- model %>% evaluate(x=xtest,y=list("classification"=testy,"autoencoder"=xtest))
test_history <- data.frame(test_history)


# saveRDS(test_history,"C:/Users/User/Desktop/2021_0709/model/Stability/testloss_CAE_ANN_10.rds")
# save_model_hdf5(model,"C:/Users/User/Desktop/2021_0709/model/Stability/CAE_ANN10.h5")

#### CAE+SVM ------------------------------------------------------------------------------------
## encoder
enc_input = layer_input(shape = c(28, 28, 1),name="input")
enc_output = enc_input %>% 
  layer_conv_2d(64,kernel_size=c(3,3), activation="relu", padding="same",name="encoder1") %>% 
  layer_max_pooling_2d(c(2,2), padding="same",name="max_pool1") %>% 
  layer_conv_2d(32,kernel_size=c(3,3), activation="relu", padding="same",name="encoder2") %>% 
  layer_max_pooling_2d(c(2,2), padding="same",name="max_pool2")%>%
  layer_conv_2d(32,kernel_size=c(3,3), activation="relu", padding="same",name="encoder3") %>% 
  layer_max_pooling_2d(c(3,3), padding="same",name="max_pool3")%>%
  layer_conv_2d(16,kernel_size=c(3,3), activation="relu", padding="same",name="encoder4") %>% 
  layer_max_pooling_2d(c(3,3), padding="same",name="max_pool4")

encoder <- keras_model(enc_input,enc_output)
summary(encoder)

## classifier 
classify <- encoder$output%>%
  layer_flatten()%>%
  layer_dense(units=32,activation="relu",name="dec_class2")%>%
  layer_dense(units=10,activation="linear",name="classification",kernel_regularizer= regularizer_l2(0.01))
classify_model <- keras_model(encoder$input,classify)
summary(classify_model)

## decoder 
decoder <- encoder$output %>%
  layer_conv_2d(16, kernel_size=c(3,3), activation="relu", padding="same",name="decoder1") %>% 
  layer_upsampling_2d(c(3,3),name="up_samp1") %>%
  layer_conv_2d(32, kernel_size=c(3,3), activation="relu", padding="same",name="decoder2") %>%
  layer_upsampling_2d(c(3,3),name="up_samp2") %>%
  layer_conv_2d(32, kernel_size=c(3,3), activation="relu", padding="valid",name="decoder3") %>% 
  layer_upsampling_2d(c(2,2),name="up_samp3") %>% 
  layer_conv_2d(64, kernel_size=c(3,3), activation="relu",padding="same",name="decoder4") %>% 
  layer_upsampling_2d(c(2,2),name="up_samp4") %>% 
  layer_conv_2d(1, kernel_size=c(3,3), activation="sigmoid",padding="same",name="autoencoder")
autoencoder <- keras_model(encoder$input,decoder)
summary(autoencoder)

## full model (model)
model <- keras_model(inputs=enc_input,outputs=c(classify,decoder))
summary(model)



model %>% compile(optimizer="RMSprop",
                  loss=list("classification"="hinge","autoencoder"="mse")
                  ,metric=list("classification"="accuracy"))
callbacks = list(
  callback_model_checkpoint("checkpoints.h5"),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1))


history <- model %>% fit(x= xtrain, y=list("classification"= trainy,"autoencoder"= xtrain),
                         validation_split=0.1,
                         batch_size = 128,epochs = 25,callback=callbacks)

test_history  <- model %>% evaluate(x=xtest,y=list("classification"=testy,"autoencoder"=xtest))
test_history <- data.frame(test_history)
# saveRDS(test_history,"C:/Users/User/Desktop/2021_0709/model/Stability/testloss_CAE_SVM_10.rds")
# save_model_hdf5(model,"C:/Users/User/Desktop/2021_0709/model/Stability/CAE_SVM10.h5")


#### CAE+LR ------------------------------------------------------------------------------------
## encoder
enc_input = layer_input(shape = c(28, 28, 1),name="input")
enc_output = enc_input %>% 
  layer_conv_2d(64,kernel_size=c(3,3), activation="relu", padding="same",name="encoder1") %>% 
  layer_max_pooling_2d(c(2,2), padding="same",name="max_pool1") %>% 
  layer_conv_2d(32,kernel_size=c(3,3), activation="relu", padding="same",name="encoder2") %>% 
  layer_max_pooling_2d(c(2,2), padding="same",name="max_pool2")%>%
  layer_conv_2d(32,kernel_size=c(3,3), activation="relu", padding="same",name="encoder3") %>% 
  layer_max_pooling_2d(c(3,3), padding="same",name="max_pool3")%>%
  layer_conv_2d(16,kernel_size=c(3,3), activation="relu", padding="same",name="encoder4") %>% 
  layer_max_pooling_2d(c(3,3), padding="same",name="max_pool4")

encoder <- keras_model(enc_input,enc_output)
summary(encoder)

## classifier (logistic regression)
classify <- encoder$output%>%
  layer_flatten()%>%
  layer_dense(units=10,activation="softmax",name="classification")
classify_model <- keras_model(encoder$input,classify)
summary(classify_model)

## decoder 
decoder <- encoder$output %>%
  layer_conv_2d(16, kernel_size=c(3,3), activation="relu", padding="same",name="decoder1") %>% 
  layer_upsampling_2d(c(3,3),name="up_samp1") %>%
  layer_conv_2d(32, kernel_size=c(3,3), activation="relu", padding="same",name="decoder2") %>%
  layer_upsampling_2d(c(3,3),name="up_samp2") %>%
  layer_conv_2d(32, kernel_size=c(3,3), activation="relu", padding="valid",name="decoder3") %>% 
  layer_upsampling_2d(c(2,2),name="up_samp3") %>% 
  layer_conv_2d(64, kernel_size=c(3,3), activation="relu",padding="same",name="decoder4") %>% 
  layer_upsampling_2d(c(2,2),name="up_samp4") %>% 
  layer_conv_2d(1, kernel_size=c(3,3), activation="sigmoid",padding="same",name="autoencoder")
autoencoder <- keras_model(encoder$input,decoder)
summary(autoencoder)

## full model (model)
model <- keras_model(inputs=enc_input,outputs=c(classify,decoder))
summary(model)

model%>% compile(optimizer="RMSprop",
                 loss=list("classification"="categorical_crossentropy","autoencoder"="mse")
                 ,metric=list("classification"="accuracy"))
callbacks = list(
  callback_model_checkpoint("checkpoints.h5"),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1))



history <- model %>% fit(x= xtrain, y=list("classification"= trainy,"autoencoder"= xtrain),
                         validation_split=0.1,
                         batch_size = 128,epochs = 25,callback=callbacks)


test_history  <- model %>% evaluate(x=xtest,y=list("classification"=testy,"autoencoder"=xtest))
test_history <- data.frame(test_history)

save_model_hdf5(model,"C:/Users/User/Desktop/2021_0709/model/CAE_SVM.h5")
saveRDS(test_history,"C:/Users/User/Desktop/2021_0709/model/Stability/testloss_CAE_LR_10.rds")
save_model_hdf5(model,"C:/Users/User/Desktop/2021_0709/model/Stability/CAE_LR10.h5")



###  CNN ------------------------------------------------------------------------------------
## encoder
enc_input = layer_input(shape = c(28, 28, 1),name="input")
enc_output = enc_input %>% 
  layer_conv_2d(64,kernel_size=c(3,3), activation="relu", padding="same",name="encoder1") %>% 
  layer_max_pooling_2d(c(2,2), padding="same",name="max_pool1") %>% 
  layer_conv_2d(32,kernel_size=c(3,3), activation="relu", padding="same",name="encoder2") %>% 
  layer_max_pooling_2d(c(2,2), padding="same",name="max_pool2")%>%
  layer_conv_2d(32,kernel_size=c(3,3), activation="relu", padding="same",name="encoder3") %>% 
  layer_max_pooling_2d(c(3,3), padding="same",name="max_pool3")%>%
  layer_conv_2d(16,kernel_size=c(3,3), activation="relu", padding="same",name="encoder4") %>% 
  layer_max_pooling_2d(c(3,3), padding="same",name="max_pool4")

encoder <- keras_model(enc_input,enc_output)
summary(encoder)

## classifier 
cnn_model <- encoder$output%>%
  layer_flatten()%>%
  layer_dense(units=16,activation="relu",name="dec_class2")%>%
  layer_dense(units=64,activation="relu",name="dec_class3")%>%
  layer_dense(units=16,activation="relu",name="dec_class4")%>%
  layer_dense(units=10,activation="softmax",name="classification")
cnn_model <- keras_model(encoder$input,cnn_model)
summary(cnn_model)

cnn_model%>% compile(optimizer="RMSprop",loss="categorical_crossentropy",metric=list("classification"="accuracy"))
callbacks = list(
  callback_model_checkpoint("checkpoints.h5"),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1))


history <- cnn_model %>% fit(x= xtrain, y= trainy,validation_split=0.1,batch_size = 128,epochs =25,callback=callbacks)

test_history  <- cnn_model %>% evaluate(x=xtest,y=testy)
test_history <- data.frame(test_history)

saveRDS(test_history,"C:/Users/User/Desktop/2021_0709/model/Stability/testloss_CNN_10.rds")
save_model_hdf5(cnn_model,"C:/Users/User/Desktop/2021_0709/model/Stability/CNN10.h5")


###  normalCAE ------------------------------------------------------------
## encoder
enc_input = layer_input(shape = c(28, 28, 1),name="input")
enc_output = enc_input %>% 
  layer_conv_2d(64,kernel_size=c(3,3), activation="relu", padding="same",name="encoder1") %>% 
  layer_max_pooling_2d(c(2,2), padding="same",name="max_pool1") %>% 
  layer_conv_2d(32,kernel_size=c(3,3), activation="relu", padding="same",name="encoder2") %>% 
  layer_max_pooling_2d(c(2,2), padding="same",name="max_pool2") %>%
  layer_conv_2d(32,kernel_size=c(3,3), activation="relu", padding="same",name="encoder3") %>% 
  layer_max_pooling_2d(c(3,3), padding="same",name="max_pool3")%>%
  layer_conv_2d(16,kernel_size=c(3,3), activation="relu", padding="same",name="encoder4") %>% 
  layer_max_pooling_2d(c(3,3), padding="same",name="max_pool4")
encoder <- keras_model(enc_input,enc_output)
summary(encoder)
## decoder 
decoder <- encoder$output %>%
  layer_conv_2d(16, kernel_size=c(3,3), activation="relu", padding="same",name="decoder1") %>% 
  layer_upsampling_2d(c(3,3),name="up_samp1") %>%
  layer_conv_2d(32, kernel_size=c(3,3), activation="relu", padding="same",name="decoder2") %>%
  layer_upsampling_2d(c(3,3),name="up_samp2") %>%
  layer_conv_2d(32, kernel_size=c(3,3), activation="relu", padding="valid",name="decoder3") %>% 
  layer_upsampling_2d(c(2,2),name="up_samp3") %>% 
  layer_conv_2d(64, kernel_size=c(3,3), activation="relu",padding="same",name="decoder4") %>% 
  layer_upsampling_2d(c(2,2),name="up_samp4") %>% 
  layer_conv_2d(1, kernel_size=c(3,3), activation="sigmoid", padding="same",name="autoencoder")
ae_model <- keras_model(encoder$input,decoder)
summary(ae_model)

ae_model %>% compile(optimizer="RMSprop",loss="mse")
callbacks = list(
  callback_model_checkpoint("checkpoints.h5"),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1))

ae_model %>% fit(xtrain,xtrain,batch_size=128,epochs=25,validation_split=0.1,callback=callbacks)


test_history  <- ae_model %>% evaluate(x=xtest,y=xtest)
test_history <- data.frame(test_history)


save_model_hdf5(ae_model,"C:/Users/User/Desktop/2021_0709/model/Stability/CAE10.h5")


##encode test data
layer_name<-"max_pool4"
encoder <- keras_model(inputs=ae_model$input,outputs=get_layer(ae_model,layer_name)$output)
summary(encoder)
encoded_imgs = encoder %>% predict(xtest)
dim(encoded_imgs)#  10000     1     1    16
encoded_train = encoder %>% predict(xtrain)

# classify_model
model <- keras_model_sequential()
model%>%
  layer_flatten()%>%
  layer_dense(units=16,activation="relu",name="class1")%>%
  layer_dense(units=64,activation="relu",name="class2")%>%
  layer_dense(units=16,activation="relu",name="class3")%>%
  layer_dense(units=10,activation="softmax",name="classification")
model%>% compile(optimizer="RMSprop",loss="categorical_crossentropy",metric="accuracy")

model %>% fit(encoded_train,trainy,batch_size = 128,epochs = 25,validation_split=0.1)

test_acc  <- model %>% evaluate(x=encoded_imgs,y=testy)
test_acc <- data.frame(test_acc)
set <- rbind(test_history,test_acc[2,])
row.names(set)<-c("rec","acc")
saveRDS(set,"C:/Users/User/Desktop/2021_0709/model/Stability/testloss_CAE10.rds")

#########################################################################################
#########################################################################################
######## 2021_0712 model stability ----------------------------
library(tensorflow)
library(keras)
library(caret)
library(data.table)
library(dplyr)
library(ggplot2)
library(readr)
library(fs)
library(abind)
library(tidyverse)
library(magrittr)
library(factoextra)
library(ggpubr)
library(ggforce)
library(amap)
library(e1071)
ANN <- list()
LR <- list()
SVM <- list()
CAE <- list()
CNN <- list()
for(i in 1:10)
{
  
  
  tmp <-read_rds(file=paste0("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Stability/testloss_CAE_ANN_",i,".rds"))
  ANN[[i]]<- data.frame(tmp[3:4,])
  tmp2 <-read_rds(file=paste0("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Stability/testloss_CAE_SVM_",i,".rds"))
  SVM[[i]]<- data.frame(tmp2[3:4,])
  tmp3 <-read_rds(file=paste0("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Stability/testloss_CAE_LR_",i,".rds"))
  LR[[i]]<- data.frame(tmp3[3:4,])
  CAE[[i]] <-read_rds(file=paste0("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Stability/testloss_CAE",i,".rds"))
  CNN[[i]] <-read_rds(file=paste0("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Stability/testloss_CNN_",i,".rds"))
  
}

ann <- list()
svm <- list()
lr <- list()
cae <- list()
cnn <- list()

for(i in 1:10)
{
  tmp <- t(data.table(ANN[[i]]))
  ann <- rbind(ann,tmp)
  tmp2 <- t(data.table(SVM[[i]]))
  svm <- rbind(svm,tmp2)
  tmp3 <- t(data.table(CAE[[i]]))
  cae <- rbind(cae,tmp3)
  tmp4 <- t(data.table(CNN[[i]]))
  cnn <- rbind(cnn,tmp4)
  tmp5 <- t(data.table(LR[[i]]))
  lr <- rbind(lr,tmp5)
}

rec <- cbind(unlist(ann[,1]),unlist(lr[,1]),unlist(svm[,1]),unlist(cae[,1]))
acc <- cbind(unlist(ann[,2]),unlist(lr[,2]),unlist(svm[,2]),unlist(cae[,2]),unlist(cnn[,2]))
colnames(rec)<-c("CAE_ANN","CAE_LR","CAE_SVM","CAE")
colnames(acc)<-c("CAE_ANN","CAE_LR","CAE_SVM","CAE","CNN")
rec_set  <-cbind(data.frame(colMeans(rec)),data.frame(apply(rec,2,var)))
acc_set <- cbind(data.frame(colMeans(acc)),data.frame(apply(acc,2,var)))
colnames(acc_set)<-c("avg_accMSE","var_accMSE")
colnames(rec_set)<-c("avg_recMSE","var_recMSE")
# write.csv(acc_set,"C:/Users/User/Desktop/2021_0709/Fashionmnist/model/Stability/acc_set.csv")
# write.csv(rec_set,"C:/Users/User/Desktop/2021_0709/Fashionmnist/model/Stability/rec_set.csv")
# 
# write.csv(acc,"C:/Users/User/Desktop/2021_0709/Fashionmnist/model/Stability/exp_acc.csv")
# write.csv(rec,"C:/Users/User/Desktop/2021_0709/Fashionmnist/model/Stability/exp_rec.csv")