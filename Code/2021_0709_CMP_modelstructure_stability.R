##################################################################################################
############################### flatten 192 feature###############################################
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
## Origin 
xtrain <- readRDS("D:/AmberChu/Handover/Data/CMP/Preprocess/final_set/A123_xtrain.rds")
xtest <- readRDS("D:/AmberChu/Handover/Data/CMP/Preprocess/final_set/A123_xtest.rds")
trainy <- readRDS("D:/AmberChu/Handover/Data/CMP/Preprocess/final_set/A123_trainy.rds")
testy <- readRDS("D:/AmberChu/Handover/Data/CMP/Preprocess/final_set/A123_testy.rds")

#### normal CAE(192) -----------------------------------------------------------
## encoderA123
enc_input = layer_input(shape = c(316, 19, 1),name="input")
enc_output = enc_input %>%
  layer_conv_2d(64,kernel_size=c(3,3), padding="same",name="encoder1") %>%
  layer_activation_leaky_relu(name="leak1")%>%
  layer_average_pooling_2d(c(3,3), padding="same",name="max_pool1")%>%
  layer_conv_2d(32,kernel_size = c(3,3),padding="same",name="encoder2")%>%
  layer_activation_leaky_relu(name="leak2")%>%
  layer_average_pooling_2d(c(3,3),padding="same",name="max_pool2")%>%
  layer_conv_2d(16,kernel_size = c(3,3),padding="same",name="encoder3")%>%
  layer_activation_leaky_relu(name="leak3")%>%
  layer_average_pooling_2d(c(3,3),padding="same",name="max_pool3")%>%
  layer_flatten(name="flatten")
encoder <- keras_model(enc_input,enc_output)
summary(encoder)
## encoder AB456
# enc_input = layer_input(shape = c(316, 19, 1),name="input")
# enc_output = enc_input %>%
#   layer_conv_2d(64,kernel_size=c(3,3), padding="same",name="encoder1") %>%
#   layer_activation_leaky_relu(name="leak1")%>%
#   layer_max_pooling_2d(c(3,3), padding="same",name="max_pool1")%>%
#   layer_conv_2d(32,kernel_size = c(3,3),padding="same",name="encoder2")%>%
#   layer_activation_leaky_relu(name="leak2")%>%
#   layer_max_pooling_2d(c(3,3),padding="same",name="max_pool2")%>%
#   layer_conv_2d(16,kernel_size = c(3,3),padding="same",name="encoder3")%>%
#   layer_activation_leaky_relu(name="leak3")%>%
#   layer_max_pooling_2d(c(3,3),padding="same",name="max_pool3")%>%
#   layer_flatten()
# encoder <- keras_model(enc_input,enc_output)
# summary(encoder)

## decoder 
decoder <- encoder$output %>%
  layer_reshape(c(12,1,16),name="reshape")%>%
  layer_conv_2d(16, kernel_size=c(3,3), padding="same",name="decoder1") %>% 
  layer_activation_leaky_relu(name="leak4")%>%
  layer_upsampling_2d(c(3,3),name="up_samp1")%>%
  layer_conv_2d(32, kernel_size=c(3,3), padding="same",name="decoder2") %>% 
  layer_activation_leaky_relu(name="leak5")%>%
  layer_upsampling_2d(c(3,3),name="up_samp2")%>%
  layer_conv_2d(64, kernel_size=c(3,3), padding="valid",name="decoder3") %>% 
  layer_activation_leaky_relu(name="leak6")%>%
  layer_upsampling_2d(c(3,3),name="up_samp3")%>%
  layer_conv_2d(1, kernel_size=c(3,3), activation="sigmoid",padding="valid",name="autoencoder")
autoencoder <- keras_model(encoder$input,decoder)
summary(autoencoder)


callbacks = list(
  callback_model_checkpoint("checkpoints.h5"), callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1)) ### 再調整！！

autoencoder%>% compile(optimizer="RMSprop", loss="mse")
history <- autoencoder %>% fit(x= xtrain, y= xtrain,validation_split=0.1,batch_size=10,epochs=200,callback=callbacks)


## predict_loss &　reconstruct_loss
# history_df <- as.data.frame(history)
# train_loss <-data.frame(t(history_df %>%
#                             filter(metric=="loss" & data=="training" & epoch==200)%>%
#                             select(value)))
# colnames(train_loss)<-"autoencoder_loss"
# 
# test_loss <-data.frame(t(history_df %>%
#                            filter(metric=="loss" & data=="validation" & epoch==200)%>%
#                            select(value)))
# colnames(test_loss)<-c("autoencoder_loss")  
# 
# saveRDS(train_loss,"C:/Users/User/Desktop/2021_0605/model/Results_loss/B456_trainloss_normalCAE.rds")
# saveRDS(test_loss,"C:/Users/User/Desktop/2021_0605/model/Results_loss/B456_testloss_normalCAE.rds")
# save_model_hdf5(autoencoder,"C:/Users/User/Desktop/2021_0605/model/B456_normalCAE.h5")


#stability 
test_history <- read_rds("C:/Users/User/Desktop/2021_0605/model/Results_loss/A456_testloss_normalCAE.rds")
test_history <- data.frame(t(test_history))


test_history  <- autoencoder %>% evaluate(x=xtest,y=xtest)
test_history <- data.frame(test_history)

saveRDS(test_history,"C:/Users/User/Desktop/2021_0709/CMP_data/Stability/A456_testloss_normalCAE1.rds")
save_model_hdf5(autoencoder,"C:/Users/User/Desktop/2021_0709/CMP_data/Stability/A456_normalCAE_1.h5")



###### CNN(192)-------------------------------------------------------------------------
## encoder 
##A123:
enc_input = layer_input(shape = c(316, 19, 1),name="input")
enc_output = enc_input %>%
  layer_conv_2d(64,kernel_size=c(3,3), padding="same",name="encoder1") %>%
  layer_activation_leaky_relu(name="leak1")%>%
  layer_average_pooling_2d(c(3,3), padding="same",name="max_pool1")%>%
  layer_conv_2d(32,kernel_size = c(3,3),padding="same",name="encoder2")%>%
  layer_activation_leaky_relu(name="leak2")%>%
  layer_average_pooling_2d(c(3,3),padding="same",name="max_pool2")%>%
  layer_conv_2d(16,kernel_size = c(3,3),padding="same",name="encoder3")%>%
  layer_activation_leaky_relu(name="leak3")%>%
  layer_average_pooling_2d(c(3,3),padding="same",name="max_pool3")%>%
  layer_flatten(name="flatten")
encoder <- keras_model(enc_input,enc_output)
summary(encoder)
##AB456:
# enc_input = layer_input(shape = c(316, 19, 1),name="input")
# enc_output = enc_input %>%
#   layer_conv_2d(64,kernel_size=c(3,3), padding="same",name="encoder1") %>%
#   layer_activation_leaky_relu(name="leak1")%>%
#   layer_max_pooling_2d(c(3,3), padding="same",name="max_pool1")%>%
#   layer_conv_2d(32,kernel_size = c(3,3),padding="same",name="encoder2")%>%
#   layer_activation_leaky_relu(name="leak2")%>%
#   layer_max_pooling_2d(c(3,3),padding="same",name="max_pool2")%>%
#   layer_conv_2d(16,kernel_size = c(3,3),padding="same",name="encoder3")%>%
#   layer_activation_leaky_relu(name="leak3")%>%
#   layer_max_pooling_2d(c(3,3),padding="same",name="max_pool3")%>%
#   layer_flatten(name="flatten")
# encoder <- keras_model(enc_input,enc_output)
# summary(encoder)

## predictor 
pred <- encoder$output%>%
  layer_dense(units=96,name="dec_class1")%>%
  layer_activation_leaky_relu(name="leak4")%>%
  layer_dense(units=48,name="dec_class2")%>%
  layer_activation_leaky_relu(name="leak5")%>%
  layer_dense(units=24,name="dec_class3")%>%
  layer_activation_leaky_relu(name="leak6")%>%
  layer_dense(units=1,name="predict")
pred_model <- keras_model(encoder$input,pred)
summary(pred_model)


cnn_model <- keras_model(encoder$input,pred)
summary(cnn_model)

callbacks = list(
  callback_model_checkpoint("checkpoints.h5"), callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1))

cnn_model %>% compile(optimizer="RMSprop", loss="mse")
callbacks = list(
  callback_model_checkpoint("checkpoints.h5"), callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1)) ### 再調整！！

history <- cnn_model %>% fit(x= xtrain, y= trainy,
                             validation_split=0.1,batch_size=10,epochs=200,callback=callbacks)

## predict_loss &　reconstruct_loss
# history_df <- as.data.frame(history)
# train_loss <-data.frame(t(history_df %>%
#                             filter(metric=="loss" & data=="training" & epoch==200)%>%
#                             select(value)))
# colnames(train_loss)<-"predict_loss"
# 
# test_loss <-data.frame(t(history_df %>%
#                            filter(metric=="loss" & data=="validation" & epoch==200)%>%
#                            select(value)))
# colnames(test_loss)<-c("predict_loss")
# 
# 
# saveRDS(train_loss,"C:/Users/User/Desktop/2021_0605/model/Results_loss/B456_trainloss_CNN.rds")
# saveRDS(test_loss,"C:/Users/User/Desktop/2021_0605/model/Results_loss/B456_testloss_CNN.rds")
# save_model_hdf5(cnn_model,"C:/Users/User/Desktop/2021_0605/model/B456_CNN.h5")




### 0711 stability --------------------------------
# test_history <- read_rds("C:/Users/User/Desktop/2021_0605/model/Results_loss/A123_testloss_CNN.rds")
# test_history <- data.frame(t(test_history))

test_history  <- cnn_model %>% evaluate(x=xtest,y=testy)
test_history <- data.frame(test_history)
print(test_history)

saveRDS(test_history,"C:/Users/User/Desktop/2021_0709/CMP_data/Stability/B456_testloss_CNN2.rds")
save_model_hdf5(cnn_model,"C:/Users/User/Desktop/2021_0709/CMP_data/Stability/B456_CNN_2.h5")




###### CAE+ANN(192)-------------------------------------------------------------------------
## encoder
#encoder A123
# enc_input = layer_input(shape = c(316, 19, 1),name="input")
# enc_output = enc_input %>%
#   layer_conv_2d(64,kernel_size=c(3,3), padding="same",name="encoder1") %>%
#   layer_activation_leaky_relu(name="leak1")%>%
#   layer_average_pooling_2d(c(3,3), padding="same",name="max_pool1")%>%
#   layer_conv_2d(32,kernel_size = c(3,3),padding="same",name="encoder2")%>%
#   layer_activation_leaky_relu(name="leak2")%>%
#   layer_average_pooling_2d(c(3,3),padding="same",name="max_pool2")%>%
#   layer_conv_2d(16,kernel_size = c(3,3),padding="same",name="encoder3")%>%
#   layer_activation_leaky_relu(name="leak3")%>%
#   layer_average_pooling_2d(c(3,3),padding="same",name="max_pool3")%>%
#   layer_flatten(name="flatten")
# encoder <- keras_model(enc_input,enc_output)
# summary(encoder)

##encoder AB456
enc_input = layer_input(shape = c(316, 19, 1),name="input")
enc_output = enc_input %>%
  layer_conv_2d(64,kernel_size=c(3,3), padding="same",name="encoder1") %>%
  layer_activation_leaky_relu(name="leak1")%>%
  layer_max_pooling_2d(c(3,3), padding="same",name="max_pool1")%>%
  layer_conv_2d(32,kernel_size = c(3,3),padding="same",name="encoder2")%>%
  layer_activation_leaky_relu(name="leak2")%>%
  layer_max_pooling_2d(c(3,3),padding="same",name="max_pool2")%>%
  layer_conv_2d(16,kernel_size = c(3,3),padding="same",name="encoder3")%>%
  layer_activation_leaky_relu(name="leak3")%>%
  layer_max_pooling_2d(c(3,3),padding="same",name="max_pool3")%>%
  layer_flatten(name="flatten")
encoder <- keras_model(enc_input,enc_output)
summary(encoder)

## predictor 
pred <- encoder$output%>%
  layer_dense(units=96,name="dec_class1")%>%
  layer_activation_leaky_relu(name="leak4")%>%
  layer_dense(units=48,name="dec_class2")%>%
  layer_activation_leaky_relu(name="leak5")%>%
  layer_dense(units=24,name="dec_class3")%>%
  layer_activation_leaky_relu(name="leak6")%>%
  layer_dense(units=1,name="predict")
pred_model <- keras_model(encoder$input,pred)
summary(pred_model)

## decoder 
decoder <- encoder$output %>%
  layer_dense(units = 192,name="dec_fully")%>%
  layer_activation_leaky_relu(name="leak7")%>%
  layer_reshape(c(12,1,16),name="reshape")%>%
  layer_conv_2d(16, kernel_size=c(3,3), padding="same",name="decoder1") %>% 
  layer_activation_leaky_relu(name="leak8")%>%
  layer_upsampling_2d(c(3,3),name="up_samp1")%>%
  layer_conv_2d(32, kernel_size=c(3,3), padding="same",name="decoder2") %>% 
  layer_activation_leaky_relu(name="leak9")%>%
  layer_upsampling_2d(c(3,3),name="up_samp2")%>%
  layer_conv_2d(64, kernel_size=c(3,3), padding="valid",name="decoder3") %>% 
  layer_activation_leaky_relu(name="leak10")%>%
  layer_upsampling_2d(c(3,3),name="up_samp3")%>%
  layer_conv_2d(1, kernel_size=c(3,3), activation="sigmoid",padding="valid",name="autoencoder")
autoencoder <- keras_model(encoder$input,decoder)
summary(autoencoder)

## full model (model)
model <- keras_model(inputs=enc_input,outputs=c(pred,decoder))
summary(model)

callbacks = list(
  callback_model_checkpoint("checkpoints.h5"), callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1)) ### 再調整！！


model%>% compile(optimizer="RMSprop", loss=list(predict="mse",autoencoder="mse"))
history <- model %>% fit(x= xtrain, y= list(predict=trainy,autoencoder=xtrain),
                         validation_split=0.1,batch_size=10,epochs=200,callback=callbacks)



## predict_loss &　reconstruct_loss
# history_df <- as.data.frame(history)
# train_loss <-data.frame(t(history_df %>%
#                             filter(metric!="loss" & data=="training" & epoch==200)%>%
#                             select(value)))
# colnames(train_loss)<-c("predict_loss","autoencoder_loss")
# 
# test_loss <-data.frame(t(history_df %>%
#                            filter(metric!="loss" & data=="validation" & epoch==200)%>%
#                            select(value)))
# colnames(test_loss)<-c("predict_loss","autoencoder_loss")


# saveRDS(train_loss,"C:/Users/User/Desktop/2021_0605/model/Results_loss/A456_trainloss_CAE_ANN.rds")
# saveRDS(test_loss,"C:/Users/User/Desktop/2021_0605/model/Results_loss/A456_testloss_CAE_ANN.rds")
# save_model_hdf5(model,"C:/Users/User/Desktop/2021_0605/model/A456_CAE_ANN.h5")


### 0711 stability:
# test_history <- readRDS("C:/Users/User/Desktop/2021_0605/model/Results_loss/B456_testloss_CAE_ANN.rds")
# test_history <- data.frame(t(test_history))

test_history  <- model %>% evaluate(x=xtest,y=list("predict"=testy,"autoencoder"=xtest))
test_history <- data.frame(test_history)
print(test_history)



saveRDS(test_history,"C:/Users/User/Desktop/2021_0709/CMP_data/Stability/B456_testloss_CAE_ANN10.rds")
save_model_hdf5(model,"C:/Users/User/Desktop/2021_0709/CMP_data/Stability/B456_CAE_ANN_10.h5")


#########################
###### CAE+ SVR (192)-------------------------------------------------------------------------
#encoder A123
# enc_input = layer_input(shape = c(316, 19, 1),name="input")
# enc_output = enc_input %>%
#   layer_conv_2d(64,kernel_size=c(3,3), padding="same",name="encoder1") %>%
#   layer_activation_leaky_relu(name="leak1")%>%
#   layer_average_pooling_2d(c(3,3), padding="same",name="max_pool1")%>%
#   layer_conv_2d(32,kernel_size = c(3,3),padding="same",name="encoder2")%>%
#   layer_activation_leaky_relu(name="leak2")%>%
#   layer_average_pooling_2d(c(3,3),padding="same",name="max_pool2")%>%
#   layer_conv_2d(16,kernel_size = c(3,3),padding="same",name="encoder3")%>%
#   layer_activation_leaky_relu(name="leak3")%>%
#   layer_average_pooling_2d(c(3,3),padding="same",name="max_pool3")%>%
#   layer_flatten(name="flatten")
# encoder <- keras_model(enc_input,enc_output)
# summary(encoder)

# ## encoder A456/B456
enc_input = layer_input(shape = c(316, 19, 1),name="input")
enc_output = enc_input %>%
  layer_conv_2d(64,kernel_size=c(3,3), padding="same",name="encoder1") %>%
  layer_activation_leaky_relu(name="leak1")%>%
  layer_max_pooling_2d(c(3,3), padding="same",name="max_pool1")%>%
  layer_conv_2d(32,kernel_size = c(3,3),padding="same",name="encoder2")%>%
  layer_activation_leaky_relu(name="leak2")%>%
  layer_max_pooling_2d(c(3,3),padding="same",name="max_pool2")%>%
  layer_conv_2d(16,kernel_size = c(3,3),padding="same",name="encoder3")%>%
  layer_activation_leaky_relu(name="leak3")%>%
  layer_max_pooling_2d(c(3,3),padding="same",name="max_pool3")%>%
  layer_flatten(name="flatten")
encoder <- keras_model(enc_input,enc_output)
summary(encoder)
## classifier 
pred <- encoder$output%>%
  layer_dense(units=96,name="dec_pred")%>%
  layer_activation_leaky_relu(name="leak4")%>%
  layer_dense(units=1,name="predict")
pred_model <- keras_model(encoder$input,pred)
summary(pred_model)
## decoder 
decoder <- encoder$output %>%
  layer_dense(units = 192,name="dec_fully")%>%
  layer_activation_leaky_relu(name="leak5")%>%
  layer_reshape(c(12,1,16),name="reshape")%>%
  layer_conv_2d(16, kernel_size=c(3,3), padding="same",name="decoder1") %>% 
  layer_activation_leaky_relu(name="leak6")%>%
  layer_upsampling_2d(c(3,3),name="up_samp1")%>%
  layer_conv_2d(32, kernel_size=c(3,3), padding="same",name="decoder2") %>% 
  layer_activation_leaky_relu(name="leak7")%>%
  layer_upsampling_2d(c(3,3),name="up_samp2")%>%
  layer_conv_2d(64, kernel_size=c(3,3), padding="valid",name="decoder3") %>% 
  layer_activation_leaky_relu(name="leak8")%>%
  layer_upsampling_2d(c(3,3),name="up_samp3")%>%
  layer_conv_2d(1, kernel_size=c(3,3), activation="sigmoid",padding="valid",name="autoencoder")
autoencoder <- keras_model(encoder$input,decoder)
summary(autoencoder)

## full model (model)
model <- keras_model(inputs=enc_input,outputs=c(pred,decoder))
summary(model)

# Declare loss function
# = max(0, abs(target - predicted) + epsilon)
# 1/2 margin width parameter = epsilon
callbacks = list(
  callback_model_checkpoint("checkpoints.h5"), callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1))

### custom loss:
eplison <- tf$constant(0.5)



# Margin term in loss
svr_loss <- function(y_true,y_pred)
{
  
  tf$reduce_mean(tf$maximum(0.,tf$subtract(tf$abs(tf$subtract(y_pred,y_true)),eplison)))
  
}

attr(svr_loss, "py_function_name") <- "svr_loss"

callbacks = list(
  callback_model_checkpoint("checkpoints.h5"), callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1)) ### 再調整！！


model%>% compile(optimizer="RMSprop", loss=list(predict=svr_loss,autoencoder="mse"))
# history <- model %>% fit(x= xtrain, y= list(predict=trainy,autoencoder=xtrain),
#                          validation_data=list(x=xtest,y=list(predict=testy,autoencoder=xtest)),batch_size=10,epochs=200,callback=callbacks)
history <- model %>% fit(x= xtrain, y= list(predict=trainy,autoencoder=xtrain),
                         validation_split=0.1,batch_size=10,epochs=200,callback=callbacks)

################
# verify predict mse:
# layer_name<-"predict"
# encoder <- keras_model(inputs=model$input,outputs=get_layer(model,layer_name)$output)
# summary(encoder)
# train_pred = encoder%>% predict(xtrain) # 570  12
# test_pred  = encoder %>% predict(xtest) #   245  12
# dim(train_pred)
# pred1 <- data.frame(trainy,train_pred)
# colnames(pred1)<-c("actual","pred")
# pred1<- pred1%>%
#   mutate(SSE=(actual-pred)^2)
# MSE_train<- mean(pred1$SSE) #19.98398 
# 
# pred2 <- data.frame(testy,test_pred)
# colnames(pred2)<-c("actual","pred")
# pred2<- pred2%>%
#   mutate(SSE=(actual-pred)^2)
# MSE_test <- mean(pred2$SSE) #25.92496
# 
# ## predict_loss &　reconstruct_loss
# history_df <- as.data.frame(history)
# train_loss <-data.frame(t(history_df %>%
#                             filter(metric!="loss" & data=="training" & epoch==200)%>%
#                             select(value)))
# train_loss <- cbind(MSE_train,train_loss$X2,train_loss$X1)
# colnames(train_loss)<-c("predict_loss","autoencoder_loss","eplison_loss")
# 
# test_loss <-data.frame(t(history_df %>%
#                            filter(metric!="loss" & data=="validation" & epoch==200)%>%
#                            select(value)))
# 
# test_loss <- cbind(MSE_test,test_loss$X2,test_loss$X1)
# colnames(test_loss)<-c("predict_loss","autoencoder_loss","eplison_loss")

# saveRDS(train_loss,"C:/Users/User/Desktop/2021_0605/model/Results_loss/B456_trainloss_CAE_SVR.rds")
# saveRDS(test_loss,"C:/Users/User/Desktop/2021_0605/model/Results_loss/B456_testloss_CAE_SVR.rds")
# save_model_hdf5(model,"C:/Users/User/Desktop/2021_0605/model/B456_CAE_SVR.h5")


### 0711 stability:
# test_history <- readRDS("C:/Users/User/Desktop/2021_0605/model/Results_loss/B456_testloss_CAE_SVR.rds")
# test_history <- data.frame(t(test_history))


layer_name<-"predict"
encoder <- keras_model(inputs=model$input,outputs=get_layer(model,layer_name)$output)
summary(encoder)

test_pred  = encoder %>% predict(xtest) #   245  12


pred2 <- data.frame(testy,test_pred)
colnames(pred2)<-c("actual","pred")
pred2<- pred2%>%
  mutate(SSE=(actual-pred)^2)
MSE_test <- mean(pred2$SSE) #25.92496

test_history  <- model %>% evaluate(x=xtest,y=list("predict"=testy,"autoencoder"=xtest))
test_history <- data.frame(test_history)


test_history[2,]<-MSE_test
print(test_history)

saveRDS(test_history,"C:/Users/User/Desktop/2021_0709/CMP_data/Stability/B456_testloss_CAE_SVR10.rds")
save_model_hdf5(model,"C:/Users/User/Desktop/2021_0709/CMP_data/Stability/B456_CAE_SVR_10.h5")

#########################################################################################
#########################################################################################
######## 2021_0712 model stability ----------------------------
ANN <- list()
SVR <- list()
CAE <- list()
CNN <- list()
for(i in 1:10)
{
  if(i!=1)
  {
    
    tmp <-read_rds(file=paste0("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/Stability/A123_testloss_CAE_ANN",i,".rds"))
    ANN[[i]]<- data.frame(tmp[2:3,])
    tmp2 <-read_rds(file=paste0("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/Stability/A123_testloss_CAE_SVR",i,".rds"))
    SVR[[i]]<- data.frame(tmp2[2:3,])
  }
  else
  {
    ANN[[i]] <-read_rds(file=paste0("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/Stability/A123_testloss_CAE_ANN",i,".rds"))
    tmp3 <-read_rds(file=paste0("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/Stability/A123_testloss_CAE_SVR",i,".rds"))
    SVR[[i]]<- data.frame(tmp3[1:2,])
  }
  
  
  CAE[[i]] <-read_rds(file=paste0("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/Stability/A123_testloss_normalCAE",i,".rds"))
  CNN[[i]] <-read_rds(file=paste0("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/Stability/A123_testloss_CNN",i,".rds"))
  
}

ann <- list()
svr <- list()
cae <- list()
cnn <- list()

for(i in 1:10)
{
  tmp <- t(data.table(ANN[[i]]))
  ann <- rbind(ann,tmp)
  tmp2 <- t(data.table(SVR[[i]]))
  svr <- rbind(svr,tmp2)
  tmp3 <- t(data.table(CAE[[i]]))
  cae <- rbind(cae,tmp3)
  tmp4 <- t(data.table(CNN[[i]]))
  cnn <- rbind(cnn,tmp4)
}

pred <- cbind(unlist(ann[,1]),unlist(svr[,1]),unlist(cnn[,1]))
rec <- cbind(unlist(ann[,2]),unlist(svr[,2]),unlist(cae[,1]))
colnames(pred)<-c("CAE_ANN","CAE_SVR","CNN")
colnames(rec)<-c("CAE_ANN","CAE_SVR","CAE")
pred_set  <-cbind(data.frame(colMeans(pred)),data.frame(apply(pred,2,var)))
rec_set <- cbind(data.frame(colMeans(rec)),data.frame(apply(rec,2,var)))
colnames(pred_set)<-c("avg_predMSE","var_predMSE")
colnames(rec_set)<-c("avg_recMSE","var_recMSE")

write.csv(pred_set,"C:/Users/User/Desktop/2021_0709/CMP_data/Stability/B456pred_set.csv")
write.csv(rec_set,"C:/Users/User/Desktop/2021_0709/CMP_data/Stability/B456rec_set.csv")
write.csv(pred,"C:/Users/User/Desktop/2021_0709/CMP_data/Stability/B456exp_pred.csv")
write.csv(rec,"C:/Users/User/Desktop/2021_0709/CMP_data/Stability/B456exp_rec.csv")





