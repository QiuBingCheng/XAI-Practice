##################################################################################################
##################################################################################################
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
xtrain<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/A456_xtrain.rds")
xtest<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/A456_xtest.rds")
trainy<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/A456_trainy.rds")
testy<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/A456_testy.rds")
###### CAE+ SVR (12)-------------------------------------------------------------------------
## encoder
## A123:
# enc_input = layer_input(shape = c(316, 19, 1),name="input")
# enc_output = enc_input %>%
#   layer_conv_2d(64,kernel_size=c(3,3), padding="same",name="encoder1") %>%
#   layer_activation_leaky_relu(name="leaky1")%>%
#   layer_average_pooling_2d(c(3,3), padding="same",name="max_pool1")%>%
#   layer_conv_2d(32,kernel_size = c(3,3),padding="same",name="encoder2")%>%
#   layer_activation_leaky_relu(name="leaky2")%>%
#   layer_average_pooling_2d(c(3,3),padding="same",name="max_pool2")%>%
#   layer_conv_2d(16,kernel_size = c(3,3),padding="same",name="encoder3")%>%
#   layer_activation_leaky_relu(name="leaky3")%>%
#   layer_average_pooling_2d(c(3,3),padding="same",name="max_pool3")%>%
#   layer_flatten()%>%
#   layer_dense(units=96,name="enc_dense1")%>%
#   layer_activation_leaky_relu(name="leaky4")%>%
#   layer_dense(units=48,name="enc_dense2")%>%
#   layer_activation_leaky_relu(name="leaky5")%>%
#   layer_dense(units=24,name='enc_dense3')%>%
#   layer_activation_leaky_relu(name="leaky6")%>%
#   layer_dense(units=12,name="hidden")%>%
#   layer_activation_leaky_relu(name="leaky7")
# encoder <- keras_model(enc_input,enc_output)
# summary(encoder)

### AA456:
enc_input = layer_input(shape = c(316, 19, 1),name="input")
enc_output = enc_input %>%
  layer_conv_2d(64,kernel_size=c(3,3), padding="same",name="encoder1") %>%
  layer_activation_leaky_relu(name="leaky1")%>%
  layer_max_pooling_2d(c(3,3), padding="same",name="max_pool1")%>%
  layer_conv_2d(32,kernel_size = c(3,3),padding="same",name="encoder2")%>%
  layer_activation_leaky_relu(name="leaky2")%>%
  layer_max_pooling_2d(c(3,3),padding="same",name="max_pool2")%>%
  layer_conv_2d(16,kernel_size = c(3,3),padding="same",name="encoder3")%>%
  layer_activation_leaky_relu(name="leaky3")%>%
  layer_max_pooling_2d(c(3,3),padding="same",name="max_pool3")%>%
  layer_flatten()%>%
  layer_dense(units=96,name="enc_dense1")%>%
  layer_activation_leaky_relu(name="leaky4")%>%
  layer_dense(units=48,name="enc_dense2")%>%
  layer_activation_leaky_relu(name="leaky5")%>%
  layer_dense(units=24,name='enc_dense3')%>%
  layer_activation_leaky_relu(name="leaky6")%>%
  layer_dense(units=12,name="hidden")%>%
  layer_activation_leaky_relu(name="leaky7")
encoder <- keras_model(enc_input,enc_output)
summary(encoder)

## classifier 
pred <- encoder$output%>%
  # layer_dense(units=6,name="dec_pred")%>%
  layer_dense(units=48,name="dec_pred")%>%
  layer_activation_leaky_relu(name="leaky8")%>%
  layer_dense(units=1,name="predict")
pred_model <- keras_model(encoder$input,pred)
summary(pred_model)

## decoder 
decoder <- encoder$output %>%
  layer_dense(units=24,name="dec_dense1")%>%
  layer_activation_leaky_relu(name="leaky9")%>%
  layer_dense(units=48,name="dec_dense2")%>%
  layer_activation_leaky_relu(name="leaky10")%>%
  layer_dense(units=96 ,name="dec_dense3")%>%
  layer_activation_leaky_relu(name="leaky11")%>%
  layer_dense(units = 192,name="dec_fully")%>%
  layer_activation_leaky_relu(name="leaky12")%>%
  layer_reshape(c(12,1,16),name="reshape")%>%
  layer_conv_2d(16, kernel_size=c(3,3), padding="same",name="decoder1") %>% 
  layer_activation_leaky_relu(name="leaky13")%>%
  layer_upsampling_2d(c(3,3),name="up_samp1")%>%
  layer_conv_2d(32, kernel_size=c(3,3), padding="same",name="decoder2") %>% 
  layer_activation_leaky_relu(name="leaky14")%>%
  layer_upsampling_2d(c(3,3),name="up_samp2")%>%
  layer_conv_2d(64, kernel_size=c(3,3), padding="valid",name="decoder3") %>% 
  layer_activation_leaky_relu(name="leaky15")%>%
  layer_upsampling_2d(c(3,3),name="up_samp3")%>%
  layer_conv_2d(1, kernel_size=c(3,3), activation="sigmoid",padding="valid",name="autoencoder")
autoencoder <- keras_model(encoder$input,decoder)
summary(autoencoder)

################################################################################################
## full model (model)
model <- keras_model(inputs=enc_input,outputs=c(pred,decoder))
summary(model)


### custom loss:
eplison <- tf$constant(0.5)

# Margin term in loss
loss <- function(y_true,y_pred)
{
  
  tf$reduce_mean(tf$maximum(0.,tf$subtract(tf$abs(tf$subtract(y_pred,y_true)),eplison)))
}


callbacks = list(
  callback_model_checkpoint("checkpoints.h5"), callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1))


# model%>% compile(optimizer="RMSprop", loss=list(predict=loss,autoencoder="mse"),loss_weights=list(predict=0.6,autoencoder=0.4))
model%>% compile(optimizer="RMSprop", loss=list(predict=loss,autoencoder="mse"))
history <- model %>% fit(x= xtrain, y= list(predict=trainy,autoencoder=xtrain),
                         validation_data=list(x=xtest,y=list(predict=testy,autoencoder=xtest)),batch_size=10,epochs=200,callback=callbacks)


################
# verify predict mse:
layer_name<-"predict"
encoder <- keras_model(inputs=model$input,outputs=get_layer(model,layer_name)$output)
summary(encoder)
train_pred = encoder%>% predict(xtrain) # 570  12
test_pred  = encoder %>% predict(xtest) #   245  12
dim(train_pred)
pred1 <- data.frame(trainy,train_pred)
colnames(pred1)<-c("actual","pred")
pred1<- pred1%>%
  mutate(SSE=(actual-pred)^2)
MSE_train<- mean(pred1$SSE) #19.98398 

pred2 <- data.frame(testy,test_pred)
colnames(pred2)<-c("actual","pred")
pred2<- pred2%>%
  mutate(SSE=(actual-pred)^2)
MSE_test <- mean(pred2$SSE) #25.92496

## predict_loss &　reconstruct_loss
history_df <- as.data.frame(history)
train_loss <-data.frame(t(history_df %>%
                            filter(metric!="loss" & data=="training" & epoch==200)%>%
                            select(value)))
train_loss <- cbind(MSE_train,train_loss$X2,train_loss$X1)
colnames(train_loss)<-c("predict_loss","autoencoder_loss","eplison_loss")

test_loss <-data.frame(t(history_df %>%
                           filter(metric!="loss" & data=="validation" & epoch==200)%>%
                           select(value)))
test_loss <- cbind(MSE_test,test_loss$X2,test_loss$X1)
colnames(test_loss)<-c("predict_loss","autoencoder_loss","eplison_loss")

### 
saveRDS(train_loss,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Results_loss/A456_trainloss_CAE_SVR.rds")
saveRDS(test_loss,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Results_loss/A456_testloss_CAE_SVR.rds")
save_model_hdf5(model,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/A456_CAE_SVR.h5")



## feature extraction--------------------------------------------------------------------------
layer_name<-"leaky7"
encoder <- keras_model(inputs=model$input,outputs=get_layer(model,layer_name)$output)
summary(encoder)
test_feature  = encoder %>% predict(xtest) #   245  12
dim(test_feature)
saveRDS(test_feature,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/A456_testfeature_CAE_SVR.rds")

#### certain - baseline 0: -----------------------------------------------------
recon_list <- list() 
recon_feature <- list()
for(i in 1L:12L)
{
  tmp <- matrix(0:0, nrow = nrow(test_feature), ncol = ncol(test_feature))
  c <- test_feature[,i]
  tmp[,i]<-c
  # decoder model -----------------------------------------
  dec_input = layer_input(shape = 12)
  dec_dense1 = get_layer(model,name="dec_dense1")
  dec_leak1 = get_layer(model,name="leaky9")
  dec_dense2 = get_layer(model,name="dec_dense2")
  dec_leak2 = get_layer(model,name="leaky10")
  dec_dense3 = get_layer(model,name="dec_dense3")
  dec_leak3 = get_layer(model,name="leaky11")
  dec_dense4 = get_layer(model,name="dec_fully")
  dec_leak4 = get_layer(model,name="leaky12")
  dec_reshape<- get_layer(model,name="reshape")
  dec1<- get_layer(model,name="decoder1")
  dec_leak5 = get_layer(model,name="leaky13")
  up_samp1<- get_layer(model,name="up_samp1")
  dec2<- get_layer(model,name="decoder2")
  dec_leak6 = get_layer(model,name="leaky14")
  up_samp2<- get_layer(model,name="up_samp2")
  dec3<- get_layer(model,name="decoder3")
  dec_leak7 = get_layer(model,name="leaky15")
  up_samp3<- get_layer(model,name="up_samp3")
  dec4<- get_layer(model,name="autoencoder")
  decoder <- keras_model(dec_input,dec4(up_samp3(dec_leak7(dec3(up_samp2(dec_leak6(dec2(up_samp1(dec_leak5(dec1(dec_reshape(dec_leak4(dec_dense4(dec_leak3(dec_dense3(dec_leak2(dec_dense2(dec_leak1(dec_dense1(dec_input))))))))))))))))))))
  # summary(decoder)
  reconstruct = decoder %>% predict(tmp)# 1981   316   19    1
  dim(reconstruct) #1981   19   19    1
  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
  }
  recon_feature[[i]]<- recon_list
}
saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Reconstruction_path/A456_reconstruct_test_CAE_SVR.rds")

###　測試使用：
recon_list <- list()
recon_feature <- list()
for(i in 1L:12L)
{
  tmp <- matrix(0:0, nrow = nrow(test_feature), ncol = ncol(test_feature))
  
  # decoder model -----------------------------------------
  dec_input = layer_input(shape = 12)
  dec_dense1 = get_layer(model,name="dec_dense1")
  dec_leak1 = get_layer(model,name="leaky9")
  dec_dense2 = get_layer(model,name="dec_dense2")
  dec_leak2 = get_layer(model,name="leaky10")
  dec_dense3 = get_layer(model,name="dec_dense3")
  dec_leak3 = get_layer(model,name="leaky11")
  dec_dense4 = get_layer(model,name="dec_fully")
  dec_leak4 = get_layer(model,name="leaky12")
  dec_reshape<- get_layer(model,name="reshape")
  dec1<- get_layer(model,name="decoder1")
  dec_leak5 = get_layer(model,name="leaky13")
  up_samp1<- get_layer(model,name="up_samp1")
  dec2<- get_layer(model,name="decoder2")
  dec_leak6 = get_layer(model,name="leaky14")
  up_samp2<- get_layer(model,name="up_samp2")
  dec3<- get_layer(model,name="decoder3")
  dec_leak7 = get_layer(model,name="leaky15")
  up_samp3<- get_layer(model,name="up_samp3")
  dec4<- get_layer(model,name="autoencoder")
  decoder <- keras_model(dec_input,dec4(up_samp3(dec_leak7(dec3(up_samp2(dec_leak6(dec2(up_samp1(dec_leak5(dec1(dec_reshape(dec_leak4(dec_dense4(dec_leak3(dec_dense3(dec_leak2(dec_dense2(dec_leak1(dec_dense1(dec_input))))))))))))))))))))
  summary(decoder)
  reconstruct = decoder %>% predict(tmp)# 1981   316   19    1
  dim(reconstruct) #1981   19   19    1
  
  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
  }
  recon_feature[[i]]<- recon_list
}
saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Reconstruction_path/A456_reconstruct_test_CAE_SVR(verify_0).rds")

## certain 0 - baseline: -------------------------------------------------------------
recon_list <- list() 
recon_feature <- list()
for(i in 1L:12L)
{
  
  cal_set <- test_feature
  tmp <- matrix(0:0,nrow=nrow(cal_set),ncol=1)
  cal_set[,i]<-tmp
  
  # decoder model -----------------------------------------
  dec_input = layer_input(shape = 12)
  dec_dense1 = get_layer(model,name="dec_dense1")
  dec_leak1 = get_layer(model,name="leaky9")
  dec_dense2 = get_layer(model,name="dec_dense2")
  dec_leak2 = get_layer(model,name="leaky10")
  dec_dense3 = get_layer(model,name="dec_dense3")
  dec_leak3 = get_layer(model,name="leaky11")
  dec_dense4 = get_layer(model,name="dec_fully")
  dec_leak4 = get_layer(model,name="leaky12")
  dec_reshape<- get_layer(model,name="reshape")
  dec1<- get_layer(model,name="decoder1")
  dec_leak5 = get_layer(model,name="leaky13")
  up_samp1<- get_layer(model,name="up_samp1")
  dec2<- get_layer(model,name="decoder2")
  dec_leak6 = get_layer(model,name="leaky14")
  up_samp2<- get_layer(model,name="up_samp2")
  dec3<- get_layer(model,name="decoder3")
  dec_leak7 = get_layer(model,name="leaky15")
  up_samp3<- get_layer(model,name="up_samp3")
  dec4<- get_layer(model,name="autoencoder")
  decoder <- keras_model(dec_input,dec4(up_samp3(dec_leak7(dec3(up_samp2(dec_leak6(dec2(up_samp1(dec_leak5(dec1(dec_reshape(dec_leak4(dec_dense4(dec_leak3(dec_dense3(dec_leak2(dec_dense2(dec_leak1(dec_dense1(dec_input))))))))))))))))))))
  
  
  summary(decoder)
  reconstruct = decoder %>% predict(cal_set)# 1981   316   19    1
  dim(reconstruct) #1981   19   19    1
  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
  }
  recon_feature[[i]]<- recon_list
}

saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Reconstruction_path/A456_reconstruct_test_CAE_SVR.rds")

###　測試使用：
recon_list <- list()
recon_feature <- list()
for(i in 1L:12L)
{
  
  # decoder model -----------------------------------------
  dec_input = layer_input(shape = 12)
  dec_dense1 = get_layer(model,name="dec_dense1")
  dec_leak1 = get_layer(model,name="leaky9")
  dec_dense2 = get_layer(model,name="dec_dense2")
  dec_leak2 = get_layer(model,name="leaky10")
  dec_dense3 = get_layer(model,name="dec_dense3")
  dec_leak3 = get_layer(model,name="leaky11")
  dec_dense4 = get_layer(model,name="dec_fully")
  dec_leak4 = get_layer(model,name="leaky12")
  dec_reshape<- get_layer(model,name="reshape")
  dec1<- get_layer(model,name="decoder1")
  dec_leak5 = get_layer(model,name="leaky13")
  up_samp1<- get_layer(model,name="up_samp1")
  dec2<- get_layer(model,name="decoder2")
  dec_leak6 = get_layer(model,name="leaky14")
  up_samp2<- get_layer(model,name="up_samp2")
  dec3<- get_layer(model,name="decoder3")
  dec_leak7 = get_layer(model,name="leaky15")
  up_samp3<- get_layer(model,name="up_samp3")
  dec4<- get_layer(model,name="autoencoder")
  decoder <- keras_model(dec_input,dec4(up_samp3(dec_leak7(dec3(up_samp2(dec_leak6(dec2(up_samp1(dec_leak5(dec1(dec_reshape(dec_leak4(dec_dense4(dec_leak3(dec_dense3(dec_leak2(dec_dense2(dec_leak1(dec_dense1(dec_input))))))))))))))))))))
  summary(decoder)
  reconstruct = decoder %>% predict(test_feature)# 1981   316   19    1
  dim(reconstruct) #1981   19   19    1
  
  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
  }
  recon_feature[[i]]<- recon_list
}
saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Reconstruction_path/A456_reconstruct_test_CAE_SVR(verify_0).rds")


#### Actual_corr - Reconstruct_corr ---------------------------------------------------------------
library(tensorflow)
library(keras)
library(caret)
library(data.table)
library(dplyr)
library(ggplot2)
library(readr)
library(fs)
library(abind)r
library(tidyverse)
library(magrittr)
library(factoextra)
library(ggpubr)
library(ggforce)
library(amap)
library(e1071)
xtest<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/A456_xtest.rds")
xtest <- array_reshape(xtest,dim=c(dim(xtest)[1],dim(xtest)[2],dim(xtest)[3]*1)) 

# recon_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Reconstruction_path/A456_reconstruct_test_CAE_SVR.rds")
# recon_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Reconstruction_path/A456_reconstruct_test_CAE_SVR(verify_0).rds")
# recon_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Reconstruction_path/A456_reconstruct_test_CAE_SVR.rds")
recon_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Reconstruction_path/A456_reconstruct_test_CAE_SVR(verify_0).rds")

dim(xtest)
corr_error3<- list()
corr_errorSVID_sum2<-list()

for(i in 1L:12L)
{
  tt <- recon_feature[[i]]
  for(j in 1L:dim(xtest)[1])
  {
    tt2<-data.frame(tt[[j]])
    real<-data.frame(xtest[j,,])
    names(real) <- paste0('X', 1:(ncol(real)))
    revise<- real-tt2
    
    ### MSE-------------------------
    square <- data.frame(revise^2)
    sum2 <- colMeans(square)
    corr_error3[[j]]<-sum2
  }
  
  corr_errorSVID_sum2[[i]] <- corr_error3
  
}

origin_errorSVID_MSE<- corr_errorSVID_sum2
each_waferList <- list()
for(j in 1:dim(xtest)[1])
{
  null <- c()
  for(i in 1:12)
  {
    tmp <- origin_errorSVID_MSE[[i]][[j]]
    tmp <- data.table(tmp)
    null<-cbind(null,tmp)
    colnames(null) <- paste0('X', 1:(ncol(null)))
    
  }
  each_waferList[[j]]<-null
}

# saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Reconstruction_path/A456_Eachwafer_MSE_test_CAE_SVR.rds")
# saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Reconstruction_path/A456_Eachwafer_MSE_test_CAE_SVR(verify0).rds")
# saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Reconstruction_path/A456_Eachwafer_MSE_test_CAE_SVR.rds")
saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Reconstruction_path/A456_Eachwafer_MSE_test_CAE_SVR(verify0).rds")

each_waferList<- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Reconstruction_path/A456_Eachwafer_MSE_test_CAE_SVR.rds")
each_waferList2<- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Reconstruction_path/A456_Eachwafer_MSE_test_CAE_SVR(verify0).rds")

each_waferList<- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Reconstruction_path/A456_Eachwafer_MSE_test_CAE_SVR.rds")
each_waferList2<- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Reconstruction_path/A456_Eachwafer_MSE_test_CAE_SVR(verify0).rds")

tmp <- matrix(0:0, nrow = 19, ncol = 12)
tmp_list<- list()
tmp <-data.table(tmp)

for(j in 1:dim(xtest)[1])
{
  tmp_list[[j]] <- each_waferList[[j]]-each_waferList2[[j]]
  tmp<-tmp+tmp_list[[j]]
  
}
avg_wafer_error <- tmp/length(tmp_list)
# saveRDS(avg_wafer_error,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Reconstruction_path/A456_avg_MSE(actual-null)_test_CAE_SVR.rds")
saveRDS(avg_wafer_error,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Reconstruction_path/A456_avg_MSE(actual-null)_test_CAE_SVR.rds")

### visualize---------------------------------------------------------------------------------------
avg_wafer_error<- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Reconstruction_path/A456_avg_MSE(actual-null)_test_CAE_SVR.rds")
avg_wafer_error<- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Reconstruction_path/A456_avg_MSE(actual-null)_test_CAE_SVR.rds")

value<- round(avg_wafer_error,digits = 4)
SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
value<- cbind(SVID,data.table(value))
value<- melt(value)
colnames(value)<-c("SVID","feature","error")
library(ggplot2)

# Plot 
mid<- (min(value$error)+max(value$error))/2
plot <- ggplot(data = value, aes(x = feature, y = SVID)) + 
  geom_tile(aes(fill = error), color = "white", size = 1) +
  scale_fill_gradient2(
    low = 'red', mid = 'white', high = 'steelblue',
    midpoint = mid, guide = 'colourbar', aesthetics = 'fill',limit=c(min(value$error),max(value$error)))+
  xlab("hidden_feature") + 
  theme_grey(base_size = 10) + 
  # ggtitle("A456 CAE+SVR: avg_MSE of each SVID (certain_nonezero-baseline)") +
  ggtitle("A456 CAE+SVR: avg_MSE of each SVID (certain_zero-baseline)") +
  geom_text(aes(label=round(error,digits = 3)),size=3.5)+
  theme(axis.ticks = element_blank(), 
        panel.background = element_blank(), 
        plot.title = element_text(size=20,hjust = 0.5,face="bold"),legend.position="right",legend.direction="vertical",
        axis.text.x= element_text(size = 10,face="bold"),axis.text.y= element_text(size = 10,face="bold"),
        axis.title.x = element_text(size=15,face="bold"),axis.title.y = element_text(size=15,face="bold"))+
  guides(fill = guide_colorbar(barwidth = 1, barheight =7,title.position = "top", title.hjust = 0.5))

print(plot)
ggsave(plot, file="C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Reconstruction_path/origin/A456_CAESVR.png", width=15, height=10)
ggsave(plot, file="C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Reconstruction_path/origin/A456_CAESVR.png", width=15, height=10)

#### 2021/05/14 each SVID range ------------------
avg_wafer_error<- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Reconstruction_path/A456_avg_MSE(actual-null)_test_CAE_SVR.rds")
avg_wafer_error<- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Reconstruction_path/A456_avg_MSE(actual-null)_test_CAE_SVR.rds")

value<- round(avg_wafer_error,digits = 4)
SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
value<- cbind(SVID,data.table(value))
value<- melt(value)
colnames(value)<-c("SVID","feature","error")
library(ggplot2)

maxmin <- function(x) (x - min(x))/(max(x)-min(x))
Normalize <- t(apply(avg_wafer_error, 1, maxmin))
Normalize<- round(Normalize,digits = 4)
SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
value<- cbind(SVID,data.table(Normalize))
value<- melt(value)
colnames(value)<-c("SVID","feature","error")
library(ggplot2)

# Plot 
mid<-(min(value$error)+max(value$error))/2
plot <- ggplot(data = value, aes(x = feature, y = SVID)) + 
  geom_tile(aes(fill = error), color = "white", size = 1) +
  # scale_fill_gradient(low = "gray95", high = "tomato",limits=c(min(value$error), max(value$error)))+
  # scale_fill_continuous_divergingx(palette = 'RdBu', mid=mid,limits = c(min(value$error),max(value$error)))+
  scale_fill_gradient2(
    low = 'red', mid = 'white', high = 'steelblue',
    midpoint = mid, guide = 'colourbar', aesthetics = 'fill',limit=c(min(value$error),max(value$error)))+
  xlab("hidden_feature") + 
  theme_grey(base_size = 10) + 
  # ggtitle("A456 CAE+SVR: Min-max avg_MSE of each SVID (certain_nonezero-baseline)") +
  ggtitle("A456 CAE+SVR: Min-max avg_MSE of each SVID (certain_zero-baseline)") +
  geom_text(aes(label=round(error,digits = 3)),size=3.5)+
  theme(axis.ticks = element_blank(), 
        panel.background = element_blank(), 
        plot.title = element_text(size=20,hjust = 0.5,face="bold"),legend.position="right",legend.direction="vertical",
        axis.text.x= element_text(size = 10,face="bold"),axis.text.y= element_text(size = 10,face="bold"),
        axis.title.x = element_text(size=15,face="bold"),axis.title.y = element_text(size=15,face="bold"))+
  guides(fill = guide_colorbar(barwidth = 1, barheight =7,title.position = "top", title.hjust = 0.5))

print(plot)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Reconstruction_path/min-max_A456_CAESVR.png", width=15, height=10)
saveRDS(Normalize,"C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/A456_minmax_avg_MSE_test_CAE_SVR.rds")

ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Reconstruction_path/min-max_A456_CAESVR.png", width=15, height=10)
saveRDS(Normalize,"C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/A456_minmax_avg_MSE_test_CAE_SVR.rds")

#####################################################################################
######################## 預測解釋性 #################################################
### certain - baseline --------------------------------------------

pred_list <- list()
pred_feature <- list()
for(i in 1L:12L)
{
  tmp <- matrix(0:0, nrow = nrow(test_feature), ncol = ncol(test_feature))
  c <- test_feature[,i]
  tmp[,i]<-c
  
  ## CAE_SVR classifier model -------------------------------------
  pred_input = layer_input(shape=12)
  dense1<-get_layer(model,name="dec_pred")
  leak1 <-get_layer(model,name="leaky8")
  dense2<-get_layer(model,name="predict")
  predictor<- keras_model(pred_input,dense2(leak1(dense1(pred_input))))
  
  summary(predictor)
  prediction = predictor %>% predict(tmp)# 240   1
  
  for(j in 1: dim(prediction)[1])
  {
    pred_list[[j]]<- prediction[j,]
  }
  pred_feature[[i]]<- pred_list
}
saveRDS(pred_feature,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Prediction_path/A456_prediction_test_CAE_SVR.rds")

###　測試使用：
pred_list <- list()
pred_feature <- list()
for(i in 1L:12L)
{
  tmp <- matrix(0:0, nrow = nrow(test_feature), ncol = ncol(test_feature))
  
  
  ## CAE_SVR classifier model -------------------------------------
  pred_input = layer_input(shape=12)
  dense1<-get_layer(model,name="dec_pred")
  leak1 <-get_layer(model,name="leaky8")
  dense2<-get_layer(model,name="predict")
  predictor<- keras_model(pred_input,dense2(leak1(dense1(pred_input))))
  
  summary(predictor)
  prediction = predictor %>% predict(tmp)# 1981   316   19    1
  
  for(j in 1: dim(prediction)[1])
  {
    pred_list[[j]]<- prediction[j,]
  }
  pred_feature[[i]]<- pred_list
}
saveRDS(pred_feature,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Prediction_path/A456_prediction_test_CAE_SVR(verify_0).rds")

### certain0 - baseline --------------------------------------------
library(tensorflow)
library(keras)
library(caret)
library(data.table)
library(dplyr)
library(ggplot2)
library(readr)
library(fs)
library(abind)r
library(tidyverse)
library(magrittr)
library(factoextra)
library(ggpubr)
library(ggforce)
library(amap)
library(e1071)

pred_list <- list()
pred_feature <- list()
for(i in 1L:12L)
{
  cal_set <- test_feature
  tmp <- matrix(0:0,nrow=nrow(cal_set),ncol=1)
  cal_set[,i]<-tmp
  
  
  ## CAE_SVR classifier model -------------------------------------
  pred_input = layer_input(shape=12)
  dense1<-get_layer(model,name="dec_pred")
  leak1 <-get_layer(model,name="leaky8")
  dense2<-get_layer(model,name="predict")
  predictor<- keras_model(pred_input,dense2(leak1(dense1(pred_input))))
  
  summary(predictor)
  prediction = predictor %>% predict(cal_set)# 240   1
  
  for(j in 1: dim(prediction)[1])
  {
    pred_list[[j]]<- prediction[j,]
  }
  pred_feature[[i]]<- pred_list
}

saveRDS(pred_feature,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Prediction_path/A456_prediction_test_CAE_SVR.rds")

###　測試使用：
pred_list <- list()
pred_feature <- list()
for(i in 1L:12L)
{
  
  ## CAE_SVR classifier model -------------------------------------
  pred_input = layer_input(shape=12)
  dense1<-get_layer(model,name="dec_pred")
  leak1 <-get_layer(model,name="leaky8")
  dense2<-get_layer(model,name="predict")
  predictor<- keras_model(pred_input,dense2(leak1(dense1(pred_input))))
  
  summary(predictor)
  prediction = predictor %>% predict(test_feature)# 1981   316   19    1
  
  for(j in 1: dim(prediction)[1])
  {
    pred_list[[j]]<- prediction[j,]
  }
  pred_feature[[i]]<- pred_list
}
saveRDS(pred_feature,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Prediction_path/A456_prediction_test_CAE_SVR(verify_0).rds")



#### Actual_corr - Reconstruct_corr
ytest<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/A456_testy.rds")

# pred_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Prediction_path/A456_prediction_test_CAE_SVR.rds")
# pred_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Prediction_path/A456_prediction_test_CAE_SVR(verify_0).rds")
# pred_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Prediction_path/A456_prediction_test_CAE_SVR.rds")
pred_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Prediction_path/A456_prediction_test_CAE_SVR(verify_0).rds")

# corr_error3<- list()
corr_errorSVID_sum2<-list()
SSE_matrix<- c()
# SSE_matrix<-matrix(0:0, nrow = 240, ncol = 192)
for(i in 1L:12)
{
  tt <- unlist(pred_feature[[i]])
  tt2<-data.frame(tt)
  real<-data.frame(ytest)
  revise<- real-tt2
  ### MSE-------------------------
  square <- data.frame(revise^2)
  SSE_matrix<- cbind(SSE_matrix,data.table(square))
  sum2 <- colMeans(square)
  # corr_error3[[i]]<-sum2
  corr_errorSVID_sum2[[i]] <- sum2
}


# saveRDS(SSE_matrix,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Prediction_path/A456_errorSVID_prediction_SSEmatrix_test_CAE_SVR.rds")
# saveRDS(corr_errorSVID_sum2,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Prediction_path/A456_errorSVID_prediction_MSE_test_CAE_SVR.rds")

# saveRDS(SSE_matrix,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Prediction_path/A456_errorSVID_prediction_SSEmatrix_test_CAE_SVR(verify0).rds")
# saveRDS(corr_errorSVID_sum2,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Prediction_path/A456_errorSVID_prediction_MSE_test_CAE_SVR(verify0).rds")

# saveRDS(SSE_matrix,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Prediction_path/A456_errorSVID_prediction_SSEmatrix_test_CAE_SVR.rds")
# saveRDS(corr_errorSVID_sum2,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Prediction_path/A456_errorSVID_prediction_MSE_test_CAE_SVR.rds")

saveRDS(SSE_matrix,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Prediction_path/A456_errorSVID_prediction_SSEmatrix_test_CAE_SVR(verify0).rds")
saveRDS(corr_errorSVID_sum2,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Prediction_path/A456_errorSVID_prediction_MSE_test_CAE_SVR(verify0).rds")


# origin_errorSVID_MSE<- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Prediction_path/A456_errorSVID_prediction_MSE_test_CAE_SVR.rds")
# origin_errorSVID_MSE2<- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Prediction_path/A456_errorSVID_prediction_MSE_test_CAE_SVR(verify0).rds")
# origin_errorSVID_SSEmatrix<- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Prediction_path/A456_errorSVID_prediction_SSEmatrix_test_CAE_SVR.rds")
# origin_errorSVID_SSEmatrix2<- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Prediction_path/A456_errorSVID_prediction_SSEmatrix_test_CAE_SVR(verify0).rds")

origin_errorSVID_MSE<- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Prediction_path/A456_errorSVID_prediction_MSE_test_CAE_SVR.rds")
origin_errorSVID_MSE2<- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Prediction_path/A456_errorSVID_prediction_MSE_test_CAE_SVR(verify0).rds")
origin_errorSVID_SSEmatrix<- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Prediction_path/A456_errorSVID_prediction_SSEmatrix_test_CAE_SVR.rds")
origin_errorSVID_SSEmatrix2<- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Prediction_path/A456_errorSVID_prediction_SSEmatrix_test_CAE_SVR(verify0).rds")

certain_MSE <- data.frame(unlist(origin_errorSVID_MSE))
baseline_MSE <- data.frame(unlist(origin_errorSVID_MSE2))
pred_performance<- certain_MSE-baseline_MSE

SSE_performance<-origin_errorSVID_SSEmatrix-origin_errorSVID_SSEmatrix2
colnames(SSE_performance) <- paste0('X', 1:(ncol(SSE_performance)))

# saveRDS(SSE_performance,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Prediction_path/A456_errorSVID_prediction_SSEpreformance_test_CAE_SVR.rds")
# saveRDS(pred_performance,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Prediction_path/A456_errorSVID_prediction_MSEperformance_test_CAE_SVR.rds")

saveRDS(SSE_performance,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Prediction_path/A456_errorSVID_prediction_SSEpreformance_test_CAE_SVR.rds")
saveRDS(pred_performance,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Prediction_path/A456_errorSVID_prediction_MSEperformance_test_CAE_SVR.rds")


### visualize---------------------------------------------------------------------------------------
SSE_performance<- readRDS("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Prediction_path/A456_errorSVID_prediction_SSEpreformance_test_CAE_SVR.rds")
SSE_performance<- readRDS("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Prediction_path/A456_errorSVID_prediction_SSEpreformance_test_CAE_SVR.rds")

value <- data.table(SSE_performance)
value<- round(value,digits = 4)

mean2 <- data.table(colMeans(value))
y_value2<- factor(seq(1,dim(value)[2],by=1),levels =seq(1,dim(value)[2],by=1))
cal2 <- cbind(y_value2,mean2)
colnames(cal2)<-c("wafer","mean")


saveRDS(cal2,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/A456_avgSSE_test_CAE_SVR.rds")
saveRDS(cal2,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/A456_avgSSE_test_CAE_SVR.rds")



######################################################################
avgSSE<-read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/A456_avgSSE_test_CAE_SVR.rds")
avgSSE<-read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/A456_avgSSE_test_CAE_SVR.rds")


maxmin <- function(x) (x - min(x))/(max(x)-min(x))
avgSSE_Normalize <- apply(avgSSE[,2], 2, maxmin)
avgSSE_Normalize<- round(avgSSE_Normalize,digits = 3)

value<- cbind(data.table(factor(seq(1,dim(avgSSE_Normalize)[1],by=1))),avgSSE_Normalize)
colnames(value)<-c("number","mean")

plot <- ggplot(value, aes(x =  number, y = mean)) +
  geom_bar(stat = "identity",fill="lightblue",width = 0.5) +
  xlab("12 feature")+ylab("min max normalize of MSE")+
  ggtitle("MSE proportion of 12 hidden feature ") +
  theme_minimal()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 10,face="bold"), 
        axis.title = element_text(size = 15,face="bold"), 
        plot.title = element_text(size = 20, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=15,face="bold"),
        axis.title.y = element_text(size=15,face="bold"),
        axis.text.y = element_text(size=15),
        axis.text.x = element_text(size=15,face="bold"),
        legend.title = element_text(size=10),
        legend.text = element_text(size=10),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank()) 
print(plot)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Prediction_path/A456_certain_nonzero_minmax_weight.png", width=15, height=10)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Prediction_path/A456_certain_zero_minmax_weight.png", width=15, height=10)




########################## prediction +reconstruction heatmap ---------------------------

Normalize<-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/A456_minmax_avg_MSE_test_CAE_SVR.rds")
avgSSE <- readRDS("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/A456_avgSSE_test_CAE_SVR.rds")

Normalize<-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/A456_minmax_avg_MSE_test_CAE_SVR.rds")
avgSSE <- readRDS("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/A456_avgSSE_test_CAE_SVR.rds")


maxmin <- function(x) (x - min(x))/(max(x)-min(x))
avgSSE_Normalize <- apply(avgSSE[,2], 2, maxmin)
avgSSE_Normalize<- round(avgSSE_Normalize,digits = 3)


matrix <- c()
for(i in 1:nrow(Normalize))
{
  cal <-Normalize[i,]*avgSSE_Normalize[,1]
  matrix <- rbind(matrix,cal)
}

revise_heatmap<- round(matrix,digits = 4)
SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
value<- cbind(SVID,data.table(revise_heatmap))

saveRDS(value,"C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/code/A456_reviseheatmap_test_CAE_SVR.rds")
saveRDS(value,"C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/code/A456_reviseheatmap_test_CAE_SVR.rds")

##### -------------------------------------------
value<- read_rds("C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/code/A456_reviseheatmap_test_CAE_SVR.rds")
value<- read_rds("C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/code/A456_reviseheatmap_test_CAE_SVR.rds")

value<- melt(value)
colnames(value)<-c("SVID","feature","error")
library(ggplot2)


# Plot 
mid<-(min(value$error)+max(value$error))/2
plot <- ggplot(data = value, aes(x = feature, y = SVID)) + 
  geom_tile(aes(fill = error), color = "white", size = 1) +
  scale_fill_gradient2(
    low = 'red', mid = 'white', high = 'steelblue',
    midpoint = mid, guide = 'colourbar', aesthetics = 'fill',limit=c(min(value$error),max(value$error)))+
  xlab("hidden_feature") + 
  theme_grey(base_size = 10) + 
  ggtitle("A456 CAE_SVR: avg_MSE of each SVID (certain_zero-baseline)") +
  # ggtitle("A456 CAE_SVR: avg_MSE of each SVID (certain_nonezero-baseline)") +
  geom_text(aes(label=round(error,digits = 3)),size=3.5)+
  theme(axis.ticks = element_blank(), 
        panel.background = element_blank(), 
        plot.title = element_text(size=20,hjust = 0.5,face="bold"),legend.position="right",legend.direction="vertical",
        axis.text.x= element_text(size = 10,face="bold"),axis.text.y= element_text(size = 10,face="bold"),
        axis.title.x = element_text(size=15,face="bold"),axis.title.y = element_text(size=15,face="bold"))+
  guides(fill = guide_colorbar(barwidth = 1, barheight =7,title.position = "top", title.hjust = 0.5))

print(plot)


ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/A456_CAESVR.png", width=15, height=10)
saveRDS(avgSSE_Normalize,"C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/A456_avgSSE_weight_test_CAE_SVR.rds")

ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/A456_CAESVR.png", width=15, height=10)
saveRDS(avgSSE_Normalize,"C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/A456_avgSSE_weight_test_CAE_SVR.rds")

########## (5/20 補 SVID mean MSE of feature barplot) ##########
reviseheatmap <-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/code/A456_reviseheatmap_test_CAE_SVR.rds")
reviseheatmap <-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/code/A456_reviseheatmap_test_CAE_SVR.rds")

heatmap <- reviseheatmap%>%
  mutate(rowMeans(reviseheatmap[,2:dim(reviseheatmap)[2]]))
colnames(heatmap)[14]<-"sum"

set <- data.frame(heatmap$SVID,heatmap$sum)
colnames(set)<-c("SVID","mean")

plot <- ggplot(set, aes(x =  SVID, y = mean)) +
  geom_bar(stat = "identity",fill="lightblue",width = 0.5) +
  xlab("19 SVID")+ylab("mean of 12 feature(MSE)")+
  ggtitle("A456 overall error ") +
  theme_minimal()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 12,face="bold"), 
        axis.title = element_text(size = 12,face="bold"), 
        plot.title = element_text(size = 30, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=15,face="bold"),
        axis.title.y = element_text(size=15,face="bold"),
        axis.text.y = element_text(size=10),
        axis.text.x = element_text(size=10,face="bold"),
        legend.title = element_text(size=12),
        legend.text = element_text(size=12),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank()) 
print(plot)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/SVID_imp/A456_SVID_imp.png", width=15, height=10)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/SVID_imp/A456_SVID_imp.png", width=15, height=10)


################  一般SVR ML model ----------------------------------------------
# variable important 
library(rminer)
data <- cbind(test_feature,testy)
data <- data.frame(data)
colnames(data)[13]<-"AVG_REMOVAL_RATE"
data_M <- rminer::fit(AVG_REMOVAL_RATE~.,kernel="vanilladot",data=data,model="ksvm")
print(data_M@object)
#A456:(改過)
# SV type: eps-svr  (regression) 
# parameter : epsilon = 0.1  cost C = 1 
# 
# Linear (vanilla) kernel function. 
# 
# Number of Support Vectors : 102 
# 
# Objective Function Value : -71.642 
# Training error : 0.953165 

#A456:
# Support Vector Machine object of class "ksvm" 
# SV type: eps-svr  (regression) 
# parameter : epsilon = 0.1  cost C = 1 
# 
# Linear (vanilla) kernel function. 
# 
# Number of Support Vectors : 218 
# 
# Objective Function Value : -105.0705 
# Training error : 0.452709 

#B456
# Support Vector Machine object of class "ksvm" 
# Support Vector Machine object of class "ksvm" 
# 
# SV type: eps-svr  (regression) 
# SV type: eps-svr  (regression) 
# parameter : epsilon = 0.1  cost C = 1 
# 
# Linear (vanilla) kernel function. 
# 
# Number of Support Vectors : 201 
# 
# Objective Function Value : -74.7629 
# Training error : 0.290525 

data.imp= rminer::Importance(data_M,data,method="sensv")
data.imp$imp

L=list(runs=1,sen=t(data.imp$imp),
       sresponses=data.imp$sresponses)
rminer::mgraph(L,graph="IMP",leg=names(data),col="gray",Grid=10)


importance_data <- data.frame(data.imp$imp[1:12])
colnames(importance_data)<-"Importance"


svr_data <- data.frame(data.table(paste0("X", 1:12)),importance_data)
colnames(svr_data)[1]<-"SVID"

saveRDS(svr_data,"C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/SVR.imp/A456_Varimp_weight.rds")


#######################################################
#######################################################
### 一般svr model:
svr.imp <- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/SVR.imp/A456_Varimp_weight.rds")


svr.imp$SVID<-factor(seq(1,dim(svr.imp)[1],by=1))
svr.imp$Importance <- round(svr.imp$Importance,digits=4)
tmp <- max(-min(svr.imp$Importance),max(svr.imp$Importance))

##加負號　再做ｍｉｎ　ｍａｘ　-------------------------
svr.imp <- data.table(-svr.imp$Importance)
# avgSSE<- data.table(svr.imp$Importance)
maxmin <- function(x) (x - min(x))/(max(x)-min(x)) ## svr 需要做min-max
svr.imp <- apply(svr.imp[,1], 2, maxmin)
svr.imp<- round(svr.imp,digits = 3)

svr.imp <-cbind(data.table(factor(seq(1,dim(svr.imp)[1],by=1))),svr.imp)
colnames(svr.imp)<-c("feature","Importance")
# svr.imp$Importance <- round(svr.imp$Importance,digits=4)
tmp <- max(-min(svr.imp$Importance),max(svr.imp$Importance))

### by each feature barplot  -----------------
plot <- ggplot(svr.imp, aes(x =  feature, y = Importance)) + ###
  geom_bar(stat = "identity",fill="lightblue",width = 0.5) +
  xlab("12 feature")+ylab("  variance based sensitivity")+
  ggtitle("A456: SVR feature important ") +
  theme_minimal()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 12,face="bold"), 
        axis.title = element_text(size = 12,face="bold"), 
        plot.title = element_text(size = 20, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=15,face="bold"),
        axis.title.y = element_text(size=15,face="bold"),
        axis.text.y = element_text(size=15),
        axis.text.x = element_text(size=15),
        legend.title = element_text(size=12),
        legend.text = element_text(size=12),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank()) 
print(plot)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/SVR.imp/A456_Varimp_weight.png", width=15, height=10)

###############################
svr.imp <- read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/SVR.imp/A456_Varimp_weight.rds")

svr.imp$SVID<-factor(seq(1,dim(svr.imp)[1],by=1))
svr.imp$Importance <- round(svr.imp$Importance,digits=4)
# Normalize<-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/A456_minmax_avg_MSE_test_CAE_SVR.rds")
Normalize<-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/A456_minmax_avg_MSE_test_CAE_SVR.rds")

## certain nonezero: 
# avgSSE <- data.table(-svr.imp$Importance) #調整svr 重要度比例
## certain zero: 
avgSSE<- data.table(svr.imp$Importance)

maxmin <- function(x) (x - min(x))/(max(x)-min(x)) ## svr 需要做min-max?
avgSSE_Normalize <- apply(avgSSE[,1], 2, maxmin)
avgSSE_Normalize<- round(avgSSE_Normalize,digits = 3)


matrix <- c()
for(i in 1:nrow(Normalize))
{
  cal <-Normalize[i,]*avgSSE_Normalize[,1]
  matrix <- rbind(matrix,cal)
}
revise_heatmap<- round(matrix,digits = 4)
SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
value<- cbind(SVID,data.table(revise_heatmap))
# saveRDS(value,"C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/SVID_imp/A456_normalSVR_heatmap_test_CAE_SVR.rds")
saveRDS(value,"C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/SVID_imp/A456_normalSVR_heatmap_test_CAE_SVR.rds")

value<- melt(value)
colnames(value)<-c("SVID","feature","error")
library(ggplot2)

# Plot 
mid<-(min(value$error)+max(value$error))/2
plot <- ggplot(data = value, aes(x = feature, y = SVID)) + 
  geom_tile(aes(fill = error), color = "white", size = 1) +
  scale_fill_gradient2(
    low = 'red', mid = 'white', high = 'steelblue',
    midpoint = mid, guide = 'colourbar', aesthetics = 'fill',limit=c(min(value$error),max(value$error)))+
  xlab("12 hidden_feature") + 
  theme_grey(base_size = 10) + 
  ggtitle("A456 CAE_SVR: Overall heatmap") + geom_text(aes(label=round(error,digits = 3)),size=3.5)+
  theme(axis.ticks = element_blank(), 
        panel.background = element_blank(), 
        plot.title = element_text(size=20,hjust = 0.5,face="bold"),legend.position="right",legend.direction="vertical",
        axis.text.x= element_text(size = 10,face="bold"),axis.text.y= element_text(size = 10,face="bold"),
        axis.title.x = element_text(size=15,face="bold"),axis.title.y = element_text(size=15,face="bold"))+
  guides(fill = guide_colorbar(barwidth = 1, barheight =7,title.position = "top", title.hjust = 0.5))

print(plot)

# ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/A456_normalSVR_heatmap.png", width=15, height=10)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/A456_normalSVR_heatmap.png", width=15, height=10)


##畫barplot ------------------------------------------
# reviseheatmap<-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/SVID_imp/A456_normalSVR_heatmap_test_CAE_SVR.rds")
reviseheatmap<-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/SVID_imp/A456_normalSVR_heatmap_test_CAE_SVR.rds")


heatmap <- reviseheatmap%>%
  mutate(rowMeans(reviseheatmap[,2:dim(reviseheatmap)[2]]))
colnames(heatmap)[14]<-"sum"

set <- data.frame(heatmap$SVID,heatmap$sum)
colnames(set)<-c("SVID","mean")

plot <- ggplot(set, aes(x =  SVID, y = mean)) +
  geom_bar(stat = "identity",fill="lightblue",width = 0.5) +
  xlab("19 SVID")+ylab("mean of 12 feature")+
  ggtitle("A456 Overall error ") +
  theme_minimal()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 12,face="bold"), 
        axis.title = element_text(size = 12,face="bold"), 
        plot.title = element_text(size = 30, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=15,face="bold"),
        axis.title.y = element_text(size=15,face="bold"),
        axis.text.y = element_text(size=10),
        axis.text.x = element_text(size=10,face="bold"),
        legend.title = element_text(size=12),
        legend.text = element_text(size=12),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank()) 
print(plot)
# ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/SVID_imp/A456_SVID_SVRimp.png", width=15, height=10)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/SVID_imp/A456_SVID_SVRimp.png", width=15, height=10)


#######################################################
#######################################################
### our model svr model:
avgSSE<-read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_nonezero/Prediction_path/A456_errorSVID_prediction_MSE_test_CAE_SVR.rds")
avgSSE<-read_rds("C:/Users/User/Desktop/2021_0514/model/12/Prediction/Explainable/certain_zero/Prediction_path/A456_errorSVID_prediction_MSE_test_CAE_SVR.rds")

avgSSE<- data.table(unlist(avgSSE))
maxmin <- function(x) (x - min(x))/(max(x)-min(x))
avgSSE_Normalize <- apply(avgSSE[,1], 2, maxmin)
avgSSE_Normalize<- round(avgSSE_Normalize,digits = 3)


seq<-data.table(factor(seq(1,dim(avgSSE_Normalize)[1],by=1)))
set <- cbind(seq,avgSSE_Normalize)
colnames(set)<-c("feature","min_max_weight")

### all value ---------------------------------------------------------------------
plot <- ggplot(set, aes(x =  feature, y = min_max_weight)) + 
  geom_bar(stat = "identity",fill="lightblue",width = 0.5) +
  xlab("12 feature")+ylab(" min_max normalize of MSE  ")+
  # ggtitle("A456: SVR min-max certain nonezero of feature") +
  ggtitle("A456: SVR min-max certain zero of feature") +
  theme_minimal()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 12,face="bold"), ###
        axis.title = element_text(size = 12,face="bold"), ###
        plot.title = element_text(size = 20, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=15,face="bold"),
        axis.title.y = element_text(size=15,face="bold"),
        axis.text.y = element_text(size=15),
        axis.text.x = element_text(size=15),
        legend.title = element_text(size=12),
        legend.text = element_text(size=12),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank()) 
print(plot)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Prediction_path/A456_model_certain_nonezero_minmax_weight.png", width=15, height=10)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Prediction_path/A456_model_certain_zero_minmax_weight.png", width=15, height=10)



### combine together  ---------------------------------------
#our model:
reviseheatmap <-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/code/A456_reviseheatmap_test_CAE_SVR.rds")
# reviseheatmap <-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/code/A456_reviseheatmap_test_CAE_SVR.rds")

heatmap <- reviseheatmap%>%
  mutate(rowMeans(reviseheatmap[,2:dim(reviseheatmap)[2]]))
colnames(heatmap)[14]<-"sum"
set1 <- data.frame(heatmap$SVID,heatmap$sum)
colnames(set1)<-c("SVID","mean")
ours <- cbind(data.table(rep("Proposed method",length=dim(set1)[1])),set1)

# svr model:
reviseheatmap<-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/SVID_imp/A456_normalSVR_heatmap_test_CAE_SVR.rds")
# reviseheatmap<-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/SVID_imp/A456_normalSVR_heatmap_test_CAE_SVR.rds")
heatmap <- reviseheatmap%>%
  mutate(rowMeans(reviseheatmap[,2:dim(reviseheatmap)[2]]))
colnames(heatmap)[14]<-"sum"
set2 <- data.frame(heatmap$SVID,heatmap$sum)
colnames(set2)<-c("SVID","mean")
svrimp <- cbind(data.table(rep("Conventional SVR",length=dim(set2)[1])),set2)

plot_set <- rbind(ours,svrimp)
colnames(plot_set)<-c("method","SVID","mean")


plot <- ggplot(plot_set, aes(x =  SVID, y = mean)) + 
  geom_bar(stat = "identity",fill="lightblue",width = 0.5) +
  facet_wrap(~method, scales="free_y",  ncol=1)+
  # geom_text(aes(label = Importance,hjust = ifelse(Importance > 0 , 1.2, 0)),size=4) +
  xlab("19 variable")+ylab("Mean of 12 feature MSE")+
  theme_minimal()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 12,face="bold"), ###
        axis.title = element_text(size = 12,face="bold"), ###
        # panel.background = element_blank(),
        plot.title = element_text(size = 20, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=15),
        axis.title.y = element_text(size=15),
        axis.text.y = element_text(size=15),
        axis.text.x = element_text(size=10),
        legend.title = element_text(size=15),
        legend.text = element_text(size=12),
        strip.text = element_text(size=20,face="bold",colour = "darkblue"),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank()) ###
print(plot)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/SVID_imp/A456_model_(certain_zero)together_output.png", width=15, height=10)
# ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/SVID_imp/A456_model_(certain_nonzero)together_output.png", width=15, height=10)

##### our method three dataset barplot ------------------------------------------------
A123 <-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/code/A123_reviseheatmap_test_CAE_SVR.rds")
A456 <-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/code/A456_reviseheatmap_test_CAE_SVR.rds")
B456 <-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/code/B456_reviseheatmap_test_CAE_SVR.rds")

A123 <-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/code/A123_reviseheatmap_test_CAE_SVR.rds")
A456 <-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/code/A456_reviseheatmap_test_CAE_SVR.rds")
B456 <-read_rds("C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/code/B456_reviseheatmap_test_CAE_SVR.rds")
List <- list(A123,A456,B456)
set <- c()
for(i in 1:length(List))
{
  reviseheatmap <- List[[i]]
  heatmap<- reviseheatmap%>%
    mutate(rowMeans(reviseheatmap[,2:dim(reviseheatmap)[2]]))
  colnames(heatmap)[14]<-"sum"
  set1 <- data.frame(heatmap$SVID,heatmap$sum)
  colnames(set1)<-c("SVID","mean")
  name<- c("A123","A456","B456")
  ours <- cbind(data.table(rep(name[i],length=dim(set1)[1])),set1)
  set <- rbind(set,ours)
  
}

colnames(set)<-c("dataset","SVID","mean")

my3cols <- c("#E7B800", "#2E9FDF", "#FC4E07")

plot <- ggplot(set, aes(x =  SVID, y = mean,fill=dataset)) + 
  geom_bar(stat = "identity",width = 0.5,position=position_dodge()) +
  scale_fill_manual(values=my3cols)+
  xlab("19 variable")+ylab("Mean of 12 feature MSE")+ggtitle("Evaluate SVID influence in three dataset")+
  theme_minimal()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 12,face="bold"), ###
        axis.title = element_text(size = 12,face="bold"), ###
        # panel.background = element_blank(),
        plot.title = element_text(size = 30, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=20),
        axis.title.y = element_text(size=20),
        axis.text.y = element_text(size=15),
        axis.text.x = element_text(size=15),
        legend.title = element_text(size=20),
        legend.text = element_text(size=12),
        # strip.text = element_text(size=20,face="bold",colour = "darkblue"),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank()) ###
print(plot)
# ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/SVID_imp/threedataset_output.png", width=20, height=10)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/SVID_imp/threedataset_output.png", width=20, height=10)

# certain nonzero - baseline 0:
A123<- set%>%
  filter(dataset=="A123")
A123$SVID<-seq(1,dim(A123)[1],by=1)
revise_A123 <- A123[order(mean,decreasing = F),2:3]

A456<- set%>%
  filter(dataset=="A456")
A456$SVID<-seq(1,dim(A456)[1],by=1)
revise_A456 <- A456[order(mean,decreasing = F),2:3]

B456<- set%>%
  filter(dataset=="B456")
B456$SVID<-seq(1,dim(B456)[1],by=1)
revise_B456 <- B456[order(mean,decreasing = F),2:3]

# certain zero - baseline nonzero
A123<- set%>%
  filter(dataset=="A123")
A123$SVID<-seq(1,dim(A123)[1],by=1)
revise_A123 <- A123[order(mean,decreasing = T),2:3]

A456<- set%>%
  filter(dataset=="A456")
A456$SVID<-seq(1,dim(A456)[1],by=1)
revise_A456 <- A456[order(mean,decreasing = T),2:3]

B456<- set%>%
  filter(dataset=="B456")
B456$SVID<-seq(1,dim(B456)[1],by=1)
revise_B456 <- B456[order(mean,decreasing = T),2:3]

dataset_rank <- cbind(revise_A123[,1],revise_A456[,1],revise_B456[,1])
dataset_rank <- cbind(data.table(seq(1,dim(dataset_rank)[1],by=1)),dataset_rank)
colnames(dataset_rank)<-c("rank","A123","A456","B456")

saveRDS(dataset_rank,"C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/SVID_imp/threedataset_output.rds")
write.csv(dataset_rank,"C:/Users/User/Desktop/2021_0520/12/certain_nonzero/CAE_SVR/Final_chart/SVID_imp/threedataset_output.csv")

saveRDS(dataset_rank,"C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/SVID_imp/threedataset_output.rds")
write.csv(dataset_rank,"C:/Users/User/Desktop/2021_0520/12/certain_zero/CAE_SVR/Final_chart/SVID_imp/threedataset_output.csv")