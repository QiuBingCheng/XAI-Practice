###########################################################################################
############################### flatten 192 ###############################################
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
#########################
###### CAE+ SVR (192)-------------------------------------------------------------------------
## encoder A123
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

## encoder A456/B456
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
loss <- function(y_true,y_pred)
{
  
  tf$reduce_mean(tf$maximum(0.,tf$subtract(tf$abs(tf$subtract(y_pred,y_true)),eplison)))
  
}


callbacks = list(
  callback_model_checkpoint("checkpoints.h5"), callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1)) ### 再調整！！


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

saveRDS(train_loss,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Results_loss/B456_trainloss_CAE_SVR.rds")
saveRDS(test_loss,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Results_loss/B456_testloss_CAE_SVR.rds")
save_model_hdf5(model,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/B456_CAE_SVR.h5")

# saveRDS(train_loss,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Results_loss/A123_trainloss_CAE_SVR.rds")
# saveRDS(test_loss,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Results_loss/A123_testloss_CAE_SVR.rds")
# save_model_hdf5(model,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/A123_CAE_SVR.h5")
## feature extraction--------------------------------------------------------------------------
layer_name<-"flatten"
encoder <- keras_model(inputs=model$input,outputs=get_layer(model,layer_name)$output)
summary(encoder)
test_feature  = encoder %>% predict(xtest) #   245  12
dim(test_feature)
# saveRDS(test_feature,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/B456_testfeature_CAE_SVR.rds")
saveRDS(test_feature,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/A123_testfeature_CAE_SVR.rds")


#####################################################################################
########################### 重建解釋性 #############################################
model <- load_model_hdf5("C:/Users/User/Desktop/2021_0502/model/192/Prediction/A123_CAE_ANN(avg_pool).h5")
model <- load_model_hdf5("C:/Users/User/Desktop/2021_0502/model/192/Prediction/B456_CAE_ANN.h5")
test_feature<- read_rds("C:/Users/User/Desktop/2021_0502/model/192/Prediction/Explainable/A123_testfeature_CAE_ANN.rds")
#### certain - baseline 0: -----------------------------------------------------
recon_list <- list() 
recon_feature <- list()
for(i in 1L:192L)
{
  tmp <- matrix(0:0, nrow = nrow(test_feature), ncol = ncol(test_feature))
  c <- test_feature[,i]
  tmp[,i]<-c
  
  # CAE SVR reconstructed model --------------
  dec_input = layer_input(shape = 192)
  leak1<-get_layer(model,name="leak5")
  dec_reshape<- get_layer(model,name="reshape")
  dec1<- get_layer(model,name="decoder1")
  leak2<-get_layer(model,name="leak6")
  up_samp1<- get_layer(model,name="up_samp1")
  dec2<- get_layer(model,name="decoder2")
  leak3<-get_layer(model,name="leak7")
  up_samp2<- get_layer(model,name="up_samp2")
  dec3<- get_layer(model,name="decoder3")
  leak4<-get_layer(model,name="leak8")
  up_samp3<- get_layer(model,name="up_samp3")
  dec4<- get_layer(model,name="autoencoder")
  
  
  decoder<-keras_model(dec_input,dec4(up_samp3(leak4(dec3(up_samp2(leak3(dec2(up_samp1(leak2(dec1(dec_reshape(leak1(dec_input)))))))))))))
  summary(decoder)
  reconstruct = decoder %>% predict(tmp)# 1981   316   19    1
  dim(reconstruct) #1981   19   19    1
  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
  }
  recon_feature[[i]]<- recon_list
}
# saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_reconstruct_test_CAE_SVR.rds")
saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_nonezero/A123_reconstruct_test_CAE_SVR.rds")

###　測試使用：
recon_list <- list()
recon_feature <- list()
for(i in 1L:192L)
{
  tmp <- matrix(0:0, nrow = nrow(test_feature), ncol = ncol(test_feature))

  # CAE SVR reconstructed model ------------------------------------
  dec_input = layer_input(shape = 192)
  leak1<-get_layer(model,name="leak5")
  dec_reshape<- get_layer(model,name="reshape")
  dec1<- get_layer(model,name="decoder1")
  leak2<-get_layer(model,name="leak6")
  up_samp1<- get_layer(model,name="up_samp1")
  dec2<- get_layer(model,name="decoder2")
  leak3<-get_layer(model,name="leak7")
  up_samp2<- get_layer(model,name="up_samp2")
  dec3<- get_layer(model,name="decoder3")
  leak4<-get_layer(model,name="leak8")
  up_samp3<- get_layer(model,name="up_samp3")
  dec4<- get_layer(model,name="autoencoder")
  decoder<-keras_model(dec_input,dec4(up_samp3(leak4(dec3(up_samp2(leak3(dec2(up_samp1(leak2(dec1(dec_reshape(leak1(dec_input)))))))))))))
  summary(decoder)
  reconstruct = decoder %>% predict(tmp)# 1981   316   19    1
  dim(reconstruct) #1981   19   19    1
  
  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
  }
  recon_feature[[i]]<- recon_list
}
# saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_reconstruct_test_CAE_SVR(verify_0).rds")
saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_nonezero/A123_reconstruct_test_CAE_SVR(verify_0).rds")

## certain 0 - baseline: -------------------------------------------------------------
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
model <- load_model_hdf5("C:/Users/User/Desktop/2021_0502/model/192/Prediction/B456_CAE_ANN.h5")
summary(model)
test_feature<- read_rds("C:/Users/User/Desktop/2021_0502/model/192/Prediction/Explainable/B456_testfeature_CAE_ANN.rds")
recon_list <- list() 
recon_feature <- list()
for(i in 1L:192L)
{
  
  cal_set <- test_feature
  tmp <- matrix(0:0,nrow=nrow(cal_set),ncol=1)
  cal_set[,i]<-tmp
  

  # CAE SVR reconstructed model ------------------------------------
  dec_input = layer_input(shape = 192)
  leak1<-get_layer(model,name="leak5")
  dec_reshape<- get_layer(model,name="reshape")
  dec1<- get_layer(model,name="decoder1")
  leak2<-get_layer(model,name="leak6")
  up_samp1<- get_layer(model,name="up_samp1")
  dec2<- get_layer(model,name="decoder2")
  leak3<-get_layer(model,name="leak7")
  up_samp2<- get_layer(model,name="up_samp2")
  dec3<- get_layer(model,name="decoder3")
  leak4<-get_layer(model,name="leak8")
  up_samp3<- get_layer(model,name="up_samp3")
  dec4<- get_layer(model,name="autoencoder")
  
  
  decoder<-keras_model(dec_input,dec4(up_samp3(leak4(dec3(up_samp2(leak3(dec2(up_samp1(leak2(dec1(dec_reshape(leak1(dec_input)))))))))))))
  summary(decoder)
  reconstruct = decoder %>% predict(cal_set)# 1981   316   19    1
  dim(reconstruct) #1981   19   19    1
  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
  }
  recon_feature[[i]]<- recon_list
}

# saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_reconstruct_test_CAE_SVR.rds")
saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_zero/A123_reconstruct_test_CAE_SVR.rds")

###　測試使用：
recon_list <- list()
recon_feature <- list()
for(i in 1L:192L)
{

  # CAE SVR reconstructed model ------------------------------------
  dec_input = layer_input(shape = 192)
  leak1<-get_layer(model,name="leak5")
  dec_reshape<- get_layer(model,name="reshape")
  dec1<- get_layer(model,name="decoder1")
  leak2<-get_layer(model,name="leak6")
  up_samp1<- get_layer(model,name="up_samp1")
  dec2<- get_layer(model,name="decoder2")
  leak3<-get_layer(model,name="leak7")
  up_samp2<- get_layer(model,name="up_samp2")
  dec3<- get_layer(model,name="decoder3")
  leak4<-get_layer(model,name="leak8")
  up_samp3<- get_layer(model,name="up_samp3")
  dec4<- get_layer(model,name="autoencoder")
  
  decoder<-keras_model(dec_input,dec4(up_samp3(leak4(dec3(up_samp2(leak3(dec2(up_samp1(leak2(dec1(dec_reshape(leak1(dec_input)))))))))))))
  summary(decoder)
  reconstruct = decoder %>% predict(test_feature)# 1981   316   19    1
  dim(reconstruct) #1981   19   19    1
  
  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
  }
  recon_feature[[i]]<- recon_list
}
# saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_reconstruct_test_CAE_SVR(verify_0).rds")
saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_zero/A123_reconstruct_test_CAE_SVR(verify_0).rds")


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
xtest<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/A123_xtest.rds")
xtest <- array_reshape(xtest,dim=c(dim(xtest)[1],dim(xtest)[2],dim(xtest)[3]*1)) 

# recon_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_reconstruct_test_CAE_SVR.rds")
# recon_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_reconstruct_test_CAE_SVR(verify_0).rds")
# 
# recon_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_reconstruct_test_CAE_SVR.rds")
# recon_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_reconstruct_test_CAE_SVR(verify_0).rds")


recon_feature <- read_rds("C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_nonezero/A123_reconstruct_test_CAE_SVR.rds")
recon_feature <- read_rds("C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_nonezero/A123_reconstruct_test_CAE_SVR(verify_0).rds")

recon_feature <- read_rds("C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_zero/A123_reconstruct_test_CAE_SVR.rds")
recon_feature <- read_rds("C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_zero/A123_reconstruct_test_CAE_SVR(verify_0).rds")

dim(xtest)
corr_error3<- list()
corr_errorSVID_sum2<-list()

for(i in 1L:192L)
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
  for(i in 1:192)
  {
    tmp <- origin_errorSVID_MSE[[i]][[j]]
    tmp <- data.table(tmp)
    null<-cbind(null,tmp)
    colnames(null) <- paste0('X', 1:(ncol(null)))
    
  }
  each_waferList[[j]]<-null
}

# saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_nonezero/A123_Eachwafer_MSE_test_CAE_SVR.rds")
# saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_nonezero/A123_Eachwafer_MSE_test_CAE_SVR(verify0).rds")
# saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_zero/A123_Eachwafer_MSE_test_CAE_SVR.rds")
saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_zero/A123_Eachwafer_MSE_test_CAE_SVR(verify0).rds")

each_waferList<- read_rds("C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_nonezero/A123_Eachwafer_MSE_test_CAE_SVR.rds")
each_waferList2<- read_rds("C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_nonezero/A123_Eachwafer_MSE_test_CAE_SVR(verify0).rds")
each_waferList<- read_rds("C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_zero/A123_Eachwafer_MSE_test_CAE_SVR.rds")
each_waferList2<- read_rds("C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_zero/A123_Eachwafer_MSE_test_CAE_SVR(verify0).rds")

tmp <- matrix(0:0, nrow = 19, ncol = 192)
tmp_list<- list()
tmp <-data.table(tmp)

for(j in 1:dim(xtest)[1])
{
  tmp_list[[j]] <- each_waferList[[j]]-each_waferList2[[j]]
  tmp<-tmp+tmp_list[[j]]
  
}
avg_wafer_error <- tmp/length(tmp_list)
saveRDS(avg_wafer_error,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_nonezero/A123_avg_MSE(actual-null)_test_CAE_SVR.rds")
saveRDS(avg_wafer_error,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_zero/A123_avg_MSE(actual-null)_test_CAE_SVR.rds")

### visualize---------------------------------------------------------------------------------------
## origin heatmap:
avg_wafer_error<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_avg_MSE(actual-null)_test_CAE_SVR.rds")
avg_wafer_error<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_avg_MSE(actual-null)_test_CAE_SVR.rds")

avg_wafer_error<- round(avg_wafer_error,digits = 4)

SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
value<- cbind(SVID,data.table(avg_wafer_error))

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
  xlab("192 hidden_feature") + ylab("19 SVID")+
  theme_grey(base_size = 10) + 
  # ggtitle("B456 CAE_SVR: avg_MSE of each SVID (certain_nonezero-baseline)") +
  ggtitle("B456 CAE_SVR:avg_MSE of each SVID (certain_zero-baseline)") +
  theme(axis.ticks = element_blank(), 
        panel.background = element_blank(), 
        plot.title = element_text(size=50,hjust = 0.5,face="bold"),legend.position="right",legend.direction="vertical",
        # axis.text.x= element_text(size = 5,face="bold"),
        axis.text.y= element_text(size = 25,face="bold"),
        axis.text.x=element_blank(),
        # axis.text.y= element_blank(),
        axis.title.x = element_text(size=35,face="bold"),#axis.title.y = element_text(size=35,face="bold"),
        axis.title.y = element_blank())+
  
  guides(fill = guide_colorbar(barwidth = 1, barheight =7,title.position = "top", title.hjust = 0.5))

print(plot)


ggsave(plot, file="C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/Reconstruction_path/origin/B456_CAESVR.png", width=45, height=20)
ggsave(plot, file="C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/Reconstruction_path/origin/B456_CAESVR.png", width=45, height=20)

# ggsave(plot, file="C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_nonezero/Reconstruction_path/origin/A123_CAESVR.png", width=45, height=20)
# ggsave(plot, file="C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_zero/Reconstruction_path/origin/A123_CAESVR.png", width=45, height=20)





### 2021/05/14 each SVID range
avg_wafer_error<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_avg_MSE(actual-null)_test_CAE_SVR.rds")
avg_wafer_error<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_avg_MSE(actual-null)_test_CAE_SVR.rds")

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
  scale_fill_gradient2(
    low = 'red', mid = 'white', high = 'steelblue',
    midpoint = mid, guide = 'colourbar', aesthetics = 'fill',limit=c(min(value$error),max(value$error)))+
  xlab("192 hidden_feature") + ylab("19 SVID")+
  theme_grey(base_size = 10) + 
  # ggtitle("B456 CAE_SVR: Min-max avg_MSE of each SVID (certain_nonezero-baseline)") +
  ggtitle("B456 CAE_SVR: Min-max avg_MSE of each SVID (certain_zero-baseline)") +
  theme(axis.ticks = element_blank(), 
        panel.background = element_blank(), 
        plot.title = element_text(size=50,hjust = 0.5,face="bold"),legend.position="right",legend.direction="vertical",
        # axis.text.x= element_text(size = 5,face="bold"),axis.text.y= element_text(size = 10,face="bold"),
        axis.text.x=element_blank(),axis.text.y= element_text(size=25,face="bold"),
        # axis.title.x = element_text(size=35,face="bold"),axis.title.y = element_text(size=35,face="bold"))+
        axis.title.x = element_text(size=35,face="bold"),axis.title.y = element_blank())+
  guides(fill = guide_colorbar(barwidth = 1, barheight =7,title.position = "top", title.hjust = 0.5))

print(plot)
ggsave(plot, file="C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/Reconstruction_path/origin/min-max_B456_CAESVR.png", width=45, height=20)
ggsave(plot, file="C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/Reconstruction_path/origin/min-max_B456_CAESVR.png", width=45, height=20)

# saveRDS(Normalize,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_nonezero/A123_minmax_avg_MSE_test_CAE_SVR.rds")
# saveRDS(Normalize,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/A123_minmax_avg_MSE_test_CAE_SVR.rds")


#####################################################################################
######################## 預測解釋性 #################################################
### certain - baseline --------------------------------------------
summary(model)
test_feature<- read_rds("C:/Users/User/Desktop/2021_0502/model/192/Prediction/Explainable/A123_testfeature_CAE_ANN.rds")

pred_list <- list()
pred_feature <- list()
for(i in 1L:192L)
{
  tmp <- matrix(0:0, nrow = nrow(test_feature), ncol = ncol(test_feature))
  c <- test_feature[,i]
  tmp[,i]<-c

  ## CAE_SVR classifier model -------------------------------------
  pred_input = layer_input(shape=192)
  dense1<-get_layer(model,name="dec_pred")
  leak1 <-get_layer(model,name="leak4")
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
saveRDS(pred_feature,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_nonezero/A123_prediction_test_CAE_SVR.rds")

###　測試使用：
pred_list <- list()
pred_feature <- list()
for(i in 1L:192L)
{
  tmp <- matrix(0:0, nrow = nrow(test_feature), ncol = ncol(test_feature))
  
  
  ## CAE_SVR classifier model -------------------------------------
  pred_input = layer_input(shape=192)
  dense1<-get_layer(model,name="dec_pred")
  leak1 <-get_layer(model,name="leak4")
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
saveRDS(pred_feature,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_nonezero/A123_prediction_test_CAE_SVR(verify_0).rds")

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
summary(model)
model <- load_model_hdf5("C:/Users/User/Desktop/2021_0502/model/192/Prediction/B456_CAE_ANN.h5")
summary(model)
test_feature<- read_rds("C:/Users/User/Desktop/2021_0502/model/192/Prediction/Explainable/A456_testfeature_CAE_ANN.rds")

pred_list <- list()
pred_feature <- list()
for(i in 1L:192L)
{
  cal_set <- test_feature
  tmp <- matrix(0:0,nrow=nrow(cal_set),ncol=1)
  cal_set[,i]<-tmp
  

  ## CAE_SVR classifier model -------------------------------------
  pred_input = layer_input(shape=192)
  dense1<-get_layer(model,name="dec_pred")
  leak1 <-get_layer(model,name="leak4")
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

saveRDS(pred_feature,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_zero/A123_prediction_test_CAE_SVR.rds")

###　測試使用：
pred_list <- list()
pred_feature <- list()
for(i in 1L:192L)
{
  
  ## CAE_SVR classifier model -------------------------------------
  pred_input = layer_input(shape=192)
  dense1<-get_layer(model,name="dec_pred")
  leak1 <-get_layer(model,name="leak4")
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
saveRDS(pred_feature,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_zero/A123_prediction_test_CAE_SVR(verify_0).rds")



#### Actual_corr - Reconstruct_corr
ytest<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/A123_testy.rds")

pred_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/A123_prediction_test_CAE_SVR.rds") 
pred_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/A123_prediction_test_CAE_SVR(verify_0).rds") 
pred_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/A123_prediction_test_CAE_SVR.rds") 
pred_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/A123_prediction_test_CAE_SVR(verify_0).rds") 

# corr_error3<- list()
corr_errorSVID_sum2<-list()
SSE_matrix<- c()
# SSE_matrix<-matrix(0:0, nrow = 240, ncol = 192)
for(i in 1L:192L)
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



saveRDS(SSE_matrix,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/A456_errorSVID_prediction_SSEmatrix_test_CAE_SVR.rds")
saveRDS(corr_errorSVID_sum2,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/A456_errorSVID_prediction_MSE_test_CAE_SVR.rds")

saveRDS(SSE_matrix,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/A456_errorSVID_prediction_SSEmatrix_test_CAE_SVR(verify0).rds")
saveRDS(corr_errorSVID_sum2,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/A456_errorSVID_prediction_MSE_test_CAE_SVR(verify0).rds")

saveRDS(SSE_matrix,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/A456_errorSVID_prediction_SSEmatrix_test_CAE_SVR.rds")
saveRDS(corr_errorSVID_sum2,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/A456_errorSVID_prediction_MSE_test_CAE_SVR.rds")

saveRDS(SSE_matrix,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/A456_errorSVID_prediction_SSEmatrix_test_CAE_SVR(verify0).rds")
saveRDS(corr_errorSVID_sum2,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/A456_errorSVID_prediction_MSE_test_CAE_SVR(verify0).rds")


origin_errorSVID_MSE<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/A123_errorSVID_prediction_MSE_test_CAE_SVR.rds")
origin_errorSVID_MSE2<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/A123_errorSVID_prediction_MSE_test_CAE_SVR(verify0).rds")
origin_errorSVID_SSEmatrix<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/A123_errorSVID_prediction_SSEmatrix_test_CAE_SVR.rds")
origin_errorSVID_SSEmatrix2<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/A123_errorSVID_prediction_SSEmatrix_test_CAE_SVR(verify0).rds")

origin_errorSVID_MSE<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/A123_errorSVID_prediction_MSE_test_CAE_SVR.rds")
origin_errorSVID_MSE2<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/A123_errorSVID_prediction_MSE_test_CAE_SVR(verify0).rds")
origin_errorSVID_SSEmatrix<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/A123_errorSVID_prediction_SSEmatrix_test_CAE_SVR.rds")
origin_errorSVID_SSEmatrix2<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/A123_errorSVID_prediction_SSEmatrix_test_CAE_SVR(verify0).rds")

certain_MSE <- data.frame(unlist(origin_errorSVID_MSE))
baseline_MSE <- data.frame(unlist(origin_errorSVID_MSE2))
pred_performance<- certain_MSE-baseline_MSE

SSE_performance<-origin_errorSVID_SSEmatrix-origin_errorSVID_SSEmatrix2
colnames(SSE_performance) <- paste0('X', 1:(ncol(SSE_performance)))

saveRDS(SSE_performance,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/A123_errorSVID_prediction_SSEpreformance_test_CAE_SVR.rds")
saveRDS(pred_performance,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/A123_errorSVID_prediction_MSEperformance_test_CAE_SVR.rds")

saveRDS(SSE_performance,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/A123_errorSVID_prediction_SSEpreformance_test_CAE_SVR.rds")
saveRDS(pred_performance,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/A123_errorSVID_prediction_MSEperformance_test_CAE_SVR.rds")


### visualize---------------------------------------------------------------------------------------
SSE_performance<- readRDS("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_errorSVID_prediction_SSEpreformance_test_CAE_SVR.rds")
SSE_performance<- readRDS("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_errorSVID_prediction_SSEpreformance_test_CAE_SVR.rds")

value <- data.table(SSE_performance)
value<- round(value,digits = 4)

mean2 <- data.table(colMeans(value))
y_value2<- factor(seq(1,dim(value)[2],by=1),levels =seq(1,dim(value)[2],by=1))
cal2 <- cbind(y_value2,mean2)
colnames(cal2)<-c("wafer","mean")

saveRDS(cal2,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_avgSSE_test_CAE_SVR.rds")
saveRDS(cal2,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_avgSSE_test_CAE_SVR.rds")


###################
# avgSSE<-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/A123_errorSVID_prediction_MSE_test_CAE_SVR.rds")

avgSSE<-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_avgSSE_test_CAE_SVR.rds")
avgSSE<-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_avgSSE_test_CAE_SVR.rds")

# avgSSE<- data.table(unlist(avgSSE))
maxmin <- function(x) (x - min(x))/(max(x)-min(x))
avgSSE_Normalize <- apply(avgSSE[,2], 2, maxmin)
avgSSE_Normalize<- round(avgSSE_Normalize,digits = 3)

value<- cbind(data.table(factor(seq(1,dim(avgSSE_Normalize)[1],by=1))),avgSSE_Normalize)
colnames(value)<-c("number","mean")

plot <- ggplot(value, aes(x =  number, y = mean)) +
  geom_bar(stat = "identity",fill="lightblue",width = 0.5) +
  xlab("192 feature")+ylab("min max normalize of MSE")+
  ggtitle("MSE proportion of 192 hidden feature ") +
  theme_minimal()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 20,face="bold"), 
        axis.title = element_text(size = 20,face="bold"), 
        plot.title = element_text(size = 40, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=30,face="bold"),
        axis.title.y = element_text(size=30,face="bold"),
        axis.text.y = element_text(size=15),
        axis.text.x = element_text(size=8,face="bold"),
        legend.title = element_text(size=12),
        legend.text = element_text(size=12),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank()) 
print(plot)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_nonzero/CAE_SVR/Prediction_path/B456_certain_nonzero_minmax_weight.png", width=30, height=10)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Prediction_path/B456_certain_zero_minmax_weight.png", width=30, height=10)

########################################################################################
########################## prediction +reconstruction heatmap ##########################
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
avgSSE<-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_avgSSE_test_CAE_SVR.rds")
# avgSSE<-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_errorSVID_prediction_MSE_test_CAE_SVR.rds")
Normalize<-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_minmax_avg_MSE_test_CAE_SVR.rds")

# avgSSE<-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_errorSVID_prediction_MSE_test_CAE_SVR.rds")
avgSSE<-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_avgSSE_test_CAE_SVR.rds")

Normalize<-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_minmax_avg_MSE_test_CAE_SVR.rds")



# avgSSE<- data.table(unlist(avgSSE))
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
saveRDS(value,"C:/Users/User/Desktop/2021_0520/192/certain_nonzero/CAE_SVR/Final_chart/code/B456_reviseheatmap_test_CAE_SVR.rds")
saveRDS(value,"C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/code/B456_reviseheatmap_test_CAE_SVR.rds")



# value<- read_rds("C:/Users/User/Desktop/2021_0520/192/certain_nonzero/CAE_SVR/Final_chart/code/B456_reviseheatmap_test_CAE_SVR.rds")
value<- read_rds("C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/code/B456_reviseheatmap_test_CAE_SVR.rds")

value<- melt(value)
colnames(value)<-c("SVID","feature","error")
library(ggplot2)

# Plot 
mid<-(min(value$error)+max(value$error))/2
plot <- ggplot(data = value, aes(x = feature, y = SVID)) + 
  geom_tile(aes(fill = error), color = "white", size = 1) +
  # scale_fill_gradient(low = "gray95", high = "tomato",limits=c(min(value$error), max(value$error)))+ 
  scale_fill_gradient2(
    low = 'red', mid = 'white', high = 'steelblue',
    midpoint = mid, guide = 'colourbar', aesthetics = 'fill',limit=c(min(value$error),max(value$error)))+
  xlab("192 hidden_feature") + #ylab("19 SVID")+
  theme_grey(base_size = 10) + 
  ggtitle("B456 CAE_SVR: avg_MSE of each SVID (certain_zero-baseline)") +
  # ggtitle("B456 CAE_SVR: avg_MSE of each SVID (certain_nonezero-baseline)") +
  theme(axis.ticks = element_blank(), 
        panel.background = element_blank(), 
        plot.title = element_text(size=50,hjust = 0.5,face="bold"),legend.position="right",legend.direction="vertical",
        axis.text.x= element_text(size = 5,face="bold"),axis.text.y= element_text(size = 20,face="bold"),
        # axis.text.y= element_text(size = 20,face="bold"),axis.text.x = element_blank(),
        axis.title.x = element_text(size=40,face="bold"),axis.title.y =element_blank(), #element_text(size=40,face="bold"),
        legend.key.height= unit(60, 'cm'),
        legend.key.width= unit(40, 'cm'),
        legend.title = element_text(size=30),
        legend.text = element_text(size=20))+
  guides(fill = guide_colorbar(barwidth = 1, barheight =7,title.position = "top", title.hjust = 0.5))

print(plot)

# ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_nonzero/CAE_SVR/Final_chart/B456_CAESVR.png", width=45, height=20)
# saveRDS(avgSSE_Normalize,"C:/Users/User/Desktop/2021_0520/192/certain_nonzero/CAE_SVR/Final_chart/B456_avgSSE_weight_test_CAE_SVR.rds")

ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/B456_CAESVR.png", width=45, height=20)
saveRDS(avgSSE_Normalize,"C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/B456_avgSSE_weight_test_CAE_SVR.rds")


################################################################

########## (5/20 補 SVID mean MSE of feature barplot) ##########
reviseheatmap <-read_rds("C:/Users/User/Desktop/2021_0520/192/certain_nonzero/CAE_SVR/Final_chart/code/B456_reviseheatmap_test_CAE_SVR.rds")
# reviseheatmap <-read_rds("C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/code/B456_reviseheatmap_test_CAE_SVR.rds")

heatmap <- reviseheatmap%>%
  mutate(rowMeans(reviseheatmap[,2:dim(reviseheatmap)[2]]))
colnames(heatmap)[194]<-"sum"

set <- data.frame(heatmap$SVID,heatmap$sum)
colnames(set)<-c("SVID","mean")

plot <- ggplot(set, aes(x =  SVID, y = mean)) +
  geom_bar(stat = "identity",fill="lightblue",width = 0.5) +
  xlab("19 SVID")+ylab("mean of 192 feature(MSE)")+
  ggtitle("B456 overall error ") +
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
# ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_nonzero/CAE_SVR/Final_chart/SVID_imp/B456_SVID_imp.png", width=15, height=10)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/SVID_imp/B456_SVID_imp.png", width=15, height=10)


################  一般SVR ML model ----------------------------------------------

test_feature<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/B456_testfeature_CAE_SVR.rds")
testy<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/B456_testy.rds")
# variable important 
library(rminer)
data <- cbind(test_feature,testy)
data <- data.frame(data)
colnames(data)[193]<-"AVG_REMOVAL_RATE"
data_M <- rminer::fit(AVG_REMOVAL_RATE~.,kernel="vanilladot",data=data,model="ksvm")
print(data_M@object)

# A123 ------------------------------------------
# Support Vector Machine object of class "ksvm" 
# SV type: eps-svr  (regression) 
# parameter : epsilon = 0.1  cost C = 1 
# 
# Linear (vanilla) kernel function. 
# 
# Number of Support Vectors : 103 
# 
# Objective Function Value : -47.5642 
# Training error : 0.530653 

# 5/18
# Support Vector Machine object of class "ksvm" 
# 
# SV type: eps-svr  (regression) 
# parameter : epsilon = 0.1  cost C = 1 
# 
# Linear (vanilla) kernel function. 
# 
# Number of Support Vectors : 103 
# 
# Objective Function Value : -45.3426 
# Training error : 0.541507 


# Support Vector Machine object of class "ksvm" 
# 
# SV type: eps-svr  (regression) 
# parameter : epsilon = 0.1  cost C = 1 
# 
# Linear (vanilla) kernel function. 
# 
# Number of Support Vectors : 102 
# 
# Objective Function Value : -46.7633 
# Training error : 0.529169 

# A456 -----------------------------------------
# Support Vector Machine object of class "ksvm" 
# 
# SV type: eps-svr  (regression) 
# parameter : epsilon = 0.1  cost C = 1 
# 
# Linear (vanilla) kernel function. 
# 
# Number of Support Vectors : 225 
# 
# Objective Function Value : -40.5567 
# Training error : 0.135473 

# B456 -----------------------------------------
# Support Vector Machine object of class "ksvm" 
# 
# SV type: eps-svr  (regression) 
# parameter : epsilon = 0.1  cost C = 1 
# 
# Linear (vanilla) kernel function. 
# 
# Number of Support Vectors : 224 
# 
# Objective Function Value : -24.3306 
# Training error : 0.067989 

data.imp=  Importance(data_M,data,method="sensv")
data.imp$imp

L=list(runs=1,sen=t(data.imp$imp),
       sresponses=data.imp$sresponses)
mgraph(L,graph="IMP",leg=names(data),col="gray",Grid=10)


importance_data <- data.frame(data.imp$imp[1:192])
colnames(importance_data)<-"Importance"


svr_data <- data.frame(data.table(paste0("X", 1:192)),importance_data)
colnames(svr_data)[1]<-"SVID"
# saveRDS(svr_data,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/SVR.imp/A123_Varimp_weight(avg_pool).rds")
# saveRDS(svr_data,"C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/SVR.imp/A123_Varimp_weight.rds")
saveRDS(svr_data,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/SVR.imp/A123_Varimp_weight.rds")

### mse:
P=data.table(predict(data_M,test_feature))
eval <- cbind(testy,P)
colnames(eval)<-c("actual","predict")
eval <- eval%>%mutate(SSE=(actual-predict)^2)
mse <- sum(eval$SSE)/dim(eval)[1]
print(mse)



#######################################################
#######################################################
### 一般svr model:
svr.imp <- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/SVR.imp/A123_Varimp_weight(avg_pool).rds")

colnames(svr.imp)<-c("feature","Importance")
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
  xlab("192 feature")+ylab("  variance based sensitivity")+
  ggtitle("A123: SVR feature important ") +
  theme_minimal()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 12,face="bold"), 
        axis.title = element_text(size = 12,face="bold"), 
        plot.title = element_text(size = 40, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=35,face="bold"),
        axis.title.y = element_text(size=35,face="bold"),
        axis.text.y = element_text(size=10),
        axis.text.x = element_text(size=6),
        legend.title = element_text(size=12),
        legend.text = element_text(size=12),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank()) 
print(plot)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/SVR.imp/A123_Varimp_weight.png", width=30, height=10)

####### 一般svr model overall heatmap #################################
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
svr.imp <- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/SVR.imp/A123_Varimp_weight(avg_pool).rds")
svr.imp <- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/SVR.imp/A456_Varimp_weight.rds")
svr.imp$SVID<-factor(seq(1,dim(svr.imp)[1],by=1))
svr.imp$Importance <- round(svr.imp$Importance,digits=4)
# Normalize<-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_minmax_avg_MSE_test_CAE_SVR.rds")
Normalize<-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/A123_minmax_avg_MSE_test_CAE_SVR.rds")

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
# saveRDS(value,"C:/Users/User/Desktop/2021_0520/192/certain_nonzero/CAE_SVR/Final_chart/SVID_imp/B456_normalSVR_heatmap_test_CAE_SVR.rds")
saveRDS(value,"C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/SVID_imp/A123_normalSVR_heatmap_test_CAE_SVR.rds")

value<- melt(value)
colnames(value)<-c("SVID","feature","error")
library(ggplot2)

# Plot 
mid<-(min(value$error)+max(value$error))/2
plot <- ggplot(data = value, aes(x = feature, y = SVID)) + 
  geom_tile(aes(fill = error), color = "white", size = 1) +
  # scale_fill_gradient(low = "gray95", high = "tomato",limits=c(min(value$error), max(value$error)))+ 
  scale_fill_gradient2(
    low = 'red', mid = 'white', high = 'steelblue',
    midpoint = mid, guide = 'colourbar', aesthetics = 'fill',limit=c(min(value$error),max(value$error)))+
  xlab("192 hidden_feature") + #ylab("19 SVID")+
  theme_grey(base_size = 10) + 
  # ggtitle("A123 CAE_SVR: avg_MSE of each SVID (certain_zero-baseline)") +
  ggtitle("B456 CAE_SVR: Overall heatmap") +
  theme(axis.ticks = element_blank(), 
        panel.background = element_blank(), 
        plot.title = element_text(size=50,hjust = 0.5,face="bold"),legend.position="right",legend.direction="vertical",
        axis.text.x= element_text(size = 5,face="bold"),axis.text.y= element_text(size = 20,face="bold"),
        # axis.text.y= element_text(size = 20,face="bold"),axis.text.x = element_blank(),
        axis.title.x = element_text(size=40,face="bold"),axis.title.y =element_blank(), 
        legend.key.height= unit(60, 'cm'),
        legend.key.width= unit(40, 'cm'),
        legend.title = element_text(size=30),
        legend.text = element_text(size=20))+
  guides(fill = guide_colorbar(barwidth = 1, barheight =7,title.position = "top", title.hjust = 0.5))

print(plot)

# ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_nonzero/CAE_SVR/Final_chart/B456_normalSVR_heatmap.png", width=45, height=20)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/B456_normalSVR_heatmap.png", width=45, height=20)
##畫barplot ------------------------------------------
reviseheatmap<-read_rds("C:/Users/User/Desktop/2021_0520/192/certain_nonzero/CAE_SVR/Final_chart/SVID_imp/B456_normalSVR_heatmap_test_CAE_SVR.rds")
reviseheatmap<-read_rds("C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/SVID_imp/B456_normalSVR_heatmap_test_CAE_SVR.rds")

heatmap <- reviseheatmap%>%
  mutate(rowMeans(reviseheatmap[,2:dim(reviseheatmap)[2]]))
colnames(heatmap)[194]<-"sum"

set <- data.frame(heatmap$SVID,heatmap$sum)
colnames(set)<-c("SVID","mean")

plot <- ggplot(set, aes(x =  SVID, y = mean)) +
  geom_bar(stat = "identity",fill="lightblue",width = 0.5) +
  xlab("19 SVID")+ylab("mean of 192 feature")+
  ggtitle("B456 Overall error ") +
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
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_nonzero/CAE_SVR/Final_chart/SVID_imp/B456_SVID_SVRimp.png", width=15, height=10)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/SVID_imp/B456_SVID_SVRimp.png", width=15, height=10)


#######################################################
#######################################################
### our model svr model:
avgSSE<-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/A123_errorSVID_prediction_MSE_test_CAE_SVR.rds")
avgSSE<- data.table(unlist(avgSSE))
maxmin <- function(x) (x - min(x))/(max(x)-min(x))
avgSSE_Normalize <- apply(avgSSE[,1], 2, maxmin)
avgSSE_Normalize<- round(avgSSE_Normalize,digits = 3)


# avgSSE_Normalize<-read_rds("C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_zero/A123_avgSSE_weight_test_CAE_SVR.rds")
# avgSSE_Normalize<-read_rds("C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/certain_nonezero/A123_avgSSE_weight_test_CAE_SVR.rds")

seq<-data.table(factor(seq(1,dim(avgSSE_Normalize)[1],by=1)))
set <- cbind(seq,avgSSE_Normalize)
colnames(set)<-c("feature","min_max_weight")

### all value ---------------------------------------------------------------------
plot <- ggplot(set, aes(x =  feature, y = min_max_weight)) + 
  geom_bar(stat = "identity",fill="lightblue",width = 0.5) +
  xlab("192 feature")+ylab(" min_max normalize of MSE  ")+
  ggtitle("A123: SVR min-max certain nonezero of feature") +
  theme_minimal()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 12,face="bold"), ###
        axis.title = element_text(size = 12,face="bold"), ###
        plot.title = element_text(size = 40, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=35,face="bold"),
        axis.title.y = element_text(size=35,face="bold"),
        axis.text.y = element_text(size=10),
        axis.text.x = element_text(size=6),
        legend.title = element_text(size=12),
        legend.text = element_text(size=12),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank()) 
print(plot)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_nonzero/Prediction_path/A123_model_certain_zero_minmax_weight.png", width=30, height=10)

# ggsave(plot, file="C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/SVR.imp/A123_model_certain_zero_minmax_weight.png", width=30, height=10)
# ggsave(plot, file="C:/Users/User/Desktop/2021_0518/model/192/Prediction/Explainable/SVR.imp/A123_model_certain_nonezero_minmax_weight.png", width=30, height=10)


### combine together  ---------------------------------------
#our model:
# reviseheatmap <-read_rds("C:/Users/User/Desktop/2021_0520/192/certain_nonzero/Final_chart/code/B456_reviseheatmap_test_CAE_SVR.rds")
reviseheatmap <-read_rds("C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/code/B456_reviseheatmap_test_CAE_SVR.rds")

heatmap <- reviseheatmap%>%
  mutate(rowMeans(reviseheatmap[,2:dim(reviseheatmap)[2]]))
colnames(heatmap)[194]<-"sum"
set1 <- data.frame(heatmap$SVID,heatmap$sum)
colnames(set1)<-c("SVID","mean")
ours <- cbind(data.table(rep("Proposed method",length=dim(set1)[1])),set1)

# svr model:
# reviseheatmap<-read_rds("C:/Users/User/Desktop/2021_0520/192/certain_nonzero/Final_chart/SVID_imp/B456_normalSVR_heatmap_test_CAE_SVR.rds")
reviseheatmap<-read_rds("C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/SVID_imp/B456_normalSVR_heatmap_test_CAE_SVR.rds")
heatmap <- reviseheatmap%>%
  mutate(rowMeans(reviseheatmap[,2:dim(reviseheatmap)[2]]))
colnames(heatmap)[194]<-"sum"
set2 <- data.frame(heatmap$SVID,heatmap$sum)
colnames(set2)<-c("SVID","mean")
svrimp <- cbind(data.table(rep("Conventional SVR",length=dim(set2)[1])),set2)

plot_set <- rbind(ours,svrimp)
colnames(plot_set)<-c("method","SVID","mean")


plot <- ggplot(plot_set, aes(x =  SVID, y = mean)) + 
  geom_bar(stat = "identity",fill="lightblue",width = 0.5) +
  facet_wrap(~method, scales="free_y",  ncol=1)+
  # geom_text(aes(label = Importance,hjust = ifelse(Importance > 0 , 1.2, 0)),size=4) +
  xlab("19 variable")+ylab("Mean of 192 feature MSE")+
  theme_minimal()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 12,face="bold"), ###
        axis.title = element_text(size = 12,face="bold"), ###
        # panel.background = element_blank(),
        plot.title = element_text(size = 40, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=25),
        axis.title.y = element_text(size=20),
        axis.text.y = element_text(size=20),
        axis.text.x = element_text(size=15),
        legend.title = element_text(size=30),
        legend.text = element_text(size=12),
        strip.text = element_text(size=30,face="bold",colour = "darkblue"),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank()) ###
print(plot)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/SVID_imp/A123_model_(certain_nonzero)together_output.png", width=30, height=15)
# ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_nonzero/Final_chart/SVID_imp/B456_model_(certain_nonzero)together_output.png", width=25, height=15)

##### our method three dataset barplot ------------------------------------------------
# A123 <-read_rds("C:/Users/User/Desktop/2021_0520/192/certain_nonzero/Final_chart/code/A123_reviseheatmap_test_CAE_SVR.rds")
# A456 <-read_rds("C:/Users/User/Desktop/2021_0520/192/certain_nonzero/Final_chart/code/A456_reviseheatmap_test_CAE_SVR.rds")
# B456 <-read_rds("C:/Users/User/Desktop/2021_0520/192/certain_nonzero/Final_chart/code/B456_reviseheatmap_test_CAE_SVR.rds")

A123 <-read_rds("C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/code/A123_reviseheatmap_test_CAE_SVR.rds")
A456 <-read_rds("C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/code/A456_reviseheatmap_test_CAE_SVR.rds")
B456 <-read_rds("C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/code/B456_reviseheatmap_test_CAE_SVR.rds")
List <- list(A123,A456,B456)
set <- c()
for(i in 1:length(List))
{
  reviseheatmap <- List[[i]]
  heatmap<- reviseheatmap%>%
    mutate(rowMeans(reviseheatmap[,2:dim(reviseheatmap)[2]]))
  colnames(heatmap)[194]<-"sum"
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
  xlab("19 variable")+ylab("Mean of 192 feature MSE")+ggtitle("Evaluate SVID influence in three dataset")+
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
# ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_nonzero/CAE_SVR/Final_chart/SVID_imp/threedataset_output.png", width=20, height=10)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/SVID_imp/threedataset_output.png", width=20, height=10)

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

# saveRDS(dataset_rank,"C:/Users/User/Desktop/2021_0520/192/certain_nonzero/CAE_SVR/Final_chart/SVID_imp/threedataset_output.rds")
# write.csv(dataset_rank,"C:/Users/User/Desktop/2021_0520/192/certain_nonzero/CAE_SVR/Final_chart/SVID_imp/threedataset_output.csv")

saveRDS(dataset_rank,"C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/SVID_imp/threedataset_output.rds")
write.csv(dataset_rank,"C:/Users/User/Desktop/2021_0520/192/certain_zero/CAE_SVR/Final_chart/SVID_imp/threedataset_output.csv")




