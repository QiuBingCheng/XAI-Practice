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
xtrain<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/B456_xtrain.rds")
xtest<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/B456_xtest.rds")
trainy<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/B456_trainy.rds")
testy<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/B456_testy.rds")

#### normal CAE(192) -----------------------------------------------------------
## encoder
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
  layer_flatten()
encoder <- keras_model(enc_input,enc_output)
summary(encoder)

## decoder 
decoder <- encoder$output %>%
  layer_reshape(c(12,1,16),name="reshape")%>%
  layer_conv_2d(16, kernel_size=c(3,3), padding="same",name="decoder1") %>% 
  layer_activation_leaky_relu(name="leak11")%>%
  layer_upsampling_2d(c(3,3),name="up_samp1")%>%
  layer_conv_2d(32, kernel_size=c(3,3), padding="same",name="decoder2") %>% 
  layer_activation_leaky_relu(name="leak12")%>%
  layer_upsampling_2d(c(3,3),name="up_samp2")%>%
  layer_conv_2d(64, kernel_size=c(3,3), padding="valid",name="decoder3") %>% 
  layer_activation_leaky_relu(name="leak13")%>%
  layer_upsampling_2d(c(3,3),name="up_samp3")%>%
  layer_conv_2d(1, kernel_size=c(3,3), activation="sigmoid",padding="valid",name="autoencoder")
autoencoder <- keras_model(encoder$input,decoder)
summary(autoencoder)


callbacks = list(
  callback_model_checkpoint("checkpoints.h5"), callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1)) ### 再調整！！

autoencoder%>% compile(optimizer="RMSprop", loss="mse")
history <- autoencoder %>% fit(x= xtrain, y= xtrain,validation_data=list(x=xtest,y=xtest),batch_size=10,epochs=200,callback=callbacks)


## predict_loss &　reconstruct_loss
history_df <- as.data.frame(history)
train_loss <-data.frame(t(history_df %>%
                            filter(metric=="loss" & data=="training" & epoch==200)%>%
                            select(value)))
colnames(train_loss)<-"autoencoder_loss"

test_loss <-data.frame(t(history_df %>%
                           filter(metric=="loss" & data=="validation" & epoch==200)%>%
                           select(value)))
colnames(test_loss)<-c("autoencoder_loss")  

saveRDS(train_loss,"C:/Users/User/Desktop/2021_0502/model/192/Prediction/Results_loss/B456_trainloss_normalCAE.rds")
saveRDS(test_loss,"C:/Users/User/Desktop/2021_0502/model/192/Prediction/Results_loss/B456_testloss_normalCAE.rds")
save_model_hdf5(autoencoder,"C:/Users/User/Desktop/2021_0502/model/192/Prediction/B456_normalCAE.h5")

################################################
################################################
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
xtrain<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/B456_xtrain.rds")
xtest<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/B456_xtest.rds")
trainy<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/B456_trainy.rds")
testy<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/B456_testy.rds")

# model <- load_model_hdf5("C:/Users/User/Desktop/2021_0502/model/192/Prediction/A123_normalCAE(avg_pool).h5")
model <- load_model_hdf5("C:/Users/User/Desktop/2021_0502/model/192/Prediction/B456_normalCAE.h5")
summary(model)

layer_name<-"flatten"
encoder <- keras_model(inputs=model$input,outputs=get_layer(model,layer_name)$output)
summary(encoder)
test_feature  = encoder %>% predict(xtest) #   245  12
dim(test_feature)

saveRDS(test_feature,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/B456_testfeature_normalCAE.rds")
#### certain - baseline 0: -----------------------------------------------------
recon_list <- list() 
recon_feature <- list()
for(i in 1L:192L)
{
  tmp <- matrix(0:0, nrow = nrow(test_feature), ncol = ncol(test_feature))
  c <- test_feature[,i]
  tmp[,i]<-c
  # normalCAE reconstructed model --------------
  dec_input = layer_input(shape = 192)
  dec_reshape<- get_layer(model,name="reshape")
  dec1<- get_layer(model,name="decoder1")
  leak1<-get_layer(model,name="leak11")
  up_samp1<- get_layer(model,name="up_samp1")
  dec2<- get_layer(model,name="decoder2")
  leak2<-get_layer(model,name="leak12")
  up_samp2<- get_layer(model,name="up_samp2")
  dec3<- get_layer(model,name="decoder3")
  leak3<-get_layer(model,name="leak13")
  up_samp3<- get_layer(model,name="up_samp3")
  dec4<- get_layer(model,name="autoencoder")
  
  
  decoder<-keras_model(dec_input,dec4(up_samp3(leak3(dec3(up_samp2(leak2(dec2(up_samp1(leak1(dec1(dec_reshape(dec_input))))))))))))
  summary(decoder)
  reconstruct = decoder %>% predict(tmp)# 1981   316   19    1
  dim(reconstruct) #1981   19   19    1
  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
  }
  recon_feature[[i]]<- recon_list
}
saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_reconstruct_test_normalCAE.rds")

###　測試使用：
recon_list <- list()
recon_feature <- list()
for(i in 1L:192L)
{
  tmp <- matrix(0:0, nrow = nrow(test_feature), ncol = ncol(test_feature))
  
  # normalCAE reconstructed model --------------
  dec_input = layer_input(shape = 192)
  dec_reshape<- get_layer(model,name="reshape")
  dec1<- get_layer(model,name="decoder1")
  leak1<-get_layer(model,name="leak11")
  up_samp1<- get_layer(model,name="up_samp1")
  dec2<- get_layer(model,name="decoder2")
  leak2<-get_layer(model,name="leak12")
  up_samp2<- get_layer(model,name="up_samp2")
  dec3<- get_layer(model,name="decoder3")
  leak3<-get_layer(model,name="leak13")
  up_samp3<- get_layer(model,name="up_samp3")
  dec4<- get_layer(model,name="autoencoder")
  
  
  decoder<-keras_model(dec_input,dec4(up_samp3(leak3(dec3(up_samp2(leak2(dec2(up_samp1(leak1(dec1(dec_reshape(dec_input))))))))))))
  summary(decoder)
  reconstruct = decoder %>% predict(tmp)# 1981   316   19    1
  dim(reconstruct) #1981   19   19    1
  
  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
  }
  recon_feature[[i]]<- recon_list
}
saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_reconstruct_test_normalCAE(verify_0).rds")

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
recon_list <- list() 
recon_feature <- list()
for(i in 1L:192L)
{
  
  cal_set <- test_feature
  tmp <- matrix(0:0,nrow=nrow(cal_set),ncol=1)
  cal_set[,i]<-tmp
  
  
  # normalCAE reconstructed model --------------
  dec_input = layer_input(shape = 192)
  dec_reshape<- get_layer(model,name="reshape")
  dec1<- get_layer(model,name="decoder1")
  leak1<-get_layer(model,name="leak11")
  up_samp1<- get_layer(model,name="up_samp1")
  dec2<- get_layer(model,name="decoder2")
  leak2<-get_layer(model,name="leak12")
  up_samp2<- get_layer(model,name="up_samp2")
  dec3<- get_layer(model,name="decoder3")
  leak3<-get_layer(model,name="leak13")
  up_samp3<- get_layer(model,name="up_samp3")
  dec4<- get_layer(model,name="autoencoder")
  
  
  decoder<-keras_model(dec_input,dec4(up_samp3(leak3(dec3(up_samp2(leak2(dec2(up_samp1(leak1(dec1(dec_reshape(dec_input))))))))))))
  summary(decoder)
  reconstruct = decoder %>% predict(cal_set)# 1981   316   19    1
  dim(reconstruct) #1981   19   19    1
  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
  }
  recon_feature[[i]]<- recon_list
}

saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_reconstruct_test_normalCAE.rds")

###　測試使用：
recon_list <- list()
recon_feature <- list()
for(i in 1L:192L)
{
  
  # normalCAE reconstructed model --------------
  dec_input = layer_input(shape = 192)
  dec_reshape<- get_layer(model,name="reshape")
  dec1<- get_layer(model,name="decoder1")
  leak1<-get_layer(model,name="leak11")
  up_samp1<- get_layer(model,name="up_samp1")
  dec2<- get_layer(model,name="decoder2")
  leak2<-get_layer(model,name="leak12")
  up_samp2<- get_layer(model,name="up_samp2")
  dec3<- get_layer(model,name="decoder3")
  leak3<-get_layer(model,name="leak13")
  up_samp3<- get_layer(model,name="up_samp3")
  dec4<- get_layer(model,name="autoencoder")
  
  
  decoder<-keras_model(dec_input,dec4(up_samp3(leak3(dec3(up_samp2(leak2(dec2(up_samp1(leak1(dec1(dec_reshape(dec_input))))))))))))
  summary(decoder)
  reconstruct = decoder %>% predict(test_feature)# 1981   316   19    1
  dim(reconstruct) #1981   19   19    1
  
  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
  }
  recon_feature[[i]]<- recon_list
}
saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_reconstruct_test_normalCAE(verify_0).rds")


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
xtest<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/B456_xtest.rds")
xtest <- array_reshape(xtest,dim=c(dim(xtest)[1],dim(xtest)[2],dim(xtest)[3]*1)) 

recon_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_reconstruct_test_normalCAE.rds")
recon_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_reconstruct_test_normalCAE(verify_0).rds")

recon_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_reconstruct_test_normalCAE.rds")
recon_feature <- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_reconstruct_test_normalCAE(verify_0).rds")


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

# saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_Eachwafer_MSE_test_normalCAE.rds")
# saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_Eachwafer_MSE_test_normalCAE(verify0).rds")
# saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_Eachwafer_MSE_test_normalCAE.rds")
saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_Eachwafer_MSE_test_normalCAE(verify0).rds")


each_waferList<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_Eachwafer_MSE_test_normalCAE.rds")
each_waferList2<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_Eachwafer_MSE_test_normalCAE(verify0).rds")
each_waferList<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_Eachwafer_MSE_test_normalCAE.rds")
each_waferList2<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_Eachwafer_MSE_test_normalCAE(verify0).rds")

tmp <- matrix(0:0, nrow = 19, ncol = 192)
tmp_list<- list()
tmp <-data.table(tmp)

for(j in 1:dim(xtest)[1])
{
  tmp_list[[j]] <- each_waferList[[j]]-each_waferList2[[j]]
  tmp<-tmp+tmp_list[[j]]
  
}
avg_wafer_error <- tmp/length(tmp_list)
saveRDS(avg_wafer_error,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_avg_MSE(actual-null)_test_normalCAE.rds")
saveRDS(avg_wafer_error,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_avg_MSE(actual-null)_test_normalCAE.rds")

### visualize---------------------------------------------------------------------------------------
## origin heatmap:
avg_wafer_error<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_avg_MSE(actual-null)_test_normalCAE.rds")
avg_wafer_error<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_avg_MSE(actual-null)_test_normalCAE.rds")

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
  # ggtitle("B456 normalCAE: avg_MSE of each SVID (certain_nonezero-baseline)") +
  ggtitle("B456 normalCAE:avg_MSE of each SVID (certain_zero-baseline)") +
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


# ggsave(plot, file="C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/Reconstruction_path/origin/B456_normalCAE.png", width=45, height=20)
ggsave(plot, file="C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/Reconstruction_path/origin/B456_normalCAE.png", width=45, height=20)



### 2021/05/14 each SVID range
avg_wafer_error<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/A123_avg_MSE(actual-null)_test_normalCAE.rds")
avg_wafer_error<- read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/A123_avg_MSE(actual-null)_test_normalCAE.rds")

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
  # ggtitle("A123 normalCAE: Min-max avg_MSE of each SVID (certain_nonezero-baseline)") +
  ggtitle("A123 normalCAE: Min-max avg_MSE of each SVID (certain_zero-baseline)") +
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
# ggsave(plot, file="C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/Reconstruction_path/origin/min-max_B456_normalCAE.png", width=45, height=20)
# ggsave(plot, file="C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/Reconstruction_path/origin/min-max_B456_normalCAE.png", width=45, height=20)
# 
# saveRDS(Normalize,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_minmax_avg_MSE_test_normalCAE.rds")
# saveRDS(Normalize,"C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_minmax_avg_MSE_test_normalCAE.rds")

ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_nonzero/normalCAE/Final_chart/min-max_A123_normalCAE.png", width=45, height=20)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_zero/normalCAE/Final_chart/min-max_A123_normalCAE.png", width=45, height=20)

saveRDS(Normalize,"C:/Users/User/Desktop/2021_0520/192/certain_nonzero/normalCAE/Final_chart/A123_minmax_avg_MSE_test_normalCAE.rds")
saveRDS(Normalize,"C:/Users/User/Desktop/2021_0520/192/certain_zero/normalCAE/Final_chart/A123_minmax_avg_MSE_test_normalCAE.rds")



########## (5/20 補 SVID mean MSE of feature barplot) ##########
reviseheatmap <-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_minmax_avg_MSE_test_normalCAE.rds")
reviseheatmap <-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_minmax_avg_MSE_test_normalCAE.rds")

revise_heatmap<- round(reviseheatmap,digits = 4)
SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
value<- cbind(SVID,data.table(revise_heatmap))


heatmap <- data.table(value)%>%
  mutate(rowMeans(value[,2:dim(value)[2]]))
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
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_nonzero/normalCAE/Final_chart/SVID_imp/B456_SVID_imp.png", width=15, height=10)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_zero/normalCAE/Final_chart/SVID_imp/B456_SVID_imp.png", width=15, height=10)


##### our method three dataset barplot ------------------------------------------------
A123 <-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/A123_minmax_avg_MSE_test_normalCAE.rds")
A456 <-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/A456_minmax_avg_MSE_test_normalCAE.rds")
B456 <-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_nonezero/B456_minmax_avg_MSE_test_normalCAE.rds")

A123 <-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/A123_minmax_avg_MSE_test_normalCAE.rds")
A456 <-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/A456_minmax_avg_MSE_test_normalCAE.rds")
B456 <-read_rds("C:/Users/User/Desktop/2021_0514/model/192/Prediction/Explainable/certain_zero/B456_minmax_avg_MSE_test_normalCAE.rds")

A123<- round(A123,digits = 4)
SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
A123<- cbind(SVID,data.table(A123))

A456<- round(A456,digits = 4)
SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
A456<- cbind(SVID,data.table(A456))

B456<- round(B456,digits = 4)
SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
B456<- cbind(SVID,data.table(B456))

List <- list(data.frame(A123),data.frame(A456),data.frame(B456))
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
# ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_nonzero/normalCAE/Final_chart/SVID_imp/threedataset_output.png", width=20, height=10)
ggsave(plot, file="C:/Users/User/Desktop/2021_0520/192/certain_zero/normalCAE/Final_chart/SVID_imp/threedataset_output.png", width=20, height=10)

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

saveRDS(dataset_rank,"C:/Users/User/Desktop/2021_0520/192/certain_nonzero/normalCAE/Final_chart/SVID_imp/threedataset_output.rds")
write.csv(dataset_rank,"C:/Users/User/Desktop/2021_0520/192/certain_nonzero/normalCAE/Final_chart/SVID_imp/threedataset_output.csv")

saveRDS(dataset_rank,"C:/Users/User/Desktop/2021_0520/192/certain_zero/normalCAE/Final_chart/SVID_imp/threedataset_output.rds")
write.csv(dataset_rank,"C:/Users/User/Desktop/2021_0520/192/certain_zero/normalCAE/Final_chart/SVID_imp/threedataset_output.csv")




