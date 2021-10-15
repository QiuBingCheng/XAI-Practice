##########################################################################################################
########################################## model explainability ##########################################
##################################  Decomposed importance ###########################################
############################ decoder module ###############################
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


### custom loss:
eplison <- tf$constant(0.5)

# Margin term in loss
svr_loss <- function(y_true,y_pred)
{
  
  tf$reduce_mean(tf$maximum(0.,tf$subtract(tf$abs(tf$subtract(y_pred,y_true)),eplison)))
  
}
with_custom_object_scope(c("svr_loss" = svr_loss), {
  
  model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/A123_CAE_SVR.h5")
  
  
})
summary(model)


# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/A123_CAE_SVR.h5")
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/A123_CAE_ANN.h5")
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/B456_CAE_normalCAE.h5")
model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/A123_CNN.h5")
summary(model)

layer_name<-"flatten"
encoder <- keras_model(inputs=model$input,outputs=get_layer(model,layer_name)$output)
summary(encoder)
test_feature  = encoder %>% predict(xtest) #   245  12

# saveRDS(test_feature,"C:/Users/User/Desktop/2021_0605/model/Explainable/B456_testfeature_normalCAE.rds")
# saveRDS(test_feature,"C:/Users/User/Desktop/2021_0605/model/Explainable/A123_testfeature_CNN.rds")

test_feature <- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/A123_testfeature_CAE_ANN.rds")


#################### certain 0 - baseline: ------------------------------------
# certain experiment:
recon_list <- list() 
recon_feature <- list()
for(i in 1L:192L)
{
  
  cal_set <- test_feature
  tmp <- matrix(0:0,nrow=nrow(cal_set),ncol=1)
  cal_set[,i]<-tmp
  
  
  # CAE_ANN reconstructed model -----------------------------------------
  dec_input = layer_input(shape = 192)
  leak1<-get_layer(model,name="leak7")
  dec_reshape<- get_layer(model,name="reshape")
  dec1<- get_layer(model,name="decoder1")
  leak2<-get_layer(model,name="leak8")
  up_samp1<- get_layer(model,name="up_samp1")
  dec2<- get_layer(model,name="decoder2")
  leak3<-get_layer(model,name="leak9")
  up_samp2<- get_layer(model,name="up_samp2")
  dec3<- get_layer(model,name="decoder3")
  leak4<-get_layer(model,name="leak10")
  up_samp3<- get_layer(model,name="up_samp3")
  dec4<- get_layer(model,name="autoencoder")
  decoder<-keras_model(dec_input,dec4(up_samp3(leak4(dec3(up_samp2(leak3(dec2(up_samp1(leak2(dec1(dec_reshape(leak1(dec_input)))))))))))))
  summary(decoder)
  #CAE SVR -----------------------------------
  ## decoder 
  # dec_input = layer_input(shape = 192)
  # leak1<-get_layer(model,name="leak5")
  # dec_reshape<- get_layer(model,name="reshape")
  # dec1<- get_layer(model,name="decoder1")
  # leak2<-get_layer(model,name="leak6")
  # up_samp1<- get_layer(model,name="up_samp1")
  # dec2<- get_layer(model,name="decoder2")
  # leak3<-get_layer(model,name="leak7")
  # up_samp2<- get_layer(model,name="up_samp2")
  # dec3<- get_layer(model,name="decoder3")
  # leak4<-get_layer(model,name="leak8")
  # up_samp3<- get_layer(model,name="up_samp3")
  # dec4<- get_layer(model,name="autoencoder")
  # decoder<-keras_model(dec_input,dec4(up_samp3(leak4(dec3(up_samp2(leak3(dec2(up_samp1(leak2(dec1(dec_reshape(leak1(dec_input)))))))))))))
  # 
  #normalCAE ---------------------------------
  ## decoder 
  # dec_input = layer_input(shape = 192)
  # dec_reshape<- get_layer(model,name="reshape")
  # dec1<- get_layer(model,name="decoder1")
  # leak2<-get_layer(model,name="leak4")
  # up_samp1<- get_layer(model,name="up_samp1")
  # dec2<- get_layer(model,name="decoder2")
  # leak3<-get_layer(model,name="leak5")
  # up_samp2<- get_layer(model,name="up_samp2")
  # dec3<- get_layer(model,name="decoder3")
  # leak4<-get_layer(model,name="leak6")
  # up_samp3<- get_layer(model,name="up_samp3")
  # dec4<- get_layer(model,name="autoencoder")
  # 
  # decoder<-keras_model(dec_input,dec4(up_samp3(leak4(dec3(up_samp2(leak3(dec2(up_samp1(leak2(dec1(dec_reshape(dec_input))))))))))))
  # 
  summary(decoder)
  reconstruct = decoder %>% predict(cal_set)# 1981   316   19    1
  dim(reconstruct) #1981   19   19    1
  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
  }
  recon_feature[[i]]<- recon_list
}

saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/decoder/tmp/A123_reconstruct_test_CAE_ANN.rds")

# baseline experiment：
recon_list <- list()
recon_feature <- list()
for(i in 1L:192L)
{
  
  # CAE_ANN decoder model -----------------------------------------
  dec_input = layer_input(shape = 192)
  leak1<-get_layer(model,name="leak7")
  dec_reshape<- get_layer(model,name="reshape")
  dec1<- get_layer(model,name="decoder1")
  leak2<-get_layer(model,name="leak8")
  up_samp1<- get_layer(model,name="up_samp1")
  dec2<- get_layer(model,name="decoder2")
  leak3<-get_layer(model,name="leak9")
  up_samp2<- get_layer(model,name="up_samp2")
  dec3<- get_layer(model,name="decoder3")
  leak4<-get_layer(model,name="leak10")
  up_samp3<- get_layer(model,name="up_samp3")
  dec4<- get_layer(model,name="autoencoder")
  decoder<-keras_model(dec_input,dec4(up_samp3(leak4(dec3(up_samp2(leak3(dec2(up_samp1(leak2(dec1(dec_reshape(leak1(dec_input)))))))))))))
  # 
  #CAE SVR -----------------------------------
  ## decoder 
  # dec_input = layer_input(shape = 192)
  # leak1<-get_layer(model,name="leak5")
  # dec_reshape<- get_layer(model,name="reshape")
  # dec1<- get_layer(model,name="decoder1")
  # leak2<-get_layer(model,name="leak6")
  # up_samp1<- get_layer(model,name="up_samp1")
  # dec2<- get_layer(model,name="decoder2")
  # leak3<-get_layer(model,name="leak7")
  # up_samp2<- get_layer(model,name="up_samp2")
  # dec3<- get_layer(model,name="decoder3")
  # leak4<-get_layer(model,name="leak8")
  # up_samp3<- get_layer(model,name="up_samp3")
  # dec4<- get_layer(model,name="autoencoder")
  # 
  # 
  # decoder<-keras_model(dec_input,dec4(up_samp3(leak4(dec3(up_samp2(leak3(dec2(up_samp1(leak2(dec1(dec_reshape(leak1(dec_input)))))))))))))
  # 
  #normalCAE ---------------------------------
  ## decoder 
  # dec_input = layer_input(shape = 192)
  # dec_reshape<- get_layer(model,name="reshape")
  # dec1<- get_layer(model,name="decoder1")
  # leak2<-get_layer(model,name="leak4")
  # up_samp1<- get_layer(model,name="up_samp1")
  # dec2<- get_layer(model,name="decoder2")
  # leak3<-get_layer(model,name="leak5")
  # up_samp2<- get_layer(model,name="up_samp2")
  # dec3<- get_layer(model,name="decoder3")
  # leak4<-get_layer(model,name="leak6")
  # up_samp3<- get_layer(model,name="up_samp3")
  # dec4<- get_layer(model,name="autoencoder")
  # decoder<-keras_model(dec_input,dec4(up_samp3(leak4(dec3(up_samp2(leak3(dec2(up_samp1(leak2(dec1(dec_reshape(dec_input))))))))))))
  summary(decoder)
  reconstruct = decoder %>% predict(test_feature)# 1981   316   19    1
  dim(reconstruct) #1981   19   19    1
  
  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
  }
  recon_feature[[i]]<- recon_list
}

saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/decoder/tmp/A123_reconstruct_test_CAE_ANN(verify_0).rds")


#### Actual_corr - Reconstruct_corr ----------------------
xtest<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/A123_xtest.rds")
xtest <- array_reshape(xtest,dim=c(dim(xtest)[1],dim(xtest)[2],dim(xtest)[3]*1)) 

# recon_feature <- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/decoder/tmp/A123_reconstruct_test_CAE_ANN.rds")
# recon_feature <- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/decoder/tmp/A123_reconstruct_test_CAE_ANN(verify_0).rds")
recon_feature <- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/decoder/tmp/A123_reconstruct_test_CAE_ANN.rds")
recon_feature <- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/decoder/tmp/A123_reconstruct_test_CAE_ANN(verify_0).rds")
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

# saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/decoder/tmp/B456_Eachwafer_MSE_test_normalCAE.rds")
# saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/decoder/tmp/B456_Eachwafer_MSE_test_normalCAE(verify0).rds")
# saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/decoder/tmp/A123_Eachwafer_MSE_test_CAE_ANN.rds")
# saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/decoder/tmp/A123_Eachwafer_MSE_test_CAE_ANN(verify0).rds")

xtest<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/B456_xtest.rds")
xtest <- array_reshape(xtest,dim=c(dim(xtest)[1],dim(xtest)[2],dim(xtest)[3]*1)) 
each_waferList<- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/decoder/tmp/B456_Eachwafer_MSE_test_normalCAE.rds")
each_waferList2<- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/decoder/tmp/B456_Eachwafer_MSE_test_normalCAE(verify0).rds")

# each_waferList<- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/decoder/tmp/A123_Eachwafer_MSE_test_CAE_ANN.rds")
# each_waferList2<- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/decoder/tmp/A123_Eachwafer_MSE_test_CAE_ANN(verify0).rds")

tmp <- matrix(0:0, nrow = 19, ncol = 192)
tmp_list<- list()
tmp <-data.table(tmp)

for(j in 1:dim(xtest)[1])
{
  tmp_list[[j]] <- each_waferList[[j]]-each_waferList2[[j]]
  tmp<-tmp+tmp_list[[j]]
  
}
avg_wafer_error <- tmp/length(tmp_list)
saveRDS(avg_wafer_error,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/decoder/tmp/revise/B456_avg_MSE(actual-null)_test_normalCAE.rds")
# saveRDS(avg_wafer_error,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/decoder/tmp/A123_avg_MSE(actual-null)_test_CAE_ANN.rds")


#######################################################
######### decoder heatmap 
## origin heatmap:
avg_wafer_error<- read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/decoder/tmp/A456_avg_MSE(actual-null)_test_CAE_ANN.rds")

##nonzero:
# avg_wafer_error <- -avg_wafer_error
maxmin <- function(x) (x - min(x))/(max(x)-min(x))
avgMSE_Normalize <- apply(avg_wafer_error, 1, maxmin)

feature<- factor(paste0('f', 1:192),levels =paste0('f', 1:192))
SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
colnames(avgMSE_Normalize)<-SVID
value<- cbind(feature,data.table(avgMSE_Normalize))

value<- melt(value)

colnames(value)<-c("feature","SVID","error")
library(ggplot2)

# Plot 
mid<-(min(value$error)+max(value$error))/2
plot <- ggplot(data = value, aes(x = SVID, y = feature)) + 
  geom_tile(aes(fill = error), color = "white", size = 1) +
  scale_fill_gradient2(
    low = 'steelblue', mid = 'white', high = 'red',
    midpoint = mid, guide = 'colourbar', aesthetics = 'fill',limit=c(min(value$error),max(value$error)))+
  xlab("19 SVID ") + ylab("192 hidden feature")+
  theme_grey(base_size = 10) + 
  ggtitle("B456: Average MSE of reconstruction heatmap") +
  theme(axis.ticks = element_blank(), 
        panel.background = element_blank(), 
        plot.title = element_text(size=30,hjust = 0.5,face="bold"),legend.position="right",legend.direction="vertical",
        axis.text.x= element_text(size = 12,face="bold"),
        legend.text = element_text(size=17),legend.title = element_text(size=20),
        axis.title.y = element_text(size=25,face="bold"),
        axis.title.x = element_blank())+
  guides(fill = guide_colorbar(barwidth = 1, barheight =7,title.position = "top", title.hjust = 0.5))
print(plot)


##########################################################################################################
########################################## predictor module ##############################################


### custom loss:
eplison <- tf$constant(0.5)

# Margin term in loss
svr_loss <- function(y_true,y_pred)
{
  
  tf$reduce_mean(tf$maximum(0.,tf$subtract(tf$abs(tf$subtract(y_pred,y_true)),eplison)))
  
}
with_custom_object_scope(c("svr_loss" = svr_loss), {
  
  model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/A123_CAE_SVR.h5")
  
  
})
summary(model)


model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/B456_CNN.h5")
summary(model)


test_feature <- read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/B456_testfeature_CNN.rds")


### certain0 - baseline --------------------------------------------
# certain experiment:
model <- load_model_hdf5("C:/Users/User/Desktop/2021_0605/model/B456_CNN.h5")
### custom loss:
# eplison <- tf$constant(0.5)
# # Margin term in loss
# svr_loss <- function(y_true,y_pred)
# {
#   
#   tf$reduce_mean(tf$maximum(0.,tf$subtract(tf$abs(tf$subtract(y_pred,y_true)),eplison)))
#   
# }
# with_custom_object_scope(c("svr_loss" = svr_loss), {
#   
#   model <- load_model_hdf5("C:/Users/User/Desktop/2021_0605/model/A123_CAE_SVR.h5")
#   
# })
# summary(model)
test_feature <- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/B456_testfeature_CNN.rds")
pred_list <- list()
pred_feature <- list()
for(i in 1L:192L)
{
  cal_set <- test_feature
  tmp <- matrix(0:0,nrow=nrow(cal_set),ncol=1)
  cal_set[,i]<-tmp
  
  ## CAE_ANN predictor model -------------------------------------
  # pred_input = layer_input(shape=192)
  # dense1<-get_layer(model,name="dec_class1")
  # leak1 <-get_layer(model,name="leak4")
  # dense2<-get_layer(model,name="dec_class2")
  # leak2 <-get_layer(model,name="leak5")
  # dense3<-get_layer(model,name="dec_class3")
  # leak3 <-get_layer(model,name="leak6")
  # dense4<-get_layer(model,name="predict")
  # predictor<- keras_model(pred_input,dense4(leak3(dense3(leak2(dense2(leak1(dense1(pred_input))))))))
  
  ## CAE_SVR predictor model -------------------------------------
  # pred_input = layer_input(shape=192)
  # dense1<-get_layer(model,name="dec_pred")
  # leak1 <-get_layer(model,name="leak4")
  # dense2<-get_layer(model,name="predict")
  # predictor<- keras_model(pred_input,dense2(leak1(dense1(pred_input))))
  
  ##CNN predictor -----------------------------------------------
  pred_input = layer_input(shape=192)
  dense1<-get_layer(model,name="dec_class1")
  leak1 <-get_layer(model,name="leak4")
  dense2<-get_layer(model,name="dec_class2")
  leak2 <-get_layer(model,name="leak5")
  dense3<-get_layer(model,name="dec_class3")
  leak3 <-get_layer(model,name="leak6")
  dense4<-get_layer(model,name="predict")
  predictor<- keras_model(pred_input,dense4(leak3(dense3(leak2(dense2(leak1(dense1(pred_input))))))))
  summary(predictor)
  prediction = predictor %>% predict(cal_set)
  
  for(j in 1: dim(prediction)[1])
  {
    pred_list[[j]]<- prediction[j,]
  }
  pred_feature[[i]]<- pred_list
}
saveRDS(pred_feature,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/B456_prediction_test_CNN.rds")

### baseline experiment
pred_list <- list()
pred_feature <- list()
for(i in 1L:192L)
{
  ## CAE_ANN classifier model -------------------------------------
  # pred_input = layer_input(shape=192)
  # dense1<-get_layer(model,name="dec_class1")
  # leak1 <-get_layer(model,name="leak4")
  # dense2<-get_layer(model,name="dec_class2")
  # leak2 <-get_layer(model,name="leak5")
  # dense3<-get_layer(model,name="dec_class3")
  # leak3 <-get_layer(model,name="leak6")
  # dense4<-get_layer(model,name="predict")
  # predictor<- keras_model(pred_input,dense4(leak3(dense3(leak2(dense2(leak1(dense1(pred_input))))))))
  
  ## CAE_SVR classifier model -------------------------------------
  # pred_input = layer_input(shape=192)
  # dense1<-get_layer(model,name="dec_pred")
  # leak1 <-get_layer(model,name="leak4")
  # dense2<-get_layer(model,name="predict")
  # predictor<- keras_model(pred_input,dense2(leak1(dense1(pred_input))))
  
  ##CNN classifier -----------------------------------------------
  pred_input = layer_input(shape=192)
  dense1<-get_layer(model,name="dec_class1")
  leak1 <-get_layer(model,name="leak4")
  dense2<-get_layer(model,name="dec_class2")
  leak2 <-get_layer(model,name="leak5")
  dense3<-get_layer(model,name="dec_class3")
  leak3 <-get_layer(model,name="leak6")
  dense4<-get_layer(model,name="predict")
  predictor<- keras_model(pred_input,dense4(leak3(dense3(leak2(dense2(leak1(dense1(pred_input))))))))

  summary(predictor)
  prediction = predictor %>% predict(test_feature)# 1981   316   19    1
  for(j in 1: dim(prediction)[1])
  {
    pred_list[[j]]<- prediction[j,]
  }
  pred_feature[[i]]<- pred_list
}

saveRDS(pred_feature,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/B456_prediction_test_CNN(verify_0).rds")

#Actual_corr - Reconstruct_corr
ytest<-read_rds("C:/Users/User/Desktop/2021_0502/data/2021_0415/dataset/LR/code/B456_testy.rds")
# pred_feature <- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/predict/tmp/B456_prediction_test_CNN.rds")
# pred_feature <- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/predict/tmp/B456_prediction_test_CNN(verify_0).rds")
# pred_feature <- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/B456_prediction_test_CNN.rds")
pred_feature <- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/B456_prediction_test_CNN(verify_0).rds")

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


saveRDS(SSE_matrix,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/predict/tmp/revise/B456_errorSVID_SSEmatrix_test_CNN.rds")
saveRDS(corr_errorSVID_sum2,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/predict/tmp/revise/B456_errorSVID_MSE_test_CNN.rds")

# saveRDS(SSE_matrix,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/predict/tmp/B456_errorSVID_SSEmatrix_test_CNN(verify0).rds")
# saveRDS(corr_errorSVID_sum2,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/predict/tmp/B456_errorSVID_MSE_test_CNN(verify0).rds")

# saveRDS(SSE_matrix,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/B456_errorSVID_SSEmatrix_test_CNN.rds")
# saveRDS(corr_errorSVID_sum2,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/B456_errorSVID_MSE_test_CNN.rds")

# saveRDS(SSE_matrix,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/B456_errorSVID_SSEmatrix_test_CNN(verify0).rds")
# saveRDS(corr_errorSVID_sum2,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/B456_errorSVID_MSE_test_CNN(verify0).rds")


origin_errorSVID_MSE<- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/predict/tmp/A123_errorSVID_MSE_test_CNN.rds")
origin_errorSVID_MSE2<- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/A123_errorSVID_MSE_test_CNN(verify0).rds")
origin_errorSVID_SSEmatrix<- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/predict/tmp/A123_errorSVID_SSEmatrix_test_CNN.rds")
origin_errorSVID_SSEmatrix2<- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/A123_errorSVID_SSEmatrix_test_CNN(verify0).rds")

# origin_errorSVID_MSE<- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/A456_errorSVID_MSE_test_CNN.rds")
# origin_errorSVID_MSE2<- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/A456_errorSVID_MSE_test_CNN(verify0).rds")
# origin_errorSVID_SSEmatrix<- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/A456_errorSVID_SSEmatrix_test_CNN.rds")
# origin_errorSVID_SSEmatrix2<- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/A456_errorSVID_SSEmatrix_test_CNN(verify0).rds")

certain_MSE <- data.frame(unlist(origin_errorSVID_MSE))
baseline_MSE <- data.frame(unlist(origin_errorSVID_MSE2))
pred_performance<- certain_MSE-baseline_MSE

SSE_performance<-origin_errorSVID_SSEmatrix-origin_errorSVID_SSEmatrix2
colnames(SSE_performance) <- paste0('X', 1:(ncol(SSE_performance)))

saveRDS(SSE_performance,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/predict/tmp/revise/A123_errorSVID_SSEpreformance_test_CNN.rds")
saveRDS(pred_performance,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/predict/tmp/revise/A123_errorSVID_MSEperformance_test_CNN.rds")

# saveRDS(SSE_performance,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/A123_errorSVID_SSEpreformance_test_CNN.rds")
# saveRDS(pred_performance,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/A123_errorSVID_MSEperformance_test_CNN.rds")


# SSE by each feature MSE -------------------
SSE_performance<- readRDS("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/predict/tmp/revise/A123_errorSVID_SSEpreformance_test_CNN.rds")
# SSE_performance<- readRDS("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/A456_errorSVID_SSEpreformance_test_CNN.rds")


value <- data.table(SSE_performance)
# value<- round(value,digits = 4)
mean2 <- data.table(colMeans(value))
y_value2<- factor(seq(1,dim(value)[2],by=1),levels =seq(1,dim(value)[2],by=1))
cal2 <- cbind(y_value2,mean2)
colnames(cal2)<-c("wafer","mean")

saveRDS(cal2,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/predict/tmp/revise/A123_avgSSE_test_CNN.rds")
# saveRDS(cal2,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/tmp/A456_avgSSE_test_CNN.rds")

###### 
avgSSE<-read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/predict/tmp/A123_avgSSE_test_CAE_ANN.rds")


maxmin <- function(x) (x - min(x))/(max(x)-min(x))
avgSSE_Normalize <- apply(avgSSE[,2], 2, maxmin)
avgSSE_Normalize<- round(avgSSE_Normalize,digits = 3)

value<- cbind(data.table(factor(seq(1,dim(avgSSE_Normalize)[1],by=1))),avgSSE_Normalize)
colnames(value)<-c("number","mean")
plot <- ggplot(value, aes(x =  number, y = mean)) +
  geom_bar(stat = "identity",fill="#739fc7",width = 0.5) +
  xlab("192 feature")+ylab("min max normalize of prediction MSE")+
  ggtitle("B456:MSE proportion of 192 hidden feature ") +
  theme_minimal()+coord_flip()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),axis.line.y = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 20,face="bold"), 
        axis.title = element_text(size = 20,face="bold"), 
        plot.title = element_text(size = 25, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=15,face="bold"),
        axis.title.y = element_text(size=15,face="bold"),
        axis.text.y = element_text(size=3),
        axis.text.x = element_text(size=10,face="bold"),
        legend.title = element_text(size=12),
        legend.text = element_text(size=12),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.y = element_blank()) 

print(plot)
ggsave(plot, file="C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/predict/tmp/revise/chart/B456_certain_nonzero_minmax_weight_CNN.png", width=10, height=10)
# ggsave(plot, file="C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/predict/chart/A456_certain_zero_minmax_weight_CNN.png", width=10, height=10)

##########################################################################################################
######################################### encoder heatmap ################################################
### custom loss:
eplison <- tf$constant(0.5)

# Margin term in loss
svr_loss <- function(y_true,y_pred)
{
  
  tf$reduce_mean(tf$maximum(0.,tf$subtract(tf$abs(tf$subtract(y_pred,y_true)),eplison)))
  
}
with_custom_object_scope(c("svr_loss" = svr_loss), {
  
  model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/A123_CAE_SVR.h5")
  
  
})
summary(model)


# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/A123_CAE_SVR.h5")
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/A123_CAE_ANN.h5")
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/B456_CAE_normalCAE.h5")
model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/A123_CNN.h5")
summary(model)



null <-c()
for(i in 1:19)
{
  
  ## certain zero - baseline origin:
  xtest<-read_rds(file = paste0("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/encoder/input_experiment/certain_zero/A123_xtest",i,".rds"))
  # test_feature <- read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/A123_testfeature_CAE_ANN.rds")
  # test_feature <- read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/B456_testfeature_CAE_SVR.rds")
  test_feature <- read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/A123_testfeature_normalCAE.rds")
  # test_feature <- read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/B456_testfeature_CNN.rds")
  layer_name<-"flatten"
  encoder <- keras_model(inputs=model$input,outputs=get_layer(model,layer_name)$output)
  summary(encoder)
  new_feature  = encoder %>% predict(xtest)
  tmp <- (new_feature - test_feature)^2
  tmp <- colMeans(tmp)
  null <- rbind(null,tmp)
  
  
  
}
dim(null)

maxmin <- function(x) (x - min(x))/(max(x)-min(x))
Normalize <- apply(null, 2, maxmin)#(byfeature)
for(i in 1:dim(Normalize)[2])
{
  if(Normalize[1,i]== "NaN" )
  {
    Normalize[,i]<-0
  }
}

rownames(Normalize)<-seq(1,dim(Normalize)[1],by=1)
heatmap <- data.frame(Normalize)
# saveRDS(heatmap,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/encoder/tmp/B456_CNN.rds")
# saveRDS(heatmap,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/encoder/tmp/A123_CAEANN.rds")
# saveRDS(heatmap,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/encoder/tmp/B456_CAESVR.rds")
# saveRDS(heatmap,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/encoder/tmp/A123_normalCAE.rds")


### visualize encoder heatmap ----------------------------------------
## heatmap 望大、紅至藍色
avg_wafer_error<-read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/encoder/tmp/A456_normalCAE.rds")

SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
feature<- factor(paste0('f', 1:192),levels =paste0('f', 1:192))
colnames(avg_wafer_error)<-feature
value<- cbind(SVID,data.table(avg_wafer_error))
value<- melt(value)
colnames(value)<-c("SVID","feature","error")
library(ggplot2)

# Plot 
mid<-(min(value$error)+max(value$error))/2
plot <- ggplot(data = value, aes(x = feature, y = SVID)) + 
  geom_tile(aes(fill = error), color = "white", size = 1) +
  scale_fill_gradient2(
    low = 'steelblue', mid = 'white', high = 'red',
    midpoint = mid, guide = 'colourbar', aesthetics = 'fill',limit=c(min(value$error),max(value$error)))+
  xlab("192 hidden feature") + ylab("19 SVID")+
  theme_grey(base_size = 10) + 
  ggtitle("B456: MSE of encoder heatmap") +
  theme(axis.ticks = element_blank(), 
        panel.background = element_blank(), 
        plot.title = element_text(size=30,hjust = 0.5,face="bold"),legend.position="right",legend.direction="vertical",
        axis.text.y= element_text(size = 17,face="bold"),
        legend.text = element_text(size=17),legend.title = element_text(size=20),
        axis.text.x=element_blank(),
        axis.title.x = element_text(size=17,face="bold"),
        axis.title.y = element_blank())+
  guides(fill = guide_colorbar(barwidth = 1, barheight =7,title.position = "top", title.hjust = 0.5))
print(plot)



####################################################################################################
############################### Combination of encoder and regressor ###############################
## certain zero:
#encoder:(已經望大且min max)

#encoder normalize by feature:
avg_wafer_error<-read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/encoder/tmp/B456_CNN.rds")

#predict:
avgSSE<-read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/predict/tmp/B456_avgSSE_test_CNN.rds")



maxmin <- function(x) (x - min(x))/(max(x)-min(x))
avgSSE_Normalize <- apply(avgSSE[,2], 2, maxmin)
avgSSE_Normalize<- round(avgSSE_Normalize,digits = 3)


enc_matrix <- as.matrix(avg_wafer_error)
pred_matrix <- as.matrix(avgSSE_Normalize)
tmp <- enc_matrix%*%pred_matrix

# tmp <- tmp/19
maxmin <- function(x) (x - min(x))/(max(x)-min(x))
tmp <- apply(tmp, 2, maxmin)

SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
pred_overall <- cbind(SVID,data.table(tmp))


plot <- ggplot(pred_overall, aes(x =  SVID, y = mean)) +
  geom_bar(stat = "identity",fill="#739fc7",width = 0.5) +
  xlab("19 SVID")+ylab("encoder*prediction MSE")+
  geom_text(aes(label=round(mean,digits=3)), vjust=-0.3, size=4)+
  ggtitle("B456: combination of encoder and regressor in CNN model") +
  theme_minimal()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 12,face="bold"), 
        axis.title = element_text(size = 12,face="bold"), 
        plot.title = element_text(size = 30, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=20,face="bold"),
        axis.title.y = element_text(size=20,face="bold"),
        axis.text.y = element_text(size=10),
        axis.text.x = element_text(size=15,face="bold"),
        legend.title = element_text(size=12),
        legend.text = element_text(size=12),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank()) 
print(plot)

ggsave(plot, file="C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/combine/encoder_minmaxbyfeature/pred/B456_SVID_imp_CNN.png", width=15, height=10)

####################################################################################################
############################### Combination of encoder and decoder ###############################
##certain zero
# encoder:
encoder <-read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/encoder/tmp/B456_normalCAE.rds")
#decoder:
decoder<- read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/decoder/tmp/B456_avg_MSE(actual-null)_test_normalCAE.rds")


maxmin <- function(x) (x - min(x))/(max(x)-min(x))
decoder <- apply(decoder, 1, maxmin)

enc_matrix <- as.matrix(encoder)
dec_matrix <- as.matrix(decoder)
tmp <- enc_matrix%*%dec_matrix

tmp <- tmp/192
tmp<- round(tmp,digits = 4)
SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
colnames(tmp)<-SVID
exp<- factor(paste0('V', 1:19),levels =paste0('V', 1:19))
value<- cbind(exp,data.table(tmp))
value<- melt(value)
colnames(value)<-c("experiment","SVID","error")
library(ggplot2)


# Plot 
mid<-(min(value$error)+max(value$error))/2
plot <- ggplot(data = value, aes(x = SVID, y = experiment)) + 
  geom_tile(aes(fill = error), color = "white", size = 1) +
  scale_fill_gradient2(
    low = 'steelblue', mid = 'white', high = 'red',
    midpoint = mid, guide = 'colourbar', aesthetics = 'fill',limit=c(min(value$error),max(value$error)))+
  xlab("SVID") + ylab("experiment")+
  theme_grey(base_size = 10) + 
  ggtitle("B456: overall SVID reconstruction influence in normalCAE model ") +
  theme(axis.ticks = element_blank(), 
        panel.background = element_blank(), 
        plot.title = element_text(size=20,hjust = 0.5,face="bold"),legend.position="right",legend.direction="vertical",
        axis.text.x= element_text(size = 12,face="bold"),
        axis.text.y= element_text(size = 12,face="bold"),
        legend.text = element_text(size=12),legend.title = element_text(size=20),
        axis.title.x = element_text(size=15,face="bold"),axis.title.y = element_text(size=15,face="bold"))+
  guides(fill = guide_colorbar(barwidth = 1, barheight =7,title.position = "top", title.hjust = 0.5))+
  geom_text(aes(label = round(error, 3)),size=3.5) 
print(plot)

ggsave(plot, file="C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/combine/encoder_minmaxbyfeature/rec/B456_normalCAE.png", width=15, height=10)

#################################################################################
################# Combination of encoder and regressor ########################
################### Proposed model and benchmark #######################
##certain zero:
#encoder normalize by feature:
enc_ANN<-read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/encoder/tmp/B456_CAEANN.rds")
enc_SVR<-read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/encoder/tmp/B456_CAESVR.rds")
enc_CNN<-read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/encoder/tmp/B456_CNN.rds")

# #predict:
pred_ANN <-read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/predict/tmp/B456_avgSSE_test_CAE_ANN.rds")
pred_SVR<-read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/predict/tmp/B456_avgSSE_test_CAE_SVR.rds")
pred_CNN<-read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/predict/tmp/B456_avgSSE_test_CNN.rds")


maxmin <- function(x) (x - min(x))/(max(x)-min(x))
ANN_Normalize <- apply(pred_ANN[,2], 2, maxmin)
ANN_Normalize<- round(ANN_Normalize,digits = 3)

maxmin <- function(x) (x - min(x))/(max(x)-min(x))
SVR_Normalize <- apply(pred_SVR[,2], 2, maxmin)
SVR_Normalize<- round(SVR_Normalize,digits = 3)

maxmin <- function(x) (x - min(x))/(max(x)-min(x))
CNN_Normalize <- apply(pred_CNN[,2], 2, maxmin)
CNN_Normalize<- round(CNN_Normalize,digits = 3)

enc_ANN <- as.matrix(enc_ANN)
ANN_Normalize <- as.matrix(ANN_Normalize)
ANN <- enc_ANN%*%ANN_Normalize
enc_SVR <- as.matrix(enc_SVR)
SVR_Normalize <- as.matrix(SVR_Normalize)
SVR <- enc_SVR%*%SVR_Normalize
enc_CNN <- as.matrix(enc_CNN)
CNN_Normalize <- as.matrix(CNN_Normalize)
CNN <- enc_CNN%*%CNN_Normalize


maxmin <- function(x) (x - min(x))/(max(x)-min(x))
ANN <- apply(ANN, 2, maxmin)
ANN<- round(ANN,digits = 3)
maxmin <- function(x) (x - min(x))/(max(x)-min(x))
SVR <- apply(SVR, 2, maxmin)
SVR<- round(SVR,digits = 3)
maxmin <- function(x) (x - min(x))/(max(x)-min(x))
CNN <- apply(CNN, 2, maxmin)
CNN<- round(CNN,digits = 3)

SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
pred_overall  <- cbind(SVID,data.table(ANN),data.table(SVR),data.table(CNN))
colnames(pred_overall)<-c("SVID","CAE_ANN","CAE_SVR","CNN")
pred_overall <- melt(pred_overall)
colnames(pred_overall)<-c("SVID","model","error")

my3cols <- c( "#E7B800","#316A9E","#C5C5C5")
plot <- ggplot(pred_overall, aes(x =  SVID, y = error,fill=model)) + 
  geom_bar(stat = "identity",width = 0.5,position=position_dodge()) +
  scale_fill_manual(values=my3cols)+
  xlab("19 variable")+ylab("Min_max of prediction error")+ggtitle("A123 dataset: SVID influence in three model")+
  theme_minimal()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 12,face="bold"),
        axis.title = element_text(size = 12,face="bold"),
        plot.title = element_text(size = 30, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=20),
        axis.title.y = element_text(size=20),
        axis.text.y = element_text(size=15),
        axis.text.x = element_text(size=15),
        legend.title = element_text(size=20),
        legend.text = element_text(size=12),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank()) 
print(plot)

# ggsave(plot, file="C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/combine/encoder_minmaxbyfeature/pred/compare/Pred_A123_SVID.png", width=18, height=10)

#### 排名:
ANN <- data.table(ANN)
SVR <- data.table(SVR)
CNN <- data.table(CNN)
SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))


Normalize1<- cbind(SVID,ANN)
Normalize2<- cbind(SVID,SVR)
Normalize3<- cbind(SVID,CNN)
revise_ANN <- Normalize1[order(mean,decreasing = T),1:2]
revise_SVR <- Normalize2[order(mean,decreasing = T),1:2]
revise_CNN <- Normalize3[order(mean,decreasing = T),1:2]
dataset_rank <- cbind(revise_ANN[,1],revise_SVR[,1],revise_CNN[,1])
dataset_rank <- cbind(data.table(seq(1,dim(dataset_rank)[1],by=1)),dataset_rank)
colnames(dataset_rank)<-c("rank","CAE_ANN","CAE_SVR","CNN")


B456_pred <- dataset_rank
A456_pred <- dataset_rank
A123_pred <- dataset_rank
total <- cbind(A123_pred,A456_pred[,2:4],B456_pred[,2:4])
# saveRDS(total,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/combine/encoder_minmaxbyfeature/pred/compare/Pred_rank.rds")
# write.csv(total,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/combine/encoder_minmaxbyfeature/pred/compare/Pred_rank.csv")


#########################################################################################################
########################################### Overall importance ##########################################
############  reconstruction ########################################
xtest<-read_rds("D:/AmberChu/Handover/Data/CMP/Preprocess/final_set/A123_xtest.rds")
testy<-read_rds("D:/AmberChu/Handover/Data/CMP/Preprocess/final_set/A123_testy.rds")
##ANN:
model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/A123_CAE_ANN.h5")
test_loss<-read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Results_loss/A123_testloss_CAE_ANN.rds")
##SVR:
# custom loss:
# eplison <- tf$constant(0.5)
# # Margin term in loss
# svr_loss <- function(y_true,y_pred)
# {
# 
#   tf$reduce_mean(tf$maximum(0.,tf$subtract(tf$abs(tf$subtract(y_pred,y_true)),eplison)))
# 
# }
# with_custom_object_scope(c("svr_loss" = svr_loss), {
# 
#   model <- load_model_hdf5("C:/Users/User/Desktop/2021_0605/model/B456_CAE_SVR.h5")
# 
# })
##CAE:
# model <- load_model_hdf5("C:/Users/User/Desktop/2021_0605/model/A123_normalCAE.h5")
# test_loss<-read_rds("C:/Users/User/Desktop/2021_0605/model/Results_loss/A123_testloss_normalCAE.rds")

summary(model)
layer_name<-"autoencoder"
decoder <- keras_model(inputs=model$input,outputs=get_layer(model,layer_name)$output)
summary(decoder)
recon_list <- list() 
recon_feature <- list()
recon_list2 <- list() 
recon_feature2 <- list()
for(i in 1L:19L)
{
  
  ## certain zero:
  xtest<-read_rds(file = paste0("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/overall_revise/input_experiment/certain_zero/B456_xtest",i,".rds"))
  ## baseline real:
  real_xtest <-read_rds("D:/AmberChu/Handover/Data/CMP/Preprocess/final_set/A123_xtest.rds")

  reconstruct = decoder %>% predict(xtest)#  110 316  19   1
  dim(reconstruct) 
  reconstruct2 = decoder %>% predict(real_xtest)#  110 316  19   1
  dim(reconstruct) 
  
  
  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
    recon_list2[[j]]<- reconstruct2[j,,,]
  }
  recon_feature[[i]]<- recon_list
  recon_feature2[[i]]<- recon_list2
}

# saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/B456_reconstruct_test_test_normalCAE.rds")
# saveRDS(recon_feature2,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/B456_reconstruct_test_test_normalCAE(baseline).rds")

recon_feature<-read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/A123_reconstruct_test_test_normalCAE.rds")
# recon_feature<-read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/A123_reconstruct_test_test_normalCAE(baseline).rds")

xtest<-read_rds("D:/AmberChu/Handover/Data/CMP/Preprocess/final_set/A123_xtest.rds")
xtest <- array_reshape(xtest,dim=c(dim(xtest)[1],dim(xtest)[2],dim(xtest)[3]*1)) 
corr_error<- list()
corr_errorSVID_sum<-list()
for(i in 1L:19L)
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
    corr_error[[j]]<-sum2
  }
  
  corr_errorSVID_sum[[i]] <- corr_error
  
}

origin_errorSVID_MSE<- corr_errorSVID_sum
each_waferList <- list()
for(j in 1:dim(xtest)[1])
{
  null <- c()
  for(i in 1:19)
  {
    tmp <- origin_errorSVID_MSE[[i]][[j]]
    tmp <- data.table(tmp)
    null<-cbind(null,tmp)
    colnames(null) <- paste0('X', 1:(ncol(null)))
    
  }
  each_waferList[[j]]<-null
}


# saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/B456_Eachwafer_MSE_test_normalCAE.rds")
# saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/B456_Eachwafer_MSE_test_normalCAE(baseline).rds")


each_waferList <- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/A123_Eachwafer_MSE_test_CAE_SVR.rds")
each_waferList2 <- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/A123_Eachwafer_MSE_test_CAE_SVR(baseline).rds")

xtest<-read_rds("D:/AmberChu/Handover/Data/CMP/Preprocess/final_set/A123_xtest.rds")
xtest <- array_reshape(xtest,dim=c(dim(xtest)[1],dim(xtest)[2],dim(xtest)[3]*1)) 
tmp <- matrix(0:0, nrow = 19, ncol = 19)
tmp_list<- list()
tmp <-data.table(tmp)

for(j in 1:dim(xtest)[1])
{
  tmp_list[[j]] <- each_waferList[[j]]-each_waferList2[[j]]
  tmp<-tmp+tmp_list[[j]]
  
}
avg_wafer_error <- tmp/length(tmp_list)

# saveRDS(avg_wafer_error,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/B456_avg_MSE(actual-null)_test_normalCAE.rds")

############  prediction ########################################
xtest<-read_rds("D:/AmberChu/Handover/Data/CMP/Preprocess/final_set/A123_xtest.rds")
testy<-read_rds("D:/AmberChu/Handover/Data/CMP/Preprocess/final_set/A123_testy.rds")
#ANN:
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/B456_CAE_ANN.h5")
#SVR:
### custom loss:
eplison <- tf$constant(0.5)

# Margin term in loss
svr_loss <- function(y_true,y_pred)
{
  
  tf$reduce_mean(tf$maximum(0.,tf$subtract(tf$abs(tf$subtract(y_pred,y_true)),eplison)))
  
}
with_custom_object_scope(c("svr_loss" = svr_loss), {
  
  model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/A123_CAE_SVR.h5")
  
  
})
summary(model)
##CNN:
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/A123_CNN.h5")
layer_name2<-"predict"
predictor <- keras_model(inputs=model$input,outputs=get_layer(model,layer_name2)$output)
summary(predictor)
pred_list <- list()
pred_feature<- list()
pred_list2 <- list()
pred_feature2<- list()
for(i in 1L:19L)
{
  i=1
  ## certain zero:
  xtest<-read_rds(file = paste0("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/overall_revise/input_experiment/certain_zero/B456_xtest",i,".rds"))
 ## baseline real:
  real_xtest <-read_rds("D:/AmberChu/Handover/Data/CMP/Preprocess/final_set/A123_xtest.rds")
  pred_y = predictor %>% predict(xtest)#  110 316  19   1
  pred_real = predictor %>% predict(real_xtest)#  110 316  19   1
  
  for(j in 1: dim(pred_y)[1])
  {
    pred_list[[j]]<- pred_y[j,]
    pred_list2[[j]]<- pred_real[j,]
  }
  pred_feature[[i]]<- pred_list
  pred_feature2[[i]]<- pred_list2
}
saveRDS(pred_feature,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/B456_prediction_test_test_CNN.rds")
saveRDS(pred_feature2,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/B456_prediction_test_test_CNN(baseline).rds")



#### Actual_error - prediction_error
ytest<-read_rds("D:/AmberChu/Handover/Data/CMP/Preprocess/final_set/B456_testy.rds")

pred_feature <- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/B456_prediction_test_test_CNN.rds")
# pred_feature <- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/B456_prediction_test_test_CNN(baseline).rds")


SSE_matrix<- c()
for(i in 1L:19L)
{
  tt <- unlist(pred_feature[[i]])
  tt2<-data.frame(tt)
  real<-data.frame(ytest)
  revise<- real-tt2
  square <- data.frame(revise^2)
  SSE_matrix<- cbind(SSE_matrix,data.table(square))
  
}

# saveRDS(SSE_matrix,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/B456_errorSVID_prediction_SSEmatrix_test_CNN.rds")
# saveRDS(SSE_matrix,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/B456_errorSVID_prediction_SSEmatrix_test_CNN(baseline).rds")


origin_errorSVID_SSEmatrix<- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/B456_errorSVID_prediction_SSEmatrix_test_CNN.rds")
origin_errorSVID_SSEmatrix2<- read_rds("C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/B456_errorSVID_prediction_SSEmatrix_test_CNN(baseline).rds")
SSE_performance<-origin_errorSVID_SSEmatrix-origin_errorSVID_SSEmatrix2
colnames(SSE_performance) <- paste0('X', 1:(ncol(SSE_performance)))

value <- data.table(SSE_performance)
value<- round(value,digits = 4)
mean2 <- data.table(colMeans(value))
y_value2<- factor(seq(1,dim(value)[2],by=1),levels =seq(1,dim(value)[2],by=1))
cal2 <- cbind(y_value2,mean2)
colnames(cal2)<-c("wafer","mean")

# saveRDS(cal2,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/B456_avgSSE_test_CNN.rds")
saveRDS(cal2,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_nonzero/overall_revise/B456_avgSSE_test_CNN.rds")

############################################################################
####################### reconstruction heatmap #############################
avg_wafer_error<- read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/overall_revise/tmp/A123_avg_MSE(actual-null)_test_normalCAE.rds")

## normalize by row SVID:
maxmin <- function(x) (x - min(x))/(max(x)-min(x))
avgMSE_Normalize <- apply(avg_wafer_error, 1, maxmin) #轉:列19次實驗
avgMSE_Normalize<- round(avgMSE_Normalize,digits = 4)
SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
colnames(avgMSE_Normalize)<-SVID
exp<- factor(paste0('V', 1:19),levels =paste0('V', 1:19))
value<- cbind(exp,data.table(avgMSE_Normalize))
value<- melt(value)
colnames(value)<-c("experiment","SVID","error")
library(ggplot2)


# Plot 
mid<-(min(value$error)+max(value$error))/2
plot <- ggplot(data = value, aes(x = SVID, y = experiment)) + 
  geom_tile(aes(fill = error), color = "white", size = 1) +
  scale_fill_gradient2(
    low = 'steelblue', mid = 'white', high = 'red',
    midpoint = mid, guide = 'colourbar', aesthetics = 'fill',limit=c(min(value$error),max(value$error)))+
  xlab("SVID") + ylab("experiment")+
  theme_grey(base_size = 10) + 
  ggtitle("B456: overall SVID reconstruction influence in CAE+SVR ") +
  theme(axis.ticks = element_blank(), 
        panel.background = element_blank(), 
        plot.title = element_text(size=20,hjust = 0.5,face="bold"),legend.position="right",legend.direction="vertical",
        axis.text.x= element_text(size = 12,face="bold"),
        axis.text.y= element_text(size = 12,face="bold"),
        legend.text = element_text(size=12),legend.title = element_text(size=20),
        axis.title.x = element_text(size=15,face="bold"),axis.title.y = element_text(size=15,face="bold"))+
  guides(fill = guide_colorbar(barwidth = 1, barheight =7,title.position = "top", title.hjust = 0.5))+
  geom_text(aes(label = round(error, 3)),size=3.5) 
print(plot)
# ggsave(plot, file="C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/CAE/rec/A123_normalCAE.png", width=15, height=10)


############################################################################
####################### prediction vector ##################################
test_score <- read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/overall_revise/tmp/B456_avgSSE_test_CAE_ANN.rds")
maxmin <- function(x) (x - min(x))/(max(x)-min(x))
Normalize1 <- apply(test_score[,2], 2, maxmin)


Normalize1 <- data.table(Normalize1)
SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
set <- data.frame(SVID,Normalize1)
colnames(set)<-c("SVID","pred_mse")

plot <- ggplot(set, aes(x =  SVID, y = pred_mse)) +
  geom_bar(stat = "identity",fill="#739fc7",width = 0.5) +
  xlab("19 SVID")+ylab("Min max of prediction error")+
  geom_text(aes(label=round(pred_mse,digits=3)), vjust=-0.3, size=4)+
  ggtitle("A123: overall SVID prediction influence in CNN model") +
  theme_minimal()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 12,face="bold"), 
        axis.title = element_text(size = 12,face="bold"), 
        plot.title = element_text(size = 30, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=20,face="bold"),
        axis.title.y = element_text(size=20,face="bold"),
        axis.text.y = element_text(size=10),
        axis.text.x = element_text(size=15,face="bold"),
        legend.title = element_text(size=12),
        legend.text = element_text(size=12),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank()) 
print(plot)
# ggsave(plot, file="C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/ANN/pred/B456_CAEANN.png", width=15, height=10)

#######################################################################################
################################ comparison model #####################################
## prediction ------------
##certain0:
ANN <- read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/overall_revise/tmp/A123_avgSSE_test_CAE_ANN.rds")
SVR <- read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/overall_revise/tmp/A123_avgSSE_test_CAE_SVR.rds")
CNN <- read_rds("D:/AmberChu/Handover/Output_result/CMP/2021_0709_finalresults/model/Explainable/certain_zero/overall_revise/tmp/A123_avgSSE_test_CNN.rds")



maxmin <- function(x) (x - min(x))/(max(x)-min(x))
ANN <- apply(ANN[,2], 2, maxmin)
maxmin <- function(x) (x - min(x))/(max(x)-min(x))
SVR <- apply(SVR[,2], 2, maxmin)
maxmin <- function(x) (x - min(x))/(max(x)-min(x))
CNN <- apply(CNN[,2], 2, maxmin)
SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))
pred_overall  <- cbind(SVID,data.table(ANN),data.table(SVR),data.table(CNN))
colnames(pred_overall)<-c("SVID","CAE_ANN","CAE_SVR","CNN")
pred_overall <- melt(pred_overall)
colnames(pred_overall)<-c("SVID","model","error")

my3cols <- c( "#E7B800","#316A9E","#C5C5C5")
plot <- ggplot(pred_overall, aes(x =  SVID, y = error,fill=model)) + 
  geom_bar(stat = "identity",width = 0.5,position=position_dodge()) +
  scale_fill_manual(values=my3cols)+
  xlab("19 variable")+ylab("Min_max of prediction error")+ggtitle("B456 dataset: SVID influence in three model")+
  theme_minimal()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 12,face="bold"), 
        axis.title = element_text(size = 12,face="bold"),
        plot.title = element_text(size = 30, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=20),
        axis.title.y = element_text(size=20),
        axis.text.y = element_text(size=15),
        axis.text.x = element_text(size=15),
        legend.title = element_text(size=20),
        legend.text = element_text(size=12),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank()) 
print(plot)
# ggsave(plot, file="C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/compare/Pred_B456_SVID.png", width=18, height=10)

#### 排名:
ANN <- data.table(ANN)
SVR <- data.table(SVR)
CNN <- data.table(CNN)
SVID<- factor(paste0('SVID', 1:19),levels =paste0('SVID', 1:19))


Normalize1<- cbind(SVID,ANN)
Normalize2<- cbind(SVID,SVR)
Normalize3<- cbind(SVID,CNN)
revise_ANN <- Normalize1[order(mean,decreasing = T),1:2]
revise_SVR <- Normalize2[order(mean,decreasing = T),1:2]
revise_CNN <- Normalize3[order(mean,decreasing = T),1:2]
dataset_rank <- cbind(revise_ANN[,1],revise_SVR[,1],revise_CNN[,1])
dataset_rank <- cbind(data.table(seq(1,dim(dataset_rank)[1],by=1)),dataset_rank)
colnames(dataset_rank)<-c("rank","CAE_ANN","CAE_SVR","CNN")


B456_pred <- dataset_rank
A456_pred <- dataset_rank
A123_pred <- dataset_rank
total <- cbind(A123_pred,A456_pred[,2:4],B456_pred[,2:4])
# saveRDS(total,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/compare/Pred_rank.rds")
# write.csv(total,"C:/Users/User/Desktop/2021_0605/model/Explainable/certain_zero/overall_revise/compare/Pred_rank.csv")


