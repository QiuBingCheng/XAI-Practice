###############################################################################################################
###############################################################################################################
###################### mdoel explainability ###################################################################
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

# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_ANN.h5")
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/normalCAE.h5")
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CNN.h5")
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_LR.h5")
model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_SVM.h5")

summary(model)
################################################################################################
######################## Decomposed importance ##########################################
####################### 1. decoder module #######################################
layer_name<-"max_pool4"
encoder <- keras_model(inputs=model$input,outputs=get_layer(model,layer_name)$output)
summary(encoder)
test_feature  = encoder %>% predict(xtest)

## certain 0 experiment-------------------------------------------------------
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_ANN.h5")
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/normalCAE.h5")
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CNN.h5")
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_LR.h5")
model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_SVM.h5")

test_feature<-readRDS("C:/Users/User/Desktop/2021_0709/model/Explainable/testfeature_CAE_SVM.rds")

recon_list <- list() 
recon_feature <- list()
for(i in 1L:16L)
{
  
  cal_set <- test_feature
  tmp <- rep(0,nrow(cal_set))
  cal_set[,1,1,i]<-tmp
  # CAE_ANN reconstructed model -----------------------------------------
  # decoder model
  # dec_input = layer_input(shape = c(1,1,16))
  # dec1<- get_layer(cae_model,name="decoder1")
  # up_samp1<- get_layer(cae_model,name="up_samp1")
  # dec2<- get_layer(cae_model,name="decoder2")
  # up_samp2<- get_layer(cae_model,name="up_samp2")
  # dec3<- get_layer(cae_model,name="decoder3")
  # up_samp3<- get_layer(cae_model,name="up_samp3")
  # dec4<- get_layer(cae_model,name="decoder4")
  # up_samp4<- get_layer(cae_model,name="up_samp4")
  # dec5<- get_layer(cae_model,name="autoencoder")
  # decoder <- keras_model(dec_input,dec5(up_samp4(dec4(up_samp3(dec3(up_samp2(dec2(up_samp1(dec1(dec_input))))))))))
  
  
  # CAE_LR reconstructed model -----------------------------------------
  # decoder model
  # dec_input = layer_input(shape = c(1,1,16))
  # dec1<- get_layer(cae_model,name="decoder1")
  # up_samp1<- get_layer(cae_model,name="up_samp1")
  # dec2<- get_layer(cae_model,name="decoder2")
  # up_samp2<- get_layer(cae_model,name="up_samp2")
  # dec3<- get_layer(cae_model,name="decoder3")
  # up_samp3<- get_layer(cae_model,name="up_samp3")
  # dec4<- get_layer(cae_model,name="decoder4")
  # up_samp4<- get_layer(cae_model,name="up_samp4")
  # dec5<- get_layer(cae_model,name="autoencoder")
  # decoder <- keras_model(dec_input,dec5(up_samp4(dec4(up_samp3(dec3(up_samp2(dec2(up_samp1(dec1(dec_input))))))))))
  
  
  #CAE SVM -----------------------------------
  # decoder model
  dec_input = layer_input(shape = c(1,1,16))
  dec1<- get_layer(cae_model,name="decoder1")
  up_samp1<- get_layer(cae_model,name="up_samp1")
  dec2<- get_layer(cae_model,name="decoder2")
  up_samp2<- get_layer(cae_model,name="up_samp2")
  dec3<- get_layer(cae_model,name="decoder3")
  up_samp3<- get_layer(cae_model,name="up_samp3")
  dec4<- get_layer(cae_model,name="decoder4")
  up_samp4<- get_layer(cae_model,name="up_samp4")
  dec5<- get_layer(cae_model,name="autoencoder")
  decoder <- keras_model(dec_input,dec5(up_samp4(dec4(up_samp3(dec3(up_samp2(dec2(up_samp1(dec1(dec_input))))))))))
  
  #normalCAE ---------------------------------
  ## decoder 
  # dec_input = layer_input(shape = c(1,1,16))
  # dec1<- get_layer(cae_model,name="decoder1")
  # up_samp1<- get_layer(cae_model,name="up_samp1")
  # dec2<- get_layer(cae_model,name="decoder2")
  # up_samp2<- get_layer(cae_model,name="up_samp2")
  # dec3<- get_layer(cae_model,name="decoder3")
  # up_samp3<- get_layer(cae_model,name="up_samp3")
  # dec4<- get_layer(cae_model,name="decoder4")
  # up_samp4<- get_layer(cae_model,name="up_samp4")
  # dec5<- get_layer(cae_model,name="autoencoder")
  # decoder <- keras_model(dec_input,dec5(up_samp4(dec4(up_samp3(dec3(up_samp2(dec2(up_samp1(dec1(dec_input))))))))))
  # 
  summary(decoder)
  reconstruct = decoder %>% predict(cal_set)# 10000   28   28    1
  dim(reconstruct) 
  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
  }
  recon_feature[[i]]<- recon_list
}

saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0709/model/Explainable/decoder/tmp/reconstruct_test_CAE_SVM.rds")

### baseline ------------------------------
recon_list <- list()
recon_feature <- list()
for(i in 1L:16L)
{
  
  # CAE_ANN reconstructed model -----------------------------------------
  # decoder model
  # dec_input = layer_input(shape = c(1,1,16))
  # dec1<- get_layer(cae_model,name="decoder1")
  # up_samp1<- get_layer(cae_model,name="up_samp1")
  # dec2<- get_layer(cae_model,name="decoder2")
  # up_samp2<- get_layer(cae_model,name="up_samp2")
  # dec3<- get_layer(cae_model,name="decoder3")
  # up_samp3<- get_layer(cae_model,name="up_samp3")
  # dec4<- get_layer(cae_model,name="decoder4")
  # up_samp4<- get_layer(cae_model,name="up_samp4")
  # dec5<- get_layer(cae_model,name="autoencoder")
  # decoder <- keras_model(dec_input,dec5(up_samp4(dec4(up_samp3(dec3(up_samp2(dec2(up_samp1(dec1(dec_input))))))))))
  
  
  # CAE_LR reconstructed model -----------------------------------------
  # decoder model
  # dec_input = layer_input(shape = c(1,1,16))
  # dec1<- get_layer(cae_model,name="decoder1")
  # up_samp1<- get_layer(cae_model,name="up_samp1")
  # dec2<- get_layer(cae_model,name="decoder2")
  # up_samp2<- get_layer(cae_model,name="up_samp2")
  # dec3<- get_layer(cae_model,name="decoder3")
  # up_samp3<- get_layer(cae_model,name="up_samp3")
  # dec4<- get_layer(cae_model,name="decoder4")
  # up_samp4<- get_layer(cae_model,name="up_samp4")
  # dec5<- get_layer(cae_model,name="autoencoder")
  # decoder <- keras_model(dec_input,dec5(up_samp4(dec4(up_samp3(dec3(up_samp2(dec2(up_samp1(dec1(dec_input))))))))))
  
  #CAE SVM -----------------------------------
  # decoder model
  dec_input = layer_input(shape = c(1,1,16))
  dec1<- get_layer(cae_model,name="decoder1")
  up_samp1<- get_layer(cae_model,name="up_samp1")
  dec2<- get_layer(cae_model,name="decoder2")
  up_samp2<- get_layer(cae_model,name="up_samp2")
  dec3<- get_layer(cae_model,name="decoder3")
  up_samp3<- get_layer(cae_model,name="up_samp3")
  dec4<- get_layer(cae_model,name="decoder4")
  up_samp4<- get_layer(cae_model,name="up_samp4")
  dec5<- get_layer(cae_model,name="autoencoder")
  decoder <- keras_model(dec_input,dec5(up_samp4(dec4(up_samp3(dec3(up_samp2(dec2(up_samp1(dec1(dec_input))))))))))
  
  #normalCAE ---------------------------------
  ## decoder 
  # dec_input = layer_input(shape = c(1,1,16))
  # dec1<- get_layer(cae_model,name="decoder1")
  # up_samp1<- get_layer(cae_model,name="up_samp1")
  # dec2<- get_layer(cae_model,name="decoder2")
  # up_samp2<- get_layer(cae_model,name="up_samp2")
  # dec3<- get_layer(cae_model,name="decoder3")
  # up_samp3<- get_layer(cae_model,name="up_samp3")
  # dec4<- get_layer(cae_model,name="decoder4")
  # up_samp4<- get_layer(cae_model,name="up_samp4")
  # dec5<- get_layer(cae_model,name="autoencoder")
  # decoder <- keras_model(dec_input,dec5(up_samp4(dec4(up_samp3(dec3(up_samp2(dec2(up_samp1(dec1(dec_input))))))))))
  # 
  
  summary(decoder)
  reconstruct = decoder %>% predict(test_feature)# 1981   316   19    1
  dim(reconstruct) #1981   19   19    1
  
  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
  }
  recon_feature[[i]]<- recon_list
}

saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0709/model/Explainable/decoder/tmp/reconstruct_test_CAE_SVM(verify_0).rds")

#### Actual_corr - Reconstruct_corr ---------------------------------------------
xtest <- readRDS("D:/AmberChu/Amber/Fashion_mnist/data/xtest.rds")
xtest <- array_reshape(xtest,dim=c(dim(xtest)[1],dim(xtest)[2],dim(xtest)[3]*1)) 

# recon_feature <- read_rds("C:/Users/User/Desktop/2021_0709/model/Explainable/decoder/tmp/reconstruct_test_CAE_SVM.rds")
recon_feature <- read_rds("C:/Users/User/Desktop/2021_0709/model/Explainable/decoder/tmp/reconstruct_test_CAE_SVM(verify_0).rds")

dim(xtest)
corr_error3<- list()
corr_errorSVID_sum2<-list()

for(i in 1L:16L)
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
  for(i in 1:16L)
  {
    tmp <- origin_errorSVID_MSE[[i]][[j]]
    tmp <- data.table(tmp)
    null<-cbind(null,tmp)
    colnames(null) <- paste0('X', 1:(ncol(null)))
  }
  each_waferList[[j]]<-null
}

# saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0709/model/Explainable/decoder/tmp/Eachset_MSE_test_CAE_SVM.rds")
saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0709/model/Explainable/decoder/tmp/Eachset_MSE_test_CAE_SVM(verify0).rds")



xtest <- readRDS("D:/AmberChu/Amber/Fashion_mnist/data/xtest.rds")
xtest <- array_reshape(xtest,dim=c(dim(xtest)[1],dim(xtest)[2],dim(xtest)[3]*1)) 
each_waferList<- read_rds("C:/Users/User/Desktop/2021_0709/model/Explainable/decoder/tmp/Eachset_MSE_test_CAE_SVM.rds")
each_waferList2<- read_rds("C:/Users/User/Desktop/2021_0709/model/Explainable/decoder/tmp/Eachset_MSE_test_CAE_SVM(verify0).rds")

tmp <- matrix(0:0, nrow = 28, ncol = 16)
tmp_list<- list()
tmp <-data.table(tmp)

for(j in 1:dim(xtest)[1])
{
  tmp_list[[j]] <- each_waferList[[j]]-each_waferList2[[j]]
  tmp<-tmp+tmp_list[[j]]
  
}
avg_wafer_error <- tmp/length(tmp_list)

saveRDS(avg_wafer_error,"C:/Users/User/Desktop/2021_0709/model/Explainable/decoder/tmp/avg_MSE(actual-null)_test_CAE_SVM.rds")


##################################################
########## visualization decoder heatmap
## origin heatmap:
avg_wafer_error<- read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/decoder/tmp/avg_MSE(actual-null)_test_CAE_SVM.rds")


maxmin <- function(x) (x - min(x))/(max(x)-min(x))
avgMSE_Normalize <- apply(avg_wafer_error, 1, maxmin)
feature<- factor(paste0('f', 1:16),levels =paste0('f', 1:16))
SVID<- factor(paste0('Pixel', 1:28),levels =paste0('Pixel', 1:28))
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
  xlab("28 pixel ") + ylab("16 hidden feature")+
  theme_grey(base_size = 10) + 
  ggtitle("Average MSE of reconstruction heatmap") +
  theme(axis.ticks = element_blank(), 
        panel.background = element_blank(), 
        plot.title = element_text(size=30,hjust = 0.5,face="bold"),legend.position="right",legend.direction="vertical",
        axis.text.x= element_text(size = 12,face="bold"),
        axis.text.y= element_text(size = 12,face="bold"),
        legend.text = element_text(size=17),legend.title = element_text(size=20),
        axis.title.y = element_text(size=25,face="bold"),
        axis.title.x = element_blank())+
  guides(fill = guide_colorbar(barwidth = 1, barheight =7,title.position = "top", title.hjust = 0.5))


print(plot)
# ggsave(plot, file="C:/Users/User/Desktop/2021_0709/model/Explainable/decoder/tmp/chart/CAE_SVM.png", width=20, height=10)


###################################################################################
####################### 2. predictor module #######################################
# cae_model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_ANN.h5")
# cae_model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CNN.h5")
# cae_model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_LR.h5")
cae_model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_SVM.h5")


test_feature<-readRDS("C:/Users/User/Desktop/2021_0709/model/Explainable/testfeature_CAE_SVM.rds")
summary(cae_model)

## certain0 experiment -------------------------
pred_list <- list()
pred_feature <- list()
for(i in 1L:16L)
{
  cal_set <- test_feature
  tmp <- rep(0,nrow(cal_set))
  cal_set[,1,1,i]<-tmp
  
  ## CAE_ANN classifier model -------------------------------------
  # pred_input = layer_input(shape = c(1,1,16))
  # flatten<- get_layer(cae_model,name="flatten_2")
  # dec1<- get_layer(cae_model,name="dec_class2")
  # dec2<- get_layer(cae_model,name="dec_class3")
  # dec3<- get_layer(cae_model,name="dec_class4")
  # dec4<- get_layer(cae_model,name="classification")
  # predictor<- keras_model(pred_input,dec4(dec3(dec2(dec1(flatten(pred_input))))))
  
  
  ## CAE_SVM classifier model -------------------------------------
  pred_input = layer_input(shape = c(1,1,16))
  flatten<- get_layer(cae_model,name="flatten")
  dec1<- get_layer(cae_model,name="dec_class2")
  dec4<- get_layer(cae_model,name="classification")
  predictor<- keras_model(pred_input,dec4(dec1(flatten(pred_input))))
  
  ##CAE_LR classifier -----------------------------------------------
  # pred_input = layer_input(shape = c(1,1,16))
  # flatten<- get_layer(cae_model,name="flatten")
  # dec4<- get_layer(cae_model,name="classification")
  # predictor<- keras_model(pred_input,dec4(flatten(pred_input)))
  # 
  
  ##CNN classifier -----------------------------------------------
  # pred_input = layer_input(shape = c(1,1,16))
  # flatten<- get_layer(cae_model,name="flatten_1")
  # dec1<- get_layer(cae_model,name="dec_class2")
  # dec2<- get_layer(cae_model,name="dec_class3")
  # dec3<- get_layer(cae_model,name="dec_class4")
  # dec4<- get_layer(cae_model,name="classification")
  # predictor<- keras_model(pred_input,dec4(dec3(dec2(dec1(flatten(pred_input))))))
  # 
  
  summary(predictor)
  prediction = predictor %>% predict(cal_set)
  
  pred <- data.frame(apply(prediction, 1, which.max))
  for(j in 1: dim(pred)[1])
  {
    pred_list[[j]]<- pred[j,]
  }
  pred_feature[[i]]<- pred_list
}

saveRDS(pred_feature,"C:/Users/User/Desktop/2021_0709/model/Explainable/predict/tmp/prediction_test_CAE_SVM.rds")

### baseline experiment ------------------------------------
pred_list <- list()
pred_feature <- list()
for(i in 1L:16L)
{
  ## CAE_ANN classifier model -------------------------------------
  # pred_input = layer_input(shape = c(1,1,16))
  # flatten<- get_layer(cae_model,name="flatten_2")
  # dec1<- get_layer(cae_model,name="dec_class2")
  # dec2<- get_layer(cae_model,name="dec_class3")
  # dec3<- get_layer(cae_model,name="dec_class4")
  # dec4<- get_layer(cae_model,name="classification")
  # predictor<- keras_model(pred_input,dec4(dec3(dec2(dec1(flatten(pred_input))))))
  # 
  
  ## CAE_SVM classifier model -------------------------------------
  pred_input = layer_input(shape = c(1,1,16))
  flatten<- get_layer(cae_model,name="flatten")
  dec1<- get_layer(cae_model,name="dec_class2")
  dec4<- get_layer(cae_model,name="classification")
  predictor<- keras_model(pred_input,dec4(dec1(flatten(pred_input))))
  
  ##CAE_LR classifier -----------------------------------------------
  # pred_input = layer_input(shape = c(1,1,16))
  # flatten<- get_layer(cae_model,name="flatten")
  # dec4<- get_layer(cae_model,name="classification")
  # predictor<- keras_model(pred_input,dec4(flatten(pred_input)))
  # 
  
  
  ##CNN classifier -----------------------------------------------
  # pred_input = layer_input(shape = c(1,1,16))
  # flatten<- get_layer(cae_model,name="flatten_1")
  # dec1<- get_layer(cae_model,name="dec_class2")
  # dec2<- get_layer(cae_model,name="dec_class3")
  # dec3<- get_layer(cae_model,name="dec_class4")
  # dec4<- get_layer(cae_model,name="classification")
  # predictor<- keras_model(pred_input,dec4(dec3(dec2(dec1(flatten(pred_input))))))

  summary(predictor)
  prediction = predictor %>% predict(test_feature)# 1981   316   19    1
  pred <- data.frame(apply(prediction, 1, which.max))
  
  
  for(j in 1: dim(pred)[1])
  {
    pred_list[[j]]<- pred[j,]
  }
  pred_feature[[i]]<- pred_list
}

saveRDS(pred_feature,"C:/Users/User/Desktop/2021_0709/model/Explainable/predict/tmp/prediction_test_CAE_SVM(verify_0).rds")


#### Actual_corr - Reconstruct_corr
fashion_mnist <- dataset_fashion_mnist()

testy <- fashion_mnist$test$y

# pred_feature <- read_rds("C:/Users/User/Desktop/2021_0709/model/Explainable/predict/tmp/prediction_test_CAE_SVM.rds")
pred_feature <- read_rds("C:/Users/User/Desktop/2021_0709/model/Explainable/predict/tmp/prediction_test_CAE_SVM(verify_0).rds")



corr_errorSVID_sum2<-list()

for(i in 1L:16L)
{
  
  data <- cbind(testy,unlist(pred_feature[[i]]))
  data <- data.frame(data)
  data$V2<- data$V2-1
  set <- table(data)
  acc <- sum(diag(set))/sum(set)
  
  corr_errorSVID_sum2[[i]] <- acc
}

# saveRDS(corr_errorSVID_sum2,"C:/Users/User/Desktop/2021_0709/model/Explainable/predict/tmp/acc_test_CAE_SVM.rds")
saveRDS(corr_errorSVID_sum2,"C:/Users/User/Desktop/2021_0709/model/Explainable/predict/tmp/acc_test_CAE_SVM(verify0).rds")


origin_errorSVID_MSE<- read_rds("C:/Users/User/Desktop/2021_0709/model/Explainable/predict/tmp/acc_test_CAE_SVM.rds")
origin_errorSVID_MSE2<- read_rds("C:/Users/User/Desktop/2021_0709/model/Explainable/predict/tmp/acc_test_CAE_SVM(verify0).rds")

certain_MSE <- data.frame(unlist(origin_errorSVID_MSE))
baseline_MSE <- data.frame(unlist(origin_errorSVID_MSE2))
pred_performance<- certain_MSE-baseline_MSE

saveRDS(pred_performance,"C:/Users/User/Desktop/2021_0709/model/Explainable/predict/tmp/acc_error_performance_test_CAE_SVM.rds")


########################################################
######### visualization predictor vector ##############
avgSSE<-read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/predict/tmp/acc_error_performance_test_CNN.rds")
barplot(avgSSE$unlist.origin_errorSVID_MSE.)

avgSSE$unlist.origin_errorSVID_MSE.<- -(avgSSE$unlist.origin_errorSVID_MSE.)

maxmin <- function(x) (x - min(x))/(max(x)-min(x))
avgSSE_Normalize <- apply(avgSSE, 2, maxmin)
avgSSE_Normalize<- round(avgSSE_Normalize,digits = 3)

value<- cbind(data.table(factor(seq(1,dim(avgSSE_Normalize)[1],by=1))),avgSSE_Normalize)
colnames(value)<-c("number","mean")


plot <- ggplot(value, aes(x =  number, y = mean)) +
  geom_bar(stat = "identity",fill="#739fc7",width = 0.5) +
  xlab("16 feature")+ylab("min max normalize of prediction error")+
  ggtitle("Error proportion of 16 hidden feature ") +
  theme_minimal()+coord_flip()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),axis.line.y = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 20,face="bold"), 
        axis.title = element_text(size = 20,face="bold"), 
        plot.title = element_text(size = 30, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=25,face="bold"),
        axis.title.y = element_text(size=25,face="bold"),
        axis.text.y = element_text(size=15),
        axis.text.x = element_text(size=15,face="bold"),
        legend.title = element_text(size=12),
        legend.text = element_text(size=12),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.y = element_blank()) 

print(plot)
# ggsave(plot, file="C:/Users/User/Desktop/2021_0709/model/Explainable/predict/tmp/chart/acc_minmax_weight_CNN.png", width=15, height=15)

##########################################################################################################
######################################### encoder module ################################################
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

### ¨input experiment : ------------------------------

train_list <- list()
for(j in 1:28)
{
  tmp <- xtrain
  for(i in 1:dim(xtrain)[1])
  {

    tmp[i,,j,]<-rep(0,28)
  }
  train_list[[j]]<-tmp
}

test_list <- list()
for(j in 1:28)
{ 
  tmp2 <- xtest
  for(i in 1:dim(xtest)[1])
  {
    tmp2[i,,j,]<-rep(0,28)
  }
  test_list[[j]]<- tmp2
}

for(i in 1:28)
{
  xtrain <- train_list[[i]]
  xtest <- test_list[[i]]
  saveRDS(xtrain,file = paste0("C:/Users/User/Desktop/2021_0709/data/","xtrain",i,".rds"))
  saveRDS(xtest,file=paste0("C:/Users/User/Desktop/2021_0709/data/","xtest",i,".rds"))
  
}

# saveRDS(trainy,"C:/Users/User/Desktop/2021_0709/data/trainy.rds")
# saveRDS(testy,"C:/Users/User/Desktop/2021_0709/data/testy.rds")

# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_ANN.h5")
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CNN.h5")
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_LR.h5")
model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_SVM.h5")

summary(model)

null <-c()
for(i in 1:28)
{

  ## certain zero - baseline origin:
  xtest<-read_rds(file = paste0("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/data/xtest",i,".rds"))
  
  # test_feature<-readRDS("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/testfeature_CAE_ANN.rds")
  test_feature<-readRDS("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/testfeature_CAE_SVM.rds")
  # test_feature<-readRDS("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/testfeature_CAE_LR.rds")
  # test_feature<-readRDS("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/testfeature_normalCAE.rds")
  # test_feature<-readRDS("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/testfeature_CNN.rds")
  layer_name<-"max_pool4"
  
  
  encoder <- keras_model(inputs=model$input,outputs=get_layer(model,layer_name)$output)
  summary(encoder)
  new_feature  = encoder %>% predict(xtest)
  
  new_feature <- array_reshape(new_feature, dim=c(dim(new_feature)[1],dim(new_feature)[2]*dim(new_feature)[3]*dim(new_feature)[4])) 
  test_feature <- array_reshape(test_feature, dim=c(dim(test_feature)[1],dim(test_feature)[2]*dim(test_feature)[3]*dim(test_feature)[4]))
  tmp <- (new_feature - test_feature)^2
  tmp <- colMeans(tmp)
  null <- rbind(null,tmp)
  
}

maxmin <- function(x) (x - min(x))/(max(x)-min(x))
Normalize <- apply(null, 2, maxmin)
for(i in 1:dim(Normalize)[2])
{
  if(Normalize[1,i]== "NaN" )
  {
    Normalize[,i]<-0
  }
}
rownames(Normalize)<-seq(1,dim(Normalize)[1],by=1)

heatmap <- data.frame(Normalize)


# saveRDS(heatmap,"C:/Users/User/Desktop/2021_0709/model/Explainable/encoder/tmp/CNN.rds")
# saveRDS(heatmap,"C:/Users/User/Desktop/2021_0709/model/Explainable/encoder/tmp/CAEANN.rds")
# saveRDS(heatmap,"C:/Users/User/Desktop/2021_0709/model/Explainable/encoder/tmp/CAELR.rds")
saveRDS(heatmap,"C:/Users/User/Desktop/2021_0709/model/Explainable/encoder/tmp/CAESVM.rds")
# saveRDS(heatmap,"C:/Users/User/Desktop/2021_0709/model/Explainable/encoder/tmp/normalCAE.rds")

##########################################################
### visualize encoder heatmap ############################
avg_wafer_error<-read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/encoder/tmp/CAESVM.rds")
Pixel<- factor(paste0('Pixel', 1:28),levels =paste0('Pixel', 1:28))
feature<- factor(paste0('h', 1:16),levels =paste0('h', 1:16))
colnames(avg_wafer_error)<-feature
value<- cbind(Pixel,data.table(avg_wafer_error))
value<- melt(value)
colnames(value)<-c("Pixel","feature","error")
library(ggplot2)

mid<-(min(value$error)+max(value$error))/2
plot <- ggplot(data = value, aes(x = feature, y = Pixel)) + 
  geom_tile(aes(fill = error), color = "white", size = 1) +
  scale_fill_gradient2(
    low = 'steelblue', mid = 'white', high = 'red',
    midpoint = mid, guide = 'colourbar', aesthetics = 'fill',limit=c(min(value$error),max(value$error)))+
  xlab("16 hidden feature") + ylab("28 Pixel")+
  theme_grey(base_size = 10) + 
  ggtitle("MSE of encoder heatmap") +
  theme(axis.ticks = element_blank(), 
        panel.background = element_blank(), 
        plot.title = element_text(size=30,hjust = 0.5,face="bold"),legend.position="right",legend.direction="vertical",
        axis.text.y= element_text(size = 17,face="bold"),
        legend.text = element_text(size=17),legend.title = element_text(size=20),
        axis.text.x= element_text(size = 17,face="bold"),
        axis.title.x = element_text(size=25,face="bold"),
        axis.title.y = element_blank())+
  guides(fill = guide_colorbar(barwidth = 1, barheight =7,title.position = "top", title.hjust = 0.5))
print(plot)

# ggsave(plot, file="C:/Users/User/Desktop/2021_0709/model/Explainable/encoder/chart/CAESVM.png", width=15, height=20)


#################################################################################################
################################### Overall importance #####################################
######## reconstruction part ##############################
## certain 0 - baseline: ----------------------------------
xtest <- readRDS("D:/AmberChu/Handover/Data/FashionMNIST/xtest.rds")
testy <- readRDS("D:/AmberChu/Handover/Data/FashionMNIST/testy.rds")
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_ANN.h5")
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CNN.h5")
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_LR.h5")
model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_SVM.h5")
summary(model)
layer_name<-"autoencoder"
decoder <- keras_model(inputs=model$input,outputs=get_layer(model,layer_name)$output)
summary(decoder)
recon_list <- list() 
recon_feature <- list()
recon_list2 <- list() 
recon_feature2 <- list()
for(i in 1L:28L)
{
  
  ## certain zero:
  xtest<-read_rds(file = paste0("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/data/xtest",i,".rds"))
  ## baseline real:
  real_xtest <- readRDS("D:/AmberChu/Handover/Data/FashionMNIST/xtest.rds")
  reconstruct = decoder %>% predict(xtest)
  dim(reconstruct) 
  reconstruct2 = decoder %>% predict(real_xtest)
  dim(reconstruct) 

  for(j in 1: dim(reconstruct)[1])
  {
    recon_list[[j]]<- reconstruct[j,,,]
    recon_list2[[j]]<- reconstruct2[j,,,]
  }
  recon_feature[[i]]<- recon_list
  recon_feature2[[i]]<- recon_list2
}

# saveRDS(recon_feature,"C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/reconstruct_test_test_CAESVM.rds")
# saveRDS(recon_feature2,"C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/reconstruct_test_test_CAESVM(baseline).rds")


# recon_feature<-read_rds("C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/reconstruct_test_test_CAESVM.rds")
recon_feature<-read_rds("C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/reconstruct_test_test_CAESVM(baseline).rds")

xtest <- readRDS("D:/AmberChu/Amber/Fashion_mnist/data/xtest.rds")
xtest <- array_reshape(xtest,dim=c(dim(xtest)[1],dim(xtest)[2],dim(xtest)[3]*1)) 

corr_error<- list()
corr_errorSVID_sum<-list()
for(i in 1L:28L)
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
  for(i in 1:28)
  {
    tmp <- origin_errorSVID_MSE[[i]][[j]]
    tmp <- data.table(tmp)
    null<-cbind(null,tmp)
    colnames(null) <- paste0('X', 1:(ncol(null)))
    
  }
  each_waferList[[j]]<-null
}

# saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/Eachwafer_MSE_test_CAESVM.rds")
saveRDS(each_waferList,"C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/Eachwafer_MSE_test_CAESVM(baseline).rds")


each_waferList <- read_rds("C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/Eachwafer_MSE_test_CAESVM.rds")
each_waferList2 <- read_rds("C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/Eachwafer_MSE_test_CAESVM(baseline).rds")
xtest <- readRDS("D:/AmberChu/Amber/Fashion_mnist/data/xtest.rds")
xtest <- array_reshape(xtest,dim=c(dim(xtest)[1],dim(xtest)[2],dim(xtest)[3]*1)) 

tmp <- matrix(0:0, nrow = 28, ncol = 28)
tmp_list<- list()
tmp <-data.table(tmp)

for(j in 1:dim(xtest)[1])
{
  tmp_list[[j]] <- each_waferList[[j]]-each_waferList2[[j]]
  tmp<-tmp+tmp_list[[j]]
  
}
avg_wafer_error <- tmp/length(tmp_list)

saveRDS(avg_wafer_error,"C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/avg_MSE(actual-null)_test_CAESVM.rds")


######## prediction part ########################################
xtest <- readRDS("D:/AmberChu/Handover/Data/FashionMNIST/xtest.rds")
testy <- readRDS("D:/AmberChu/Handover/Data/FashionMNIST/testy.rds")

# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_ANN.h5")
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CNN.h5")
# model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_LR.h5")
model <- load_model_hdf5("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/CAE_SVM.h5")
summary(model)
layer_name2<-"classification"
predictor <- keras_model(inputs=model$input,outputs=get_layer(model,layer_name2)$output)
summary(predictor)



pred_list <- list()
pred_feature<- list()
pred_list2 <- list()
pred_feature2<- list()
for(i in 1L:28L)
{
  
  ## certain zero:
  xtest<-read_rds(file = paste0("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/data/xtest",i,".rds"))
  
  ## baseline real:
  real_xtest <- readRDS("D:/AmberChu/Handover/Data/FashionMNIST/xtest.rds")

  pred_y = predictor %>% predict(xtest)
  dim(pred_y) 
  pred_real = predictor %>% predict(real_xtest)
  dim(pred_real) 
  pred <- data.frame(apply(pred_y, 1, which.max))
  pred2 <- data.frame(apply(pred_real, 1, which.max))
  
  for(j in 1: dim(pred)[1])
  {
    pred_list[[j]]<- pred[j,]
    pred_list2[[j]]<- pred2[j,]
  }
  pred_feature[[i]]<- pred_list
  pred_feature2[[i]]<- pred_list2
}
saveRDS(pred_feature,"C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/prediction_test_test_CAESVM.rds")
saveRDS(pred_feature2,"C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/prediction_test_test_CAESVM(baseline).rds")


#### Actual_error - prediction_error
fashion_mnist <- dataset_fashion_mnist()
testy <- fashion_mnist$test$y

# pred_feature <- read_rds("C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/prediction_test_test_CAESVM.rds")
pred_feature <- read_rds("C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/prediction_test_test_CAESVM(baseline).rds")
acc_matrix<- c()
for(i in 1L:28L)
{
  
  data <- cbind(testy,unlist(pred_feature[[i]]))
  data <- data.frame(data)
  data$V2<- data$V2-1
  set <- table(data)
  acc <- sum(diag(set))/sum(set)
  acc_matrix <- rbind(acc_matrix,acc)
  
}

# saveRDS(acc_matrix,"C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/prediction_accmatrix_test_CAESVM.rds")
saveRDS(acc_matrix,"C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/prediction_accmatrix_test_CAESVM(baseline).rds")


origin_errorSVID_SSEmatrix<- read_rds("C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/prediction_accmatrix_test_CAESVM.rds")
origin_errorSVID_SSEmatrix2<- read_rds("C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/prediction_accmatrix_test_CAESVM(baseline).rds")
SSE_performance<-origin_errorSVID_SSEmatrix-origin_errorSVID_SSEmatrix2
colnames(SSE_performance) <- paste0('X', 1:(ncol(SSE_performance)))

value <- data.table(SSE_performance)
value<- round(value,digits = 4)
y_value2<- factor(seq(1,dim(value)[1],by=1),levels =seq(1,dim(value)[1],by=1))
cal2 <- cbind(y_value2,value)
colnames(cal2)<-c("pixel","error")

saveRDS(cal2,"C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/acc_error_test_CAESVM.rds")

#####################################################################
####################### reconstruction heatmap ######################
avg_wafer_error<- read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/overall_revise/avg_MSE(actual-null)_test_normalCAE.rds")
## normalize by row SVID:
maxmin <- function(x) (x - min(x))/(max(x)-min(x))
avgMSE_Normalize <- apply(avg_wafer_error, 1, maxmin) 
avgMSE_Normalize<- round(avgMSE_Normalize,digits = 4)
Pixel<- factor(paste0('P', 1:28),levels =paste0('P', 1:28))
colnames(avgMSE_Normalize)<-Pixel
exp<- factor(paste0('V', 1:28),levels =paste0('V', 1:28))
value<- cbind(exp,data.table(avgMSE_Normalize))
value<- melt(value)
colnames(value)<-c("experiment","Pixel","error")
library(ggplot2)


# Plot 
mid<-(min(value$error)+max(value$error))/2
plot <- ggplot(data = value, aes(x = Pixel, y = experiment)) + 
  geom_tile(aes(fill = error), color = "white", size = 1) +
  scale_fill_gradient2(
    low = 'steelblue', mid = 'white', high = 'red',
    midpoint = mid, guide = 'colourbar', aesthetics = 'fill',limit=c(min(value$error),max(value$error)))+
  xlab("pixel") + ylab("experiment")+
  theme_grey(base_size = 10) + 
  ggtitle("Overall pixel reconstruction influence in normalCAE ") +
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


# ggsave(plot, file="C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/rec/normalCAE.png", width=15, height=10)


#####################################################################
####################### prediction vector ##########################
test_score <- read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/overall_revise/acc_error_test_CNN.rds")

test_score$error <- -(test_score$error)
maxmin <- function(x) (x - min(x))/(max(x)-min(x))
Normalize1 <- apply(test_score[,2], 2, maxmin)
Normalize1 <- data.table(Normalize1)
Pixel<- factor(paste0('P', 1:28),levels =paste0('P', 1:28))
set <- data.frame(Pixel,Normalize1)
colnames(set)<-c("Pixel","pred_error")
plot <- ggplot(set, aes(x =  Pixel, y = pred_error)) +
  geom_bar(stat = "identity",fill="#739fc7",width = 0.5) +
  xlab("28 pixel")+ylab("Min max of prediction error")+
  geom_text(aes(label=round(pred_error,digits=3)), vjust=-0.3, size=4)+
  ggtitle("Overall pixel prediction influence in CNN model") +
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
# ggsave(plot, file="C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/pred/CNN.png", width=15, height=10)



################################################################
######################## comparison model ######################
## prediction 
##zero:
#CAE+ANN
ANN <- read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/overall_revise/acc_error_test_CAEANN.rds")
## CAE+SVM
SVM <- read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/overall_revise/acc_error_test_CAESVM.rds")

# # # CAE+LR
LR <- read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/overall_revise/acc_error_test_CAELR.rds")
# ## CNN
CNN <- read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/overall_revise/acc_error_test_CNN.rds")

ANN$error <- -(ANN$error)
SVM$error <- -(SVM$error)
LR$error <- -(LR$error)
CNN$error <- -(CNN$error)

maxmin <- function(x) (x - min(x))/(max(x)-min(x))
ANN <- apply(ANN[,2], 2, maxmin)

maxmin <- function(x) (x - min(x))/(max(x)-min(x))
SVM <- apply(SVM[,2], 2, maxmin)

maxmin <- function(x) (x - min(x))/(max(x)-min(x))
LR <- apply(LR[,2], 2, maxmin)


maxmin <- function(x) (x - min(x))/(max(x)-min(x))
CNN <- apply(CNN[,2], 2, maxmin)

Pixel <- factor(paste0('P', 1:28),levels =paste0('P', 1:28))
pred_overall  <- cbind(Pixel,data.table(ANN),data.table(SVM),data.table(LR),data.table(CNN))
colnames(pred_overall)<-c("Pixel","CAE_ANN","CAE_SVM","CAE_LR","CNN")
pred_overall <- melt(pred_overall)
colnames(pred_overall)<-c("Pixel","model","error")
my3cols <- c( "#CC0000","#316A9E","#E7B800","#858585")


plot <- ggplot(pred_overall, aes(x =  Pixel, y = error,fill=model)) + 
  geom_bar(stat = "identity",width = 0.5,position=position_dodge()) +
  scale_fill_manual(values=my3cols)+
  xlab("28 pixel")+ylab("Min_max of prediction error")+ggtitle("Pixel influence in four model")+
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
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank()) ###
print(plot)
# ggsave(plot, file="C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/compare/Pred.png", width=18, height=10)

#### ??????:
ANN <- data.table(ANN)
SVM <- data.table(SVM)
LR <- data.table(LR)
CNN <- data.table(CNN)
Pixel<- factor(paste0('P', 1:28),levels =paste0('P', 1:28))


Normalize1<- cbind(Pixel,ANN)
Normalize2<- cbind(Pixel,SVM)
Normalize3<- cbind(Pixel,LR)
Normalize4<- cbind(Pixel,CNN)
revise_ANN <- Normalize1[order(error,decreasing = T),1:2]
revise_SVM <- Normalize2[order(error,decreasing = T),1:2]
revise_LR  <- Normalize3[order(error,decreasing = T),1:2]
revise_CNN <- Normalize4[order(error,decreasing = T),1:2]

dataset_rank <- cbind(revise_ANN[,1],revise_SVM[,1],revise_LR[,1],revise_CNN[,1])
dataset_rank <- cbind(data.table(seq(1,dim(dataset_rank)[1],by=1)),dataset_rank)
colnames(dataset_rank)<-c("rank","CAE_ANN","CAE_SVM","CAE_LR","CNN")

saveRDS(dataset_rank,"C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/compare/Pred_rank.rds")
write.csv(dataset_rank,"C:/Users/User/Desktop/2021_0709/model/Explainable/overall_revise/compare/Pred_rank.csv")

####################################################################################################
############################### Combination of encoder and predictor module ###########################
## certain zero:
#encoder:(??????min max)
avg_wafer_error<-read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/encoder/tmp/CNN.rds")

#predict:
avgSSE<-read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/predict/tmp/acc_error_performance_test_CNN.rds")
avgSSE$unlist.origin_errorSVID_MSE. <- -(avgSSE$unlist.origin_errorSVID_MSE.)


maxmin <- function(x) (x - min(x))/(max(x)-min(x))
avgSSE_Normalize <- apply(avgSSE, 2, maxmin)
avgSSE_Normalize<- round(avgSSE_Normalize,digits = 3)
enc_matrix <- as.matrix(avg_wafer_error)
pred_matrix <- as.matrix(avgSSE_Normalize)
tmp <- enc_matrix%*%pred_matrix


maxmin <- function(x) (x - min(x))/(max(x)-min(x))
tmp <- apply(tmp, 2, maxmin)

Pixel <- factor(paste0('P', 1:28),levels =paste0('P', 1:28))
pred_overall <- cbind(Pixel,data.table(tmp))
colnames(pred_overall)<-c("Pixel","error")

plot <- ggplot(pred_overall, aes(x =  Pixel, y = error)) +
  geom_bar(stat = "identity",fill="#739fc7",width = 0.5) +
  xlab("28 Pixel")+ylab("encoder*prediction")+
  geom_text(aes(label=round(error,digits=3)), vjust=-0.3, size=4)+
  ggtitle(" Combination of encoder and regressor in CNN model") +
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


# ggsave(plot, file="C:/Users/User/Desktop/2021_0709/model/Explainable/combine/pred/pixel_imp_CNN.png", width=15, height=10)



###########################################################################################
######################## Combination of encoder and decoder module ########################
##certain zero
# encoder normalize by feature:
encoder<-read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/encoder/tmp/normalCAE.rds")

#decoder:
decoder<- read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/decoder/tmp/avg_MSE(actual-null)_test_normalCAE.rds")

maxmin <- function(x) (x - min(x))/(max(x)-min(x))
decoder <- apply(decoder, 1, maxmin)
enc_matrix <- as.matrix(encoder)
dec_matrix <- as.matrix(decoder)
tmp <- enc_matrix%*%dec_matrix
tmp <- tmp/16
tmp<- round(tmp,digits = 4)
Pixel<- factor(paste0('P', 1:28),levels =paste0('P', 1:28))
colnames(tmp)<-Pixel
exp<- factor(paste0('V', 1:28),levels =paste0('V', 1:28))
value<- cbind(exp,data.table(tmp))
value<- melt(value)
colnames(value)<-c("experiment","Pixel","error")
library(ggplot2)

# Plot 
mid<-(min(value$error)+max(value$error))/2
plot <- ggplot(data = value, aes(x = Pixel, y = experiment)) + 
  geom_tile(aes(fill = error), color = "white", size = 1) +
  scale_fill_gradient2(
    low = 'steelblue', mid = 'white', high = 'red',
    midpoint = mid, guide = 'colourbar', aesthetics = 'fill',limit=c(min(value$error),max(value$error)))+
  xlab("Pixel") + ylab("experiment")+
  theme_grey(base_size = 10) + 
  ggtitle("Overall pixel reconstruction influence in normalCAE model ") +
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

# ggsave(plot, file="C:/Users/User/Desktop/2021_0709/model/Explainable/combine/rec/normalCAE.png", width=15, height=10)



#######################################
## comparison(encoder*regressor) ------
##certain zero:
#encoder normalize by feature:
enc_ANN<-read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/encoder/tmp/CAEANN.rds")
enc_SVM<-read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/encoder/tmp/CAESVM.rds")
enc_LR<-read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/encoder/tmp/CAELR.rds")
enc_CNN<-read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/encoder/tmp/CNN.rds")
# 
# #predict:
pred_ANN <-read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/predict/tmp/acc_error_performance_test_CAE_ANN.rds")
pred_SVM<-read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/predict/tmp/acc_error_performance_test_CAE_SVM.rds")
pred_LR<-read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/predict/tmp/acc_error_performance_test_CAE_LR.rds")
pred_CNN<-read_rds("D:/AmberChu/Handover/Output_result/Fashionmnist/2021_0709_finalresults/model/Explainable/predict/tmp/acc_error_performance_test_CNN.rds")


pred_ANN$unlist.origin_errorSVID_MSE.<- -pred_ANN$unlist.origin_errorSVID_MSE.
pred_SVM$unlist.origin_errorSVID_MSE.<- -pred_SVM$unlist.origin_errorSVID_MSE.
pred_LR$unlist.origin_errorSVID_MSE.<- -pred_LR$unlist.origin_errorSVID_MSE.
pred_CNN$unlist.origin_errorSVID_MSE.<- -pred_CNN$unlist.origin_errorSVID_MSE.


maxmin <- function(x) (x - min(x))/(max(x)-min(x))
ANN_Normalize <- apply(pred_ANN, 2, maxmin)
ANN_Normalize<- round(ANN_Normalize,digits = 3)

maxmin <- function(x) (x - min(x))/(max(x)-min(x))
SVM_Normalize <- apply(pred_SVM, 2, maxmin)
SVM_Normalize<- round(SVM_Normalize,digits = 3)

maxmin <- function(x) (x - min(x))/(max(x)-min(x))
LR_Normalize <- apply(pred_LR, 2, maxmin)
LR_Normalize<- round(LR_Normalize,digits = 3)

maxmin <- function(x) (x - min(x))/(max(x)-min(x))
CNN_Normalize <- apply(pred_CNN, 2, maxmin)
CNN_Normalize<- round(CNN_Normalize,digits = 3)

enc_ANN <- as.matrix(enc_ANN)
ANN_Normalize <- as.matrix(ANN_Normalize)
ANN <- enc_ANN%*%ANN_Normalize


enc_SVM <- as.matrix(enc_SVM)
SVM_Normalize <- as.matrix(SVM_Normalize)
SVM <- enc_SVM%*%SVM_Normalize

enc_LR <- as.matrix(enc_LR)
LR_Normalize <- as.matrix(LR_Normalize)
LR <- enc_LR%*%LR_Normalize



enc_CNN <- as.matrix(enc_CNN)
CNN_Normalize <- as.matrix(CNN_Normalize)
CNN <- enc_CNN%*%CNN_Normalize


maxmin <- function(x) (x - min(x))/(max(x)-min(x))
ANN <- apply(ANN, 2, maxmin)
ANN<- round(ANN,digits = 3)

maxmin <- function(x) (x - min(x))/(max(x)-min(x))
SVM <- apply(SVM, 2, maxmin)
SVM<- round(SVM,digits = 3)


maxmin <- function(x) (x - min(x))/(max(x)-min(x))
LR <- apply(LR, 2, maxmin)
LR<- round(LR,digits = 3)


maxmin <- function(x) (x - min(x))/(max(x)-min(x))
CNN <- apply(CNN, 2, maxmin)
CNN<- round(CNN,digits = 3)

Pixel<- factor(paste0('P', 1:28),levels =paste0('P', 1:28))
pred_overall  <- cbind(Pixel,data.table(ANN),data.table(SVM),data.table(LR),data.table(CNN))
colnames(pred_overall)<-c("Pixel","CAE_ANN","CAE_SVM","CAE_LR","CNN")
pred_overall <- melt(pred_overall)
colnames(pred_overall)<-c("Pixel","model","error")
my3cols <- c( "#CC0000","#316A9E","#E7B800","#858585")


plot <- ggplot(pred_overall, aes(x =  Pixel, y = error,fill=model)) + 
  geom_bar(stat = "identity",width = 0.5,position=position_dodge()) +
  scale_fill_manual(values=my3cols)+
  xlab("28 Pixel")+ylab("Min_max of prediction error")+ggtitle("Pixel influence in four model")+
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

# ggsave(plot, file="C:/Users/User/Desktop/2021_0709/model/Explainable/combine/pred/compare/Pred_pixel.png", width=18, height=10)



#### ??????:
ANN <- data.table(ANN)
SVM <- data.table(SVM)
LR <- data.table(LR)
CNN <- data.table(CNN)
Pixel<- factor(paste0('P', 1:28),levels =paste0('P', 1:28))


Normalize1<- cbind(Pixel,ANN)
Normalize2<- cbind(Pixel,SVM)
Normalize3<- cbind(Pixel,LR)
Normalize4<- cbind(Pixel,CNN)
revise_ANN <- Normalize1[order(unlist.origin_errorSVID_MSE.,decreasing = T),1:2]
revise_SVM <- Normalize2[order(unlist.origin_errorSVID_MSE.,decreasing = T),1:2]
revise_LR <- Normalize3[order(unlist.origin_errorSVID_MSE.,decreasing = T),1:2]
revise_CNN <- Normalize4[order(unlist.origin_errorSVID_MSE.,decreasing = T),1:2]


dataset_rank <- cbind(revise_ANN[,1],revise_SVM[,1],revise_LR[,1],revise_CNN[,1])
dataset_rank <- cbind(data.table(seq(1,dim(dataset_rank)[1],by=1)),dataset_rank)
colnames(dataset_rank)<-c("rank","CAE_ANN","CAE_SVM","CAE_LR","CNN")


# saveRDS(dataset_rank,"C:/Users/User/Desktop/2021_0709/model/Explainable/combine/pred/compare/Pred_rank.rds")
# write.csv(dataset_rank,"C:/Users/User/Desktop/2021_0709/model/Explainable/combine/pred/compare/Pred_rank.csv")













