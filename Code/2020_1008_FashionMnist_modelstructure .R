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

xtrain <- array_reshape(x_train, dim=c(dim(x_train)[1],dim(x_train)[2],dim(x_train)[3],1)) #60000    28    28     1
xtest <- array_reshape(x_test, dim=c(dim(x_test)[1],dim(x_test)[2],dim(x_test)[3],1))#10000    28    28     1
## input data (xtrain、xtest、testY、trainY)
xtrain <- readRDS("C:/Users/User/Desktop/Amber/Fashion_mnist/xtrain")
xtest <- readRDS("C:/Users/User/Desktop/Amber/Fashion_mnist/xtest")
testY <- readRDS("C:/Users/User/Desktop/Amber/Fashion_mnist/testY")
trainY <- readRDS("C:/Users/User/Desktop/Amber/Fashion_mnist/trainY")


#### build proposed model -----------------------------------------------------
## encoder
enc_input = layer_input(shape = c(28, 28, 1),name="input")
enc_output = enc_input %>% 
  layer_conv_2d(64,kernel_size=c(3,3), activation="relu", padding="same",name="encoder1") %>% 
  layer_max_pooling_2d(c(2,2), padding="same",name="max_pool1") %>% 
  layer_conv_2d(32,kernel_size=c(3,3), activation="relu", padding="same",name="encoder2") %>% 
  layer_max_pooling_2d(c(2,2), padding="same",name="max_pool2")%>%
  layer_conv_2d(32,kernel_size=c(3,3), activation="relu", padding="same",name="encoder3") %>% 
  layer_max_pooling_2d(c(3,3), padding="same",name="max_pool3")%>%
  layer_conv_2d(20,kernel_size=c(3,3), activation="relu", padding="same",name="encoder4") %>% 
  layer_max_pooling_2d(c(3,3), padding="same",name="max_pool4")

encoder <- keras_model(enc_input,enc_output)
summary(encoder)

## classifier 
classify <- encoder$output%>%
  layer_flatten()%>%
  layer_dense(units=20,activation="relu",name="dec_class2")%>%
  layer_dense(units=64,activation="relu",name="dec_class3")%>%
  layer_dense(units=16,activation="relu",name="dec_class4")%>%
  layer_dense(units=10,activation="softmax",name="classification")
classify_model <- keras_model(encoder$input,classify)
summary(classify_model)

## decoder 
decoder <- encoder$output %>%
  layer_conv_2d(20, kernel_size=c(3,3), activation="relu", padding="same",name="decoder1") %>% 
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

parallel_model <- multi_gpu_model(model, gpus = 2)

# training model ---------------------------
parallel_model%>% compile(optimizer="RMSprop",
                 loss=list("classification"="categorical_crossentropy","autoencoder"="mse")
                 ,metric=list("classification"="accuracy"))
callbacks = list(
callback_model_checkpoint("checkpoints.h5"),
callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1))
#callbacks <- callback_early_stopping(monitor="val_loss",patience = 5)

parallel_model %>% fit(x= xtrain, y=list("classification"= trainY,"autoencoder"= xtrain),
              validation_data=list(x=xtest,y=list("classification"=testY,"autoencoder"=xtest)),
              batch_size = 128,epochs = 25,callback=callbacks)
save_model_hdf5(model,"C:/Users/User/Desktop/CAE+ANN/CAE_ANN.h5")

cae_model <- load_model_hdf5("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/visualization/classification CAE/nolossweight/CAE_ANN.h5")
summary(cae_model)


## encode test data----------------------
layer_name<-"max_pool4"
encoder <- keras_model(inputs=cae_model$input,outputs=get_layer(cae_model,layer_name)$output)
summary(encoder)
encoded_imgs = encoder %>% predict(xtest)
dim(encoded_imgs)#  10000     1     1    16

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
summary(decoder)
decoded_imgs = decoder %>% predict(encoded_imgs)# 10000    28    28     1


# calculating reconstruction error (mse)
reconstruction_error =metric_mean_squared_error(xtest,decoded_imgs)
paste("reconstruction error: "
      ,k_get_value(k_mean(reconstruction_error)))  

recon_error <- k_get_value(k_mean(reconstruction_error))

## reconstruct -----------------------------------------------------
## AE reconstruct error 
decoded_imgs = decoder %>% predict(encoded_imgs) 
error <- rowMeans((decoded_imgs - xtest)**2) 
error <- data.table(error)
error 


## classification error
eval <- data.frame(error=error, class = as.factor(y_test))
eval %>% group_by(class) %>%summarise(avg_error = mean(error))
matrix<- eval %>% group_by(class) %>%summarise(avg_error = mean(error))#各種數字的平均error
avg_error <- as.numeric(matrix$avg_error) 
sum(avg_error)/10 
eval
#plot error
library(ggplot2)
eval %>% group_by(class) %>% summarise(avg_error = mean(error)) %>%
  ggplot(aes(x = class, y = avg_error, fill = class)) +
  geom_col() + ggtitle("Reconstruction error") + 
  theme_minimal()+theme(plot.title = element_text(size=17,hjust = 0.5,face="bold"),axis.text.x= element_text(size = 12))

# visualization 
dim(xtest) <- c(nrow(xtest), 28, 28)
dim(decoded_imgs) <- c(nrow(decoded_imgs), 28, 28)
n = 30
op = par(mfrow=c(5,2), mar=c(1,0,0,0))
for (i in 1:n) 
{
  plot(as.raster(xtest[i,,]))
  plot(as.raster(decoded_imgs[i,,]))
}


# feature distribution ------------------------------------------------
library(ggplot2)
library(dplyr)
batch_size=128
x_test_encoded <- predict(encoder, xtest, batch_size = batch_size)
dim(x_test_encoded)
x_test_encoded %>%
  as_data_frame() %>%
  mutate(class = as.factor(fashion_mnist$test$y)) %>%
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point()+
  ggtitle('Feature distribution')+
  theme(plot.title = element_text(size=17,hjust = 0.5,face="bold"))


## latent visualization------------------------------------------------
dim(encoded_imgs) #10000     1     1    16
encoded_matrix <- array_reshape(encoded_imgs, c(nrow(encoded_imgs),16), order = "F")
library(corrplot)
library(reshape2)
library(ggplot2)

## correlation 分析----------------------------------------------------
cormat <- round(cor(encoded_matrix),2)
head(cormat)
melted_cormat <- melt(cormat)

# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}
upper_tri <- get_upper_tri(cormat)
upper_tri
# Melt the correlation matrix
library(reshape2)
melted_cormat <- melt(upper_tri, na.rm = TRUE)
# Heatmap
library(ggplot2)
ggheatmap<- ggplot(data = melted_cormat, aes(Var2, Var1, fill = value),width=200,height=200)+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_gray()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+coord_fixed()
ggheatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
  theme(
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))
### pca 分析--------------------------------------------------------------------------------------
encoded.eigen<-eigen(cormat)
encoded.eigen$values/(sum(encoded.eigen$values))
encoded_pca <-princomp(encoded_matrix, cor=TRUE, scores=TRUE)
summary(encoded_pca)
loadings(encoded_pca)
screeplot(encoded_pca,type="lines")
pcscores <-as.data.frame(encoded_pca$scores)
head(pcscores)
PC<-cbind(encoded_matrix, pcscores)
### 分布圖 ### 
## lattice method ------------------------
library(lattice)
xyplot(Comp.2~Comp.1, groups=y_test, data=PC,
       main=list("PCA for latent representation ", cex=1.6),pch=20,grid=TRUE,auto.key = list(
         title="number"
         , corner = c(0.2, 0)
         , x = 0.95
         , y = 0.1,cex=0.7))
## ggplot2 -----------------------------
PC %>%
  as_data_frame() %>%
  mutate(class = as.factor(fashion_mnist$test$y)) %>%
  ggplot(aes(x =Comp.1, y =Comp.2, colour = class)) + geom_point()+
  ggtitle('PCA for Latent Representation(classification CAE) ')+
  theme_bw()+theme(legend.title = element_text(size=12,face="bold"),
                   legend.text = element_text(size=10,face="bold"),
                   plot.title = element_text(size=17,hjust = 0.5,face="bold"))


####-------------------------------------
pca = prcomp(encoded_matrix, center=T, scale.=T)
## eigenvalues
round(pca$sdev^2, 2)
pcs = data.frame(pca$x)
str(pcs)
## percentage pcs 
cov = round(pca$sdev^2/sum(pca$sdev^2)*100, 2)
cov = data.frame(c(1:16),cov)
names(cov)[1] = 'PCs'
names(cov)[2] = 'Variance'
cov
sum(cov$Variance[1:2])
sum(cov$Variance[1:3])
sum(cov$Variance[1:4])
library(RCurl)
library(knitr)
library(plyr)
library(rCharts)
library(qcc)
library(threejs)
library(rgl)
library(pca3d)
library(gridExtra)
PCA = pca$sdev^2
names(PCA) = paste0('PC', cov$PCs)
qcc::pareto.chart(PCA)
## sum of PC12 
sum(cov$Variance[1:2])
# PC123
sum(cov$Variance[1:3])

pcs%>%
  mutate(class = as.factor(fashion_mnist$test$y)) %>%
  ggplot(aes(x =PC1, y =PC2, colour = class)) + geom_point()+
  ggtitle('PCA for Latent Representation')+
  theme_bw()+theme(legend.title = element_text(size=15,face="bold"),
                   legend.text = element_text(size=15),
                   plot.title = element_text(size=17,hjust = 0.5,face="bold"))+
  guides(colour = guide_legend(override.aes = list(size=2, stroke=1.5))) 


p1<-pcs%>%
  mutate(class = as.factor(fashion_mnist$test$y)) %>%
  ggplot(aes(x =PC1, y =PC2, colour = class)) + geom_point()+
  theme_bw()+theme(legend.title = element_text(size=12,face="bold"),
                   legend.text = element_text(size=10,face="bold"),
                   plot.title = element_text(size=17,hjust = 0.5,face="bold"))
p2<-pcs%>%
  mutate(class = as.factor(fashion_mnist$test$y)) %>%
  ggplot(aes(x =PC2, y =PC3, colour = class)) + geom_point()+
  theme_bw()+theme(legend.title = element_text(size=12,face="bold"),
                   legend.text = element_text(size=10,face="bold"),
                   plot.title = element_text(size=17,hjust = 0.5,face="bold"))
p3<-pcs%>%
  mutate(class = as.factor(fashion_mnist$test$y)) %>%
  ggplot(aes(x =PC1, y =PC3, colour = class)) + geom_point()+
  theme_bw()+theme(legend.title = element_text(size=12,face="bold"),
                   legend.text = element_text(size=10,face="bold"),
                   plot.title = element_text(size=17,hjust = 0.5,face="bold"))
p4<-pcs%>%
  mutate(class = as.factor(fashion_mnist$test$y)) %>%
  ggplot(aes(x =PC1, y =PC4, colour = class)) + geom_point()+
  theme_bw()+theme(legend.title = element_text(size=12,face="bold"),
                   legend.text = element_text(size=10,face="bold"),
                   plot.title = element_text(size=17,hjust = 0.5,face="bold"))
grid.arrange(p1, p2, p3, p4, ncol=2)


#########################################################################################################
## 一般CAE
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
ae_model %>% fit(xtrain,xtrain,batch_size=128,epochs=25,
                 validation_data=list(xtest,xtest),callback=callbacks) 
save_model_hdf5(ae_model,"C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/visualization/normal CAE/nolossweight/normalCAE.h5")
ae_model <- load_model_hdf5("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/visualization/normal CAE/nolossweight/normalCAE.h5")

summary(ae_model)
## encode test data----------------------
layer_name<-"max_pool4"
encoder <- keras_model(inputs=ae_model$input,outputs=get_layer(ae_model,layer_name)$output)
summary(encoder)
encoded_imgs = encoder %>% predict(xtest)
dim(encoded_imgs)#  10000     1     1    16

# decoder model
dec_input = layer_input(shape = c(1,1,16))
dec1<- get_layer(ae_model,name="decoder1")
up_samp1<- get_layer(ae_model,name="up_samp1")
dec2<- get_layer(ae_model,name="decoder2")
up_samp2<- get_layer(ae_model,name="up_samp2")
dec3<- get_layer(ae_model,name="decoder3")
up_samp3<- get_layer(ae_model,name="up_samp3")
dec4<- get_layer(ae_model,name="decoder4")
up_samp4<- get_layer(ae_model,name="up_samp4")
dec5<- get_layer(ae_model,name="autoencoder")
decoder <- keras_model(dec_input,dec5(up_samp4(dec4(up_samp3(dec3(up_samp2(dec2(up_samp1(dec1(dec_input))))))))))
summary(decoder)
decoded_imgs = decoder %>% predict(encoded_imgs)# 10000    28    28     1

# calculating reconstruction error (mse)
reconstruction_error =metric_mean_squared_error(xtest,decoded_imgs)
paste("reconstruction error: "
      ,k_get_value(k_mean(reconstruction_error))) 

recon_error <- k_get_value(k_mean(reconstruction_error))
## reconstruct -----------------------------------------------------
## AE reconstruct error 
decoded_imgs = decoder %>% predict(encoded_imgs) #10000   784
error <- rowMeans((decoded_imgs - xtest)**2) 
error <- data.table(error)
error 

## classification error(細部檢查結果)
eval <- data.frame(error=error, class = as.factor(y_test))

matrix<- eval %>% group_by(class) %>%summarise(avg_error = mean(error))#各種數字的平均error
avg_error <- as.numeric(matrix$avg_error) 
sum(avg_error)/10 
#plot error
eval %>% group_by(class) %>% summarise(avg_error = mean(error)) %>%
  ggplot(aes(x = class, y = avg_error, fill = class)) +
  geom_col() + ggtitle("Reconstruction error") + 
  theme_minimal()+theme(plot.title = element_text(size=17,hjust = 0.5,face="bold"))

# visualization 
dim(xtest) <- c(nrow(xtest), 28, 28)
dim(decoded_imgs) <- c(nrow(decoded_imgs), 28, 28)
n = 30
op = par(mfrow=c(5,2), mar=c(1,0,0,0))
for (i in 1:n) 
{
  plot(as.raster(xtest[i,,]))
  plot(as.raster(decoded_imgs[i,,]))
}

# feature distribution ------------------------------------------------
library(ggplot2)
library(dplyr)
batch_size=128
x_test_encoded <- predict(encoder, xtest, batch_size = batch_size)
dim(x_test_encoded)
x_test_encoded %>%
  as_data_frame() %>%
  mutate(class = as.factor(fashion_mnist$test$y)) %>%
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point()+
  ggtitle('Feature distribution')+
  theme(plot.title = element_text(size=17,hjust = 0.5,face="bold"))

## latent visualization------------------------------------------------
dim(encoded_imgs) #10000     1     1    16
encoded_matrix <- array_reshape(encoded_imgs, c(nrow(encoded_imgs),16), order = "F")
library(corrplot)
library(reshape2)
library(ggplot2)
## correlation 分析----------------------------------------------------
cormat <- round(cor(encoded_matrix),2)
head(cormat)
melted_cormat <- melt(cormat)
head(melted_cormat)
# ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
#   geom_tile()

# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}

upper_tri <- get_upper_tri(cormat)
upper_tri
# Melt the correlation matrix
library(reshape2)
melted_cormat <- melt(upper_tri, na.rm = TRUE)
# Heatmap
library(ggplot2)
ggheatmap<- ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", name="Pearson\nCorrelation") +
  theme_gray()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1))+coord_fixed()
ggheatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
  theme(legend.justification = c(1, 0),
        legend.position = c(0.6, 0.7),
        legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,title.position = "top", title.hjust = 0.5))
### pca 分析--------------------------------------------------------------------------------------
encoded.eigen<-eigen(cormat)
encoded.eigen$values/(sum(encoded.eigen$values))
encoded_pca <-princomp(encoded_matrix, cor=TRUE, scores=TRUE)
summary(encoded_pca)
loadings(encoded_pca)
screeplot(encoded_pca)
pcscores <-as.data.frame(encoded_pca$scores)
head(pcscores)

PC<-cbind(encoded_matrix, pcscores)
### 分布圖 ### 
## lattice method ------------------------
col <- 10
library(lattice)
key.species <- list(title="numbers",
                    space="right",
                    text=list(levels(as.factor(fashion_mnist$test$y))),
                    points=list(pch=20,colour=class)) 


xyplot(Comp.2~Comp.1, groups=y_test, data=PC,colour=class,
       main=list("PCA for AE latent representation ", cex=1.6),pch=20,key=key.species)

## ggplot2 -----------------------------
PC %>%
  mutate(class = as.factor(fashion_mnist$test$y)) %>%
  ggplot(aes(x =Comp.1, y =Comp.2, colour = class)) + geom_point()+
  ggtitle('PCA for latent representation(reconstruction CAE) ')+
  theme_bw()+
  theme(legend.title = element_text(size=12,face="bold"),
        legend.text = element_text(size=10,face="bold"),
        plot.title = element_text(size=17,hjust = 0.5,face="bold"))


####-------------------------------------
pca = prcomp(encoded_matrix, center=T, scale.=T)
## eigenvalues
round(pca$sdev^2, 2)
pcs = data.frame(pca$x)
str(pcs)
## percentage pcs 
cov = round(pca$sdev^2/sum(pca$sdev^2)*100, 2)
cov = data.frame(c(1:16),cov)
names(cov)[1] = 'PCs'
names(cov)[2] = 'Variance'
sum(cov$Variance[1:2])
sum(cov$Variance[1:4])
sum(cov$Variance[1:5])


PCA = pca$sdev^2
names(PCA) = paste0('PC', cov$PCs)
qcc::pareto.chart(PCA)
## sum of PC12 
sum(cov$Variance[1:2])
# PC123
sum(cov$Variance[1:3])

pcs%>%
  mutate(class = as.factor(fashion_mnist$test$y)) %>%
  ggplot(aes(x =PC1, y =PC2, colour = class)) + geom_point()+
  ggtitle('PCA for Latent Representation')+
  theme_bw()+theme(legend.title = element_text(size=12,face="bold"),
                   legend.text = element_text(size=12),
                   plot.title = element_text(size=17,hjust = 0.5,face="bold"))+
  guides(colour = guide_legend(override.aes = list(size=2, stroke=1.5))) 

p1<-pcs%>%
  mutate(class = as.factor(fashion_mnist$test$y)) %>%
  ggplot(aes(x =PC1, y =PC2, colour = class)) + geom_point()+
  theme_bw()+theme(legend.title = element_text(size=12,face="bold"),
                   legend.text = element_text(size=10,face="bold"),
                   plot.title = element_text(size=17,hjust = 0.5,face="bold"))
p2<-pcs%>%
  mutate(class = as.factor(fashion_mnist$test$y)) %>%
  ggplot(aes(x =PC2, y =PC3, colour = class)) + geom_point()+
  theme_bw()+theme(legend.title = element_text(size=12,face="bold"),
                   legend.text = element_text(size=10,face="bold"),
                   plot.title = element_text(size=17,hjust = 0.5,face="bold"))
p3<-pcs%>%
  mutate(class = as.factor(fashion_mnist$test$y)) %>%
  ggplot(aes(x =PC1, y =PC3, colour = class)) + geom_point()+
  theme_bw()+theme(legend.title = element_text(size=12,face="bold"),
                   legend.text = element_text(size=10,face="bold"),
                   plot.title = element_text(size=17,hjust = 0.5,face="bold"))
p4<-pcs%>%
  mutate(class = as.factor(fashion_mnist$test$y)) %>%
  ggplot(aes(x =PC1, y =PC4, colour = class)) + geom_point()+
  theme_bw()+theme(legend.title = element_text(size=12,face="bold"),
                   legend.text = element_text(size=10,face="bold"),
                   plot.title = element_text(size=17,hjust = 0.5,face="bold"))
grid.arrange(p1, p2, p3, p4, ncol=2)


## encode test data----------------------
layer_name<-"max_pool4"
encoder <- keras_model(inputs=ae_model$input,outputs=get_layer(ae_model,layer_name)$output)
summary(encoder)
encoded_imgs = encoder %>% predict(xtest)
dim(encoded_imgs)#  10000     1     1    16
encoded_train = encoder %>% predict(xtrain)
################################################################################
# classify_model
model <- keras_model_sequential()
model%>%
  layer_flatten()%>%
  layer_dense(units=16,activation="relu",name="class1")%>%
  layer_dense(units=64,activation="relu",name="class2")%>%
  layer_dense(units=16,activation="relu",name="class3")%>%
  layer_dense(units=10,activation="softmax",name="classification")
model%>% compile(optimizer="RMSprop",loss="categorical_crossentropy",metric="accuracy")
model %>% fit(encoded_train,trainY,batch_size = 128,epochs = 25,validation_data=list(encoded_imgs,testY))
save_model_hdf5(model,"C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/visualization/normal CAE/nolossweight/ae_classify.h5")


