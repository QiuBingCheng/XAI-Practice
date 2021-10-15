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
dim(x_train)[1]
xtrain <- array_reshape(x_train, dim=c(dim(x_train)[1],dim(x_train)[2],dim(x_train)[3],1)) #60000    28    28     1
xtest <- array_reshape(x_test, dim=c(dim(x_test)[1],dim(x_test)[2],dim(x_test)[3],1))#10000    28    28     1
## input data (xtrain、xtest、testY、trainY)
xtrain <- readRDS("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/xtrain")
xtest <- readRDS("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/xtest")
testY <- readRDS("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/testY")
trainY <- readRDS("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/trainY")
##-------------------------------------------------------------------------------------------------------
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
# training model -----------------------------------------------------------------------------------------
cnn_model%>% compile(optimizer="RMSprop",loss="categorical_crossentropy",metric=list("classification"="accuracy"))
callbacks = list(
  callback_model_checkpoint("checkpoints.h5"),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1))
#callbacks <- callback_early_stopping(monitor="val_loss",patience = 5)

cnn_model %>% fit(x= xtrain, y= trainY,validation_data=list(x=xtest,y=testY),batch_size = 128,epochs =30,callback=callbacks)
save_model_hdf5(cnn_model,"C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/visualization/classification CAE/nolossweight/cnn_model.h5")
cnn_model <- load_model_hdf5("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/visualization/classification CAE/nolossweight/cnn_model.h5")





layer_name<-"max_pool4"
encoder <- keras_model(inputs=model$input,outputs=get_layer(model,layer_name)$output)
summary(encoder)
## latent visualization------------------------------------------------
encoded_imgs <- encoder %>% predict(xtest)
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
sum(cov$Variance[1:2])#39.72
# PC123
sum(cov$Variance[1:3])#53.34




pcs%>%
  mutate(class = as.factor(fashion_mnist$test$y)) %>%
  ggplot(aes(x =PC1, y =PC2, colour = class)) + geom_point()+
  ggtitle('PCA for Latent Representation(classification CNN) ')+
  theme_bw()+theme(legend.title = element_text(size=12,face="bold"),
                   legend.text = element_text(size=10,face="bold"),
                   plot.title = element_text(size=17,hjust = 0.5,face="bold"))
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
















