library(tensorflow)
library(keras)
library(caret)
library(data.table)
library(dplyr)
library(ggplot2)
## fashion mnist data (x_train、x_test、trainY、testY) 
fashion_mnist <- dataset_fashion_mnist()
fashion_mnist$train$x
x_train <- fashion_mnist$train$x/255 #60000    28    28
x_test <- fashion_mnist$test$x/255 #10000    28    28
y_train <- fashion_mnist$train$y
y_test <- fashion_mnist$test$y
trainY<- to_categorical(y_train)#60000    10
testY<- to_categorical(y_test)#60000    10

##################################### Normal AE ##############################################
enc_input <- layer_input(shape=c(28,28))
enc_output<- enc_input %>% 
  layer_conv_1d(64,kernel_size=3,dilation_rate=1,padding="causal",strides=1,activation="relu",name="conv_1")%>%
  layer_max_pooling_1d(2,padding="same",name="max_1")%>%
  layer_conv_1d(32,kernel_size=3,dilation_rate=1,padding="causal",strides=1,activation="relu",name="conv_2")%>%
  layer_max_pooling_1d(2,padding="same",name="max_2")%>%
  layer_conv_1d(32,kernel_size=3,dilation_rate=1,padding="causal",strides=1,activation="relu",name="conv_3")%>%
  layer_max_pooling_1d(3,padding="same",name="max_3")%>%
  layer_conv_1d(16,kernel_size=3,dilation_rate=1,padding="causal",strides=1,activation="relu",name="conv_4")%>%
  layer_max_pooling_1d(3,padding="same",name="max_4")
encoder <- keras_model(enc_input,enc_output)
summary(encoder)  

decoder<-encoder$output%>%
  layer_conv_1d(16,kernel_size=3,dilation_rate=1,padding="causal",strides=1,activation="relu",name="deconv_1")%>%
  layer_upsampling_1d(3,name="up_samp1")%>%
  layer_conv_1d(32,kernel_size=3,dilation_rate=1,padding="causal",strides=1,activation="relu",name="deconv_2")%>%
  layer_upsampling_1d(3,name="up_samp2")%>%
  layer_conv_1d(32,kernel_size=3,dilation_rate=1,padding="valid",strides=1,activation="relu",name="deconv_3")%>%
  layer_upsampling_1d(2,name="up_samp3")%>%
  layer_conv_1d(64,kernel_size=3,dilation_rate=1,padding="causal",strides=1,activation="relu",name="deconv_4")%>%
  layer_upsampling_1d(2,name="up_samp4")%>%
  layer_conv_1d(28,kernel_size=3,dilation_rate=1,padding="causal",strides=1,activation="sigmoid",name="autoencoder")
 
AE_TCN <- keras_model(encoder$input,decoder)
summary(AE_TCN)

# training model ---------------------------------------------------------
AE_TCN %>% compile(optimizer="RMSprop",loss="mse")
AE_TCN %>% fit(x_train,x_train,validation_data=list(x_test,x_test), batch_size = 128,epochs = 25)
# Epoch 25/25
# loss: 0.0169 - val_loss: 0.0189
save_model_hdf5(AE_TCN,"C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/visualization/TCN/normal_AE_TCN.h5")
normal_AE <- load_model_hdf5("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/visualization/TCN/normal_AE_TCN.h5")
## encode test data----------------------

layer_name<-"max_4"
encoder <- keras_model(inputs=normal_AE$input,outputs=get_layer(normal_AE,layer_name)$output)
summary(encoder)
encoded_imgs = encoder %>% predict(x_test)
dim(encoded_imgs)#  10000     1         16

#decoder model
dec_input = layer_input(shape = c(1,16))
dec1<- get_layer(normal_AE,name="deconv_1")
up_samp1<- get_layer(normal_AE,name="up_samp1")
dec2<- get_layer(normal_AE,name="deconv_2")
up_samp2<- get_layer(normal_AE,name="up_samp2")
dec3<- get_layer(normal_AE,name="deconv_3")
up_samp3<- get_layer(normal_AE,name="up_samp3")
dec4<- get_layer(normal_AE,name="deconv_4")
up_samp4<- get_layer(normal_AE,name="up_samp4")
dec5<- get_layer(normal_AE,name="autoencoder")
decoder <- keras_model(dec_input,dec5(up_samp4(dec4(up_samp3(dec3(up_samp2(dec2(up_samp1(dec1(dec_input))))))))))
summary(decoder)
decoded_imgs = decoder %>% predict(encoded_imgs)# 10000     28    28
dim(decoded_imgs)

# calculating reconstruction error (mse)
reconstruction_error =metric_mean_squared_error(x_test,decoded_imgs)
paste("reconstruction error: "
      ,k_get_value(k_mean(reconstruction_error))) # 0.0182828447683835

recon_error <- k_get_value(k_mean(reconstruction_error))
## reconstruct -----------------------------------------------------
## AE reconstruct error 
decoded_imgs = decoder %>% predict(encoded_imgs) #10000   784
error <- rowMeans((decoded_imgs - x_test)**2) 
error <- data.table(error)
error 
## classification error(細部檢查結果)
eval <- data.frame(error=error, class = as.factor(y_test))

#eval %>% group_by(class) %>%summarise(avg_error = mean(error))
matrix<- eval %>% group_by(class) %>%summarise(avg_error = mean(error))#各種數字的平均error
avg_error <- as.numeric(matrix$avg_error) 
sum(avg_error)/10 
#plot error
eval %>% group_by(class) %>% summarise(avg_error = mean(error)) %>%
  ggplot(aes(x = class, y = avg_error, fill = class)) +
  geom_col() + ggtitle("Reconstruction error") + 
  theme_minimal()+theme(plot.title = element_text(size=17,hjust = 0.5,face="bold"))

# visualization 
dim(x_test) <- c(nrow(x_test), 28, 28)
dim(decoded_imgs) <- c(nrow(decoded_imgs), 28, 28)
n = 30
op = par(mfrow=c(5,2), mar=c(1,0,0,0))
for (i in 1:n) 
{
  plot(as.raster(x_test[i,,]))
  plot(as.raster(decoded_imgs[i,,]))
}
## latent visualization------------------------------------------------
dim(encoded_imgs) #10000     1     1    16
encoded_matrix <- array_reshape(encoded_imgs, c(nrow(encoded_imgs),16), order = "F")
library(corrplot)
library(reshape2)
library(ggplot2)
encoded_matrix
#### PCA---------------------------------------------------------------
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
sum(cov$Variance[1:2])# 50.82
sum(cov$Variance[1:4])#72.05
sum(cov$Variance[1:5])#78.29

PCA = pca$sdev^2
names(PCA) = paste0('PC', cov$PCs)
qcc::pareto.chart(PCA)
## sum of PC12 
sum(cov$Variance[1:2])#79.07
# PC123
sum(cov$Variance[1:3])#85.16

categories <- c("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                "Sneaker", "Bag", "Boot")
actual <- factor (y_test, labels = categories)
TCN_AE_PCA<- pcs%>%
  mutate(class = actual) %>%
  ggplot(aes(x =PC1, y =PC2, colour = class)) + geom_point()+
  ggtitle('PCA for Latent Representation')+
  theme_bw()+theme(legend.title = element_text(size=12,face="bold"),
                   legend.text = element_text(size=12),
                   plot.title = element_text(size=17,hjust = 0.5,face="bold"))+
  guides(colour = guide_legend(override.aes = list(size=2, stroke=1.5))) 

TCN_AE_PCA+scale_colour_discrete(labels = c("T-shirt", "Trouser", "Pullover", "Dress", 
                                   "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"))

### classification
model <- keras_model_sequential()
model%>%
  layer_flatten()%>%
  layer_dense(units=16,activation="relu",name="class1")%>%
  layer_dense(units=64,activation="relu",name="class2")%>%
  layer_dense(units=16,activation="relu",name="class3")%>%
  layer_dense(units=10,activation="softmax",name="classification")
model%>% compile(optimizer="RMSprop",loss="categorical_crossentropy",metric="accuracy")
model %>% fit(encoded_imgs,testY, batch_size = 128,epochs = 25)# loss: 0.5461 - accuracy: 0.8007
save_model_hdf5(model,"C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/visualization/TCN/normal_TCN_classify.h5")
summary(model)

ae_pred = model %>% predict(x_test)#  10000    10
#install.packages("ramify")
library(ramify)
c<-argmax(ae_pred, rows = TRUE)
data.table(c)
categories <- c("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                "Sneaker", "Bag", "Boot")
pred <- factor(c, labels=categories)
actual <- factor (y_test, labels = categories)
confusion<- caret::confusionMatrix(as.factor(pred), as.factor(actual ))
table <- data.frame(confusionMatrix(pred, actual)$table)
# Heatmap
# percentage
plotTable <- table %>%
  group_by(Reference)%>%
  mutate(prop = Freq/sum(Freq))

ggplot(plotTable, aes(x=Prediction, y=Reference, fill=prop),width=200,height=200) +
  geom_tile() + theme_bw() + coord_equal() +
  scale_fill_distiller(palette="Blues", direction=1) +
  guides(fill=F, title.position = "top", title.hjust = 0.5) + # removing legend for `fill`
  labs(title = "Confusion matrix")+ # using a title instead
  geom_text(aes(label=prop), color="black",face="bold")+  # printing values
  theme(plot.title = element_text(size=17,hjust = 0.5,face="bold"),
        legend.text = element_text(size=10,face="bold"),
        legend.title= element_text(size=10,face="bold"))

######################################## Research method ######################################################
#encoder 
enc_input <- layer_input(shape=c(28,28))
enc_output<- enc_input %>% 
  layer_conv_1d(64,kernel_size=3,dilation_rate=1,padding="causal",strides=1,activation="relu",name="conv_1")%>%
  layer_max_pooling_1d(2,padding="same",name="max_1")%>%
  layer_conv_1d(32,kernel_size=3,dilation_rate=1,padding="causal",strides=1,activation="relu",name="conv_2")%>%
  layer_max_pooling_1d(2,padding="same",name="max_2")%>%
  layer_conv_1d(32,kernel_size=3,dilation_rate=1,padding="causal",strides=1,activation="relu",name="conv_3")%>%
  layer_max_pooling_1d(3,padding="same",name="max_3")%>%
  layer_conv_1d(16,kernel_size=3,dilation_rate=1,padding="causal",strides=1,activation="relu",name="conv_4")%>%
  layer_max_pooling_1d(3,padding="same",name="max_4")
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
# decoder 
decoder<-encoder$output%>%
  layer_conv_1d(16,kernel_size=3,dilation_rate=1,padding="causal",strides=1,activation="relu",name="deconv_1")%>%
  layer_upsampling_1d(3,name="up_samp1")%>%
  layer_conv_1d(32,kernel_size=3,dilation_rate=1,padding="causal",strides=1,activation="relu",name="deconv_2")%>%
  layer_upsampling_1d(3,name="up_samp2")%>%
  layer_conv_1d(32,kernel_size=3,dilation_rate=1,padding="valid",strides=1,activation="relu",name="deconv_3")%>%
  layer_upsampling_1d(2,name="up_samp3")%>%
  layer_conv_1d(64,kernel_size=3,dilation_rate=1,padding="causal",strides=1,activation="relu",name="deconv_4")%>%
  layer_upsampling_1d(2,name="up_samp4")%>%
  layer_conv_1d(28,kernel_size=3,dilation_rate=1,padding="causal",strides=1,activation="sigmoid",name="autoencoder")

## full model (model)
model <- keras_model(inputs=enc_input,outputs=c(classify,decoder))
summary(model)

# training model -----------------------------------------------------------------------------------------
model%>% compile(optimizer="RMSprop",
                 loss=list("classification"="categorical_crossentropy","autoencoder"="mse")
                 ,metric=list("classification"="accuracy"))
callbacks = list(
  callback_model_checkpoint("checkpoints.h5"),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1))


model %>% fit(x= x_train, y=list("classification"= trainY,"autoencoder"= x_train),
              validation_data=list(x=x_test,y=list("classification"=testY,"autoencoder"=x_test)),
              batch_size = 128,epochs = 25,callback=callbacks)

save_model_hdf5(model,"C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/visualization/TCN/TCN_research.h5")
model <- load_model_hdf5("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/visualization/TCN/TCN_research.h5")
summary(model)


#decoder model
dec_input = layer_input(shape = c(1,16))
dec1<- get_layer(model,name="deconv_1")
up_samp1<- get_layer(model,name="up_samp1")
dec2<- get_layer(model,name="deconv_2")
up_samp2<- get_layer(model,name="up_samp2")
dec3<- get_layer(model,name="deconv_3")
up_samp3<- get_layer(model,name="up_samp3")
dec4<- get_layer(model,name="deconv_4")
up_samp4<- get_layer(model,name="up_samp4")
dec5<- get_layer(model,name="autoencoder")
decoder <- keras_model(dec_input,dec5(up_samp4(dec4(up_samp3(dec3(up_samp2(dec2(up_samp1(dec1(dec_input))))))))))
summary(decoder)
decoded_imgs = decoder %>% predict(encoded_imgs)# 10000     28    28

#calculating reconstruction error (mse)
encoded_imgs<- encoder %>% predict(x_test)
decoded_imgs <-decoder %>% predict(encoded_imgs)# 10000    28    28     1
reconstruction_error =metric_mean_squared_error(x_test,decoded_imgs)
paste("reconstruction error: "
      ,k_get_value(k_mean(reconstruction_error))) #0.0289637760598057"

recon_error <- k_get_value(k_mean(reconstruction_error))
## reconstruct -----------------------------------------------------
## AE reconstruct error 
error <- rowMeans((decoded_imgs - x_test)**2) 
error <- data.table(error)
error 
## classification error(細部檢查結果)
eval <- data.frame(error=error, class = as.factor(y_test))
#eval %>% group_by(class) %>%summarise(avg_error = mean(error))
matrix<- eval %>% group_by(class) %>%summarise(avg_error = mean(error))#各種數字的平均error
avg_error <- as.numeric(matrix$avg_error) 
sum(avg_error)/10 
#plot error
eval %>% group_by(class) %>% summarise(avg_error = mean(error)) %>%
  ggplot(aes(x = class, y = avg_error, fill = class)) +
  geom_col() + ggtitle("Reconstruction error") + 
  theme_minimal()+theme(plot.title = element_text(size=17,hjust = 0.5,face="bold"))

# visualization 
dim(x_test) <- c(nrow(x_test), 28, 28)
dim(decoded_imgs) <- c(nrow(decoded_imgs), 28, 28)
n = 30
op = par(mfrow=c(5,2), mar=c(1,0,0,0))
for (i in 1:n) 
{
  plot(as.raster(x_test[i,,]))
  plot(as.raster(decoded_imgs[i,,]))
}


## encode test data----------------------
layer_name<-"max_4"
encoder <- keras_model(inputs=model$input,outputs=get_layer(model,layer_name)$output)
summary(encoder)
encoded_imgs = encoder %>% predict(x_test)
dim(encoded_imgs)#  10000      1    16
## latent visualization------------------------------------------------
dim(encoded_imgs) #10000     1     1    16
encoded_matrix <- array_reshape(encoded_imgs, c(nrow(encoded_imgs),16), order = "F")
head(encoded_matrix)
library(corrplot)
library(reshape2)
library(ggplot2)
#### PCA---------------------------------------------------------------
pca = prcomp(encoded_matrix, center=T, scale.=T)
## eigenvalues
round(pca$sdev^2, 2)#[1] 4.02 3.29 2.29 1.34 1.03 0.95 0.80 0.67 0.48 0.31 0.25 0.17 0.15 0.10 0.09 0.07
pcs = data.frame(pca$x)
str(pcs)
## percentage pcs 
cov = round(pca$sdev^2/sum(pca$sdev^2)*100, 2)
cov = data.frame(c(1:16),cov)
names(cov)[1] = 'PCs'
names(cov)[2] = 'Variance'
sum(cov$Variance[1:2])# 45.69
sum(cov$Variance[1:4])# 68.37
sum(cov$Variance[1:5])# 74.78

PCA = pca$sdev^2
names(PCA) = paste0('PC', cov$PCs)
qcc::pareto.chart(PCA)
## sum of PC12 
sum(cov$Variance[1:2])#79.07
# PC123
sum(cov$Variance[1:3])#85.16

categories <- c("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                "Sneaker", "Bag", "Boot")
actual <- factor (y_test, labels = categories)
TCN_AE_PCA<- pcs%>%
  mutate(class = actual) %>%
  ggplot(aes(x =PC1, y =PC2, colour = class)) + geom_point()+
  ggtitle('PCA for Latent Representation')+
  theme_bw()+theme(legend.title = element_text(size=12,face="bold"),
                   legend.text = element_text(size=12),
                   plot.title = element_text(size=17,hjust = 0.5,face="bold"))+
  guides(colour = guide_legend(override.aes = list(size=2, stroke=1.5))) 

TCN_AE_PCA+scale_colour_discrete(labels = c("T-shirt", "Trouser", "Pullover", "Dress", 
                                            "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"))


## research model-------------------------------------
layer_name<-"classification"
classify <- keras_model(inputs=model$input,outputs=get_layer(model,layer_name)$output)
summary(classify)
classify_pred = classify %>% predict(x_test)#  10000    10
#install.packages("ramify")
library(ramify)
c<-argmax(classify_pred, rows = TRUE)
data.table(c)
categories <- c("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                "Sneaker", "Bag", "Boot")
pred <- factor(c, labels=categories)
actual <- factor (y_test, labels = categories)
confusion<- caret::confusionMatrix(as.factor(pred), as.factor(actual ))
table <- data.frame(confusionMatrix(pred, actual)$table)

# Heatmap
# percentage
plotTable <- table %>%
  group_by(Reference)%>%
  mutate(prop = Freq/sum(Freq))

ggplot(plotTable, aes(x=Prediction, y=Reference, fill=prop),width=200,height=200) +
  geom_tile() + theme_bw() + coord_equal() +
  scale_fill_distiller(palette="Blues", direction=1) +
  guides(fill=F, title.position = "top", title.hjust = 0.5,face="bold") +
  labs(title = "Confusion matrix")+ 
  geom_text(aes(label=prop), color="black",face="bold")+  
  theme(plot.title = element_text(size=20,hjust = 0.5,face="bold"),
        legend.text = element_text(size=10,face="bold"),
        legend.title= element_text(size=10,face="bold"))




