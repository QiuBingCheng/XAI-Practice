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


#### build model ------------------------------------------------------------------------------------
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

## classifier (logistic regression)
classify <- encoder$output%>%
  layer_flatten()%>%
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


# training model -----------------------------------------------------------------------------------------
model%>% compile(optimizer="RMSprop",
                 loss=list("classification"="categorical_crossentropy","autoencoder"="mse")
                 ,metric=list("classification"="accuracy"))
callbacks = list(
  callback_model_checkpoint("checkpoints.h5"),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1))


model %>% fit(x= xtrain, y=list("classification"= trainY,"autoencoder"= xtrain),
              validation_data=list(x=xtest,y=list("classification"=testY,"autoencoder"=xtest)),
              batch_size = 128,epochs = 25,callback=callbacks)
save_model_hdf5(model,"C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/visualization/classification CAE/nolossweight/LRcae.h5")

save_model_hdf5(model,"C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/1203train different feature/cae_feature12.h5")

cae_model <- load_model_hdf5("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/visualization/classification CAE/nolossweight/LRcae.h5")
summary(cae_model)


## encode test data----------------------
layer_name<-"max_pool4"
encoder <- keras_model(inputs=cae_model$input,outputs=get_layer(cae_model,layer_name)$output)
summary(encoder)
encoded_imgs = encoder %>% predict(xtest)
dim(encoded_imgs)#  10000     1     1    16 (壓縮feature)

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
decoded_imgs = decoder %>% predict(encoded_imgs)# 10000    28    28     1 (還原影像)
# predict classification -------------------------------------
layer_name<-"classification"
classify <- keras_model(inputs=cae_model$input,outputs=get_layer(cae_model,layer_name)$output)
summary(classify)
classify_pred = classify %>% predict(xtest)#  10000    10
round(classify_pred , digits = 3)

library(ramify)
c<-argmax(classify_pred, rows = TRUE)
data.table(c)
categories <- c("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                "Sneaker", "Bag", "Boot")
pred <- factor(c, labels=categories)
actual <- factor (y_test, labels = categories)
confusion<- caret::confusionMatrix(as.factor(pred), as.factor(actual ))
table <- data.frame(confusionMatrix(pred, actual)$table)
# quantity
ggplot(table, aes(x=Prediction, y=Reference, fill=Freq),width=200,height=200) +
  geom_tile() + theme_bw() + coord_equal() +
  scale_fill_distiller(palette="Blues", direction=1) +
  guides(fill=F, title.position = "top", title.hjust = 0.5) + # removing legend for `fill`
  labs(title = "Confusion matrix")+ # using a title instead
  geom_text(aes(label=Freq), color="black",face="bold")+  # printing values
  theme(plot.title = element_text(size=17,hjust = 0.5,face="bold"),
        legend.text = element_text(size=10,face="bold"),
        legend.title= element_text(size=10,face="bold"))

##### check feature coefficient significant --------------------------------------------
weights <- cae_model$weights[[17]]
bias <- cae_model$weights[[18]]
matrix <- t(as.matrix(weights)) 
absolute_weights<- abs(matrix )#10 16
bias <- as.matrix(bias)

## weight 絕對值排列
absolute_weights
row <- c(1:16)
tmp <- list()
split_data <- split(absolute_weights, row(absolute_weights))

for( i in 1:10)
{
  row1 <- rbind(split_data[[i]],row)
  tmp<- rbind(tmp,row1[,order(row1[1,],decreasing = TRUE)])
}
tmp
for(j in 1:10)
{
  tmp <- tmp[-j,]
  sort_matrix<- tmp
}
sort_matrix

rownames(sort_matrix) <-  c("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                                        "Sneaker", "Bag", "Boot")
sort_matrix


sort_weights <- do.call(rbind, lapply(split(absolute_weights, row(absolute_weights)), sort, decreasing = TRUE))

### PCA ---------------------------------------------------------------------------------
dim(encoded_imgs) #10000     1     1    16
encoded_matrix <- array_reshape(encoded_imgs, c(nrow(encoded_imgs),16), order = "F")
library(corrplot)
library(reshape2)
library(ggplot2)

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
### our model
library(plotly)
categories <- c("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                "Sneaker", "Bag", "Boot")
actual <- factor (y_test, labels = categories)
research_pc<-pcs%>%
  mutate(class = actual) %>%
  ggplot(aes(x =PC1, y =PC2, colour = class)) + geom_point()+
  ggtitle('PCA for Latent Representation')+
  theme_bw()+theme(legend.title = element_text(size=15,face="bold"),
                   legend.text = element_text(size=15),
                   plot.title = element_text(size=17,hjust = 0.5,face="bold"))+
  guides(colour = guide_legend(override.aes = list(size=2, stroke=1.5)))

research_pc<-research_pc+
  scale_colour_discrete(labels = c("T-shirt", "Trouser", "Pullover", "Dress", 
                                   "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"))
research_pc
ggplotly(research_pc)










