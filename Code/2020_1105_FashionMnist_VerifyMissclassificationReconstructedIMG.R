#################### check model classify #############################
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
x_train[1,,]
round(x_train[1,,],digits = 3)
xtrain <- array_reshape(x_train, dim=c(dim(x_train)[1],dim(x_train)[2],dim(x_train)[3],1)) #60000    28    28     1
xtest <- array_reshape(x_test, dim=c(dim(x_test)[1],dim(x_test)[2],dim(x_test)[3],1))#10000    28    28     1
## input data (xtrain¡Bxtest¡BtestY¡BtrainY)
xtrain <- readRDS("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/xtrain")
xtest <- readRDS("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/xtest")
testY <- readRDS("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/testY")
trainY <- readRDS("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/trainY")
cae_model <- load_model_hdf5("C:/Users/cindy chu/Desktop/109-1/research/Fashion_mnist/visualization/classification CAE/nolossweight/cae4.h5")
summary(cae_model)

## research model-------------------------------------
layer_name<-"classification"
classify <- keras_model(inputs=cae_model$input,outputs=get_layer(cae_model,layer_name)$output)
summary(classify)
classify_pred = classify %>% predict(xtest)#  10000    10

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
  guides(fill=F, title.position = "top", title.hjust = 0.5) + 
  labs(title = "Confusion matrix")+ 
  geom_text(aes(label=prop), color="black",face="bold")+  
  theme(plot.title = element_text(size=17,hjust = 0.5,face="bold"),
        legend.text = element_text(size=10,face="bold"),
        legend.title= element_text(size=10,face="bold"))

# quantity
ggplot(table, aes(x=Prediction, y=Reference, fill=Freq),width=200,height=200) +
  geom_tile() + theme_bw() + coord_equal() +
  scale_fill_distiller(palette="Blues", direction=1) +
  guides(fill=F, title.position = "top", title.hjust = 0.5) + 
  labs(title = "Confusion matrix")+ 
  geom_text(aes(label=Freq), color="black",face="bold")+  
  theme(plot.title = element_text(size=17,hjust = 0.5,face="bold"),
        legend.text = element_text(size=10,face="bold"),
        legend.title= element_text(size=10,face="bold"))

# view predict True and False
pred <- factor(c, labels=categories)
actual <- factor (y_test, labels = categories)
tmp <- cbind(actual,pred)
tmp <- as.data.frame(tmp)
tmp$actual


########################## Shirt ##########################################################
#### prediction
label_predict <- data.frame(pred)
label_actual <- data.frame(actual )
col_seq <- seq(from=1,to=10000,by=1)
label_matrix <- cbind(col_seq,label_actual,label_predict)

shirt <- label_matrix %>% filter(label_matrix$actual=="Shirt")

falsepredict_tshirt <- shirt %>% filter(shirt$pred=="T-shirt") %>% select(col_seq) #141   1
falsepredict_trouser <- shirt %>% filter(shirt$pred=="Trouser") %>% select(col_seq)
falsepredict_pullover <- shirt %>% filter(shirt$pred=="Pullover") %>% select(col_seq)
falsepredict_dress <- shirt %>% filter(shirt$pred=="Dress") %>% select(col_seq)
falsepredict_coat <- shirt %>% filter(shirt$pred=="Coat") %>% select(col_seq)
falsepredict_bag<- shirt %>% filter(shirt$pred=="Bag") %>% select(col_seq)

## t_shirt
tshirt <- shirt %>% filter(shirt$pred=="T-shirt")
color <- '#bb0000'
op = par(mfrow=c(3,5), mar=c(1,0,0,0))
for(i in 1L:141L)
{
  j <- falsepredict_tshirt[i,]
  predict <- as.matrix(tshirt$pred[i])
  actual <- as.matrix(tshirt$actual[i])
  #plot(as.raster(decoded_imgs[j,,]))+
  plot(as.raster(x_test[j,,]))+
  #title(paste0(actual,"(",predict,")"), cex.main=1,line=-2,col="red")
  title(paste0(actual,"(",predict,")"), cex.main=1,line=-1,col="red")
}

## trouser
trouser <- shirt %>% filter(shirt$pred=="Trouser") 
color <- '#bb0000'
op = par(mfrow=c(3,5), mar=c(1,0,0,0))
for(i in 1L:1L)
{
  j <- falsepredict_trouser[i,]
  predict <- as.matrix(trouser$pred[i])
  actual <- as.matrix(trouser$actual[i])
  plot(as.raster(x_test[j,,]))+
    title(paste0(actual,"(",predict,")"), cex.main=1,line=-1,col="red")
}

## Pullover
pull <- shirt %>% filter(shirt$pred=="Pullover") 
color <- '#bb0000'
op = par(mfrow=c(3,5), mar=c(1,0,0,0))
for(i in 1L:dim(falsepredict_pullover)[1])
{
  j <- falsepredict_pullover[i,]
  predict <- as.matrix(pull$pred[i])
  actual <- as.matrix(pull$actual[i])
  plot(as.raster(x_test[j,,]))+
  title(paste0(actual,"(",predict,")"), cex.main=1,line=-1,col="red")
}

## Dress
dress <- shirt %>% filter(shirt$pred=="Dress") 
color <- '#bb0000'
op = par(mfrow=c(3,5), mar=c(1,0,0,0))
for(i in 1L:dim(falsepredict_dress)[1])
{
  j <- falsepredict_dress[i,]
  predict <- as.matrix(dress$pred[i])
  actual <- as.matrix(dress$actual[i])
  plot(as.raster(x_test[j,,]))+
  title(paste0(actual,"(",predict,")"), cex.main=1,line=-1,col="red")
}

## Coat
coat <- shirt %>% filter(shirt$pred=="Coat") 
color <- '#bb0000'
op = par(mfrow=c(3,5), mar=c(1,0,0,0))
for(i in 1L:dim(falsepredict_coat)[1])
{
  j <- falsepredict_coat[i,]
  predict <- as.matrix(coat$pred[i])
  actual <- as.matrix(coat$actual[i])
  plot(as.raster(x_test[j,,]))+
  title(paste0(actual,"(",predict,")"), cex.main=1,line=-1,col="red")
}
## Bag
bag <- shirt %>% filter(shirt$pred=="Bag")
color <- '#bb0000'
op = par(mfrow=c(3,5), mar=c(1,0,0,0))
for(i in 1L:dim(falsepredict_bag)[1])
{
  j <- falsepredict_bag[i,]
  predict <- as.matrix(bag$pred[i])
  actual <- as.matrix(bag$actual[i])
  plot(as.raster(x_test[j,,]))+
  title(paste0(actual,"(",predict,")"), cex.main=1,line=-1,col="red")
}

### non normalize  -----------------------------------------------------------

fashion_mnist <- dataset_fashion_mnist()
fashion_mnist$train$x
x_train <- fashion_mnist$train$x #60000    28    28
x_test <- fashion_mnist$test$x #10000    28    28
y_train <- fashion_mnist$train$y
y_test <- fashion_mnist$test$y

library(reshape2)
library(plyr)
subarray <- apply(x_test[1:10, , ], 1, as.data.frame)
subarray <- lapply(subarray, function(df){
  colnames(df) <- seq_len(ncol(df))
  df['y'] <- seq_len(nrow(df))
  df <- melt(df, id = 'y')
  return(df)
})
cloth_cats = data.frame(category = c('Top', 'Trouser', 'Pullover', 'Dress', 'Coat',  
                                     'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'), 
                        label = seq(0, 9))
plotdf <- rbind.fill(subarray)
first_ten_labels <- cloth_cats$category[match(y_test[1:10], cloth_cats$label)]
first_ten_categories <- paste0('Image ', 1:10, ': ', first_ten_labels)
plotdf['Image'] <- factor(rep(first_ten_categories, unlist(lapply(subarray, nrow))), 
                          levels = unique(first_ten_categories))


library(ggplot2)
ggplot() + 
  geom_raster(data = plotdf, aes(x = variable, y = y, fill = value)) + 
  facet_wrap(~ Image, nrow = 5, ncol = 2) + 
  scale_fill_gradient(low = "white", high = "black", na.value = NA) + 
  theme(aspect.ratio = 1, legend.position = "none") + 
  labs(x = NULL, y = NULL) + 
  scale_x_discrete(breaks = seq(0, 28, 7), expand = c(0, 0)) + 
  scale_y_reverse(breaks = seq(0, 28, 7), expand = c(0, 0))

#####################################################################################

label_actual <- data.frame(actual )
col_seq <- seq(from=1,to=10000,by=1)
label_matrix <- cbind(col_seq,label_actual)
value <- label_matrix %>% filter(label_matrix$actual=="Shirt")
head(value)
dim(xtest) <- c(nrow(xtest), 28, 28)
plot(as.raster(xtest[8,,]))

categories <- c("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                "Sneaker", "Bag", "Boot")







