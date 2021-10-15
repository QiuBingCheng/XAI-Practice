###################################################################################################################
#################################### 分群方法　####################################################################
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

xtrain<-read_rds("C:/Users/User/Desktop/2021_0416/dataset/normalize/total_xtrain.rds")
y_removal<-read_rds("C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/only_scale/trainy.rds")
y_train_class<-read_rds("C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/only_scale/trainy_class.rds")

xtrain<-read_rds("C:/Users/User/Desktop/2021_0415/dataset/normalize/total_xtrain.rds")
y_removal<-read_rds("C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/only_scale/trainy.rds")
y_train_class<-read_rds("C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/only_scale/trainy_class.rds")
ytrain<- data.frame(y_removal)
order <- ytrain[order(ytrain$AVG_REMOVAL_RATE,decreasing = T),]
defect <- order[1:4,]
# WAFER_ID STAGE AVG_REMOVAL_RATE
# 311 2058207580     A         4326.154
# 195 1834206730     A         4202.112
# 197 1834206944     A         4182.417
# 200 1834206972     A         4129.494
y <- ytrain[-as.numeric(defect$number),]#364   3
x <- xtrain[-as.numeric(defect$number),,]

train_class <- y_train_class[-as.numeric(defect$number),]
dim(x) #1977  316   19
dim(y) # 1977    4

#### 更改(0421: 只有12SVID分群) ----------------------------------------------
var <- c(2,3,5,6,7,8,9,12,15,16,17,19)
tmp <- x[,,var]
x <- tmp
dim(x)

#### 更改(0421 :參考論文6SVID分群) -----------------------------------------------
var <- c(c(1:4),c(10,11))
tmp <- x[,,var]
x<-tmp
dim(x)
###Corr + K-means  --------------------------------------------------------------
correlation <- list()
cor_data <- list()
cor_uppertrain<- list()
new_set<-list()

for(i in 1: dim(x)[1])
{
  
  tmp <- x[i,,]
  colnames(tmp) <- paste0("V",seq(1,ncol(tmp)))
  cor_matrix <- cor(tmp,method = "pearson")
  cor_matrix[is.na(cor_matrix)]=0
  cor_matrix <- round(cor_matrix , digits = 4)
  new_set[[i]]<-cor_matrix
  melted_cormat <- melt(cor_matrix)
  # Get upper triangle of the correlation matrix
  get_upper_tri <- function(cor_matrix){
    cor_matrix[lower.tri(cor_matrix)]<- NA
    return(cor_matrix)
  }
  
  upper_tri <- get_upper_tri(cor_matrix)
  correlation[[i]]<- upper_tri
  
  #melt <- melt(upper_tri)
  # melt2<- melt[complete.cases(melt), ]
  ### 修改:
  melt <- melt(upper_tri)
  order <- melt[order(melt$Var1,decreasing = F),]
  melt2<- order[complete.cases(order), ]
  
  set <- c()
  for(j in 1:nrow(melt2))
  {
    if(melt2[j,1]!= melt2[j,2])
    {
      set <- rbind(set,melt2[j,])
    }
    
  }
  cor_data[[i]]<- set[,3]
  cor_uppertrain[[i]]<- melt2
} 

cor_data <- abind(cor_data , along = 0) # 1977 171
correlation_set<- abind(new_set,along=0) #1977   19   19

# saveRDS(cor_data,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/total_kmeansdata.rds")
# saveRDS(correlation_set,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/total_corrdata.rds")

# saveRDS(cor_data,"C:/Users/User/Desktop/2021_0415/dataset/code/total_kmeansdata.rds")
# saveRDS(correlation_set,"C:/Users/User/Desktop/2021_0415/dataset/code/total_corrdata.rds")

saveRDS(cor_data,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/total_kmeansdata.rds")
saveRDS(correlation_set,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/total_corrdata.rds")
###### ---------------------------------------------------------------
# cor_data<- read_rds("C:/Users/User/Desktop/0415_data_preprocessing/dataset/code/total_kmeansdata.rds")
# cor_data<- read_rds("C:/Users/User/Desktop/2021_0416/dataset/finaldataset/total_kmeansdata.rds")
cor_data<- read_rds("C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/total_kmeansdata.rds")

kmeans.cluster <- Kmeans(cor_data,centers=2,nstart=25,method="euclidean")
kmeans.cluster2 <- Kmeans(cor_data,centers=2,nstart=25,method="abspearson") ## 僅需兩群

str(kmeans.cluster)
str(kmeans.cluster_test)
library(factoextra)
plot <-fviz_cluster(kmeans.cluster2,           # 分群結果
                    data = cor_data,              # 資料
                    geom = c("point","text"), # 點和標籤(point & label)
                    frame.type = "norm")      # 框架型態
print(plot)



saveRDS(kmeans.cluster,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/euclidean_cluster.rds")
saveRDS(kmeans.cluster2,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/abspearson_cluster.rds")

saveRDS(kmeans.cluster,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/euclidean_cluster.rds")
saveRDS(kmeans.cluster2,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/abspearson_cluster.rds")


# saveRDS(kmeans.cluster,"C:/Users/User/Desktop/0415_data_preprocessing/dataset/code/euclidean_cluster.rds")
# saveRDS(kmeans.cluster2,"C:/Users/User/Desktop/0415_data_preprocessing/dataset/code/abspearson_cluster.rds")

#############################################
##### 之前correlation 1977 *171 更改不影響結果！
# kmeans.cluster<- read_rds("C:/Users/User/Desktop/0415_data_preprocessing/dataset/code/euclidean_cluster.rds")
# kmeans.cluster2<- read_rds("C:/Users/User/Desktop/0415_data_preprocessing/dataset/code/abspearson_cluster.rds")
# cor_data <- read_rds("C:/Users/User/Desktop/0415_data_preprocessing/dataset/code/total_kmeansdata.rds")
# cor_train <- read_rds("C:/Users/User/Desktop/0415_data_preprocessing/dataset/code/total_corrdata.rds")

kmeans.cluster<- read_rds("C:/Users/User/Desktop/2021_0416/dataset/finaldataset/euclidean_cluster.rds")
kmeans.cluster2<- read_rds("C:/Users/User/Desktop/2021_0416/dataset/finaldataset/abspearson_cluster.rds")
cor_data <- read_rds("C:/Users/User/Desktop/2021_0416/dataset/finaldataset/total_kmeansdata.rds")
cor_train <- read_rds("C:/Users/User/Desktop/2021_0416/dataset/finaldataset/total_corrdata.rds")


#### 發現第293片SVID全部獨立
tmp <- data.frame(seq(1,nrow(cor_data),by=1),cor_data)
tmp <- data.frame(tmp,data.frame(colSums(t(cor_data))))

tmp%>%
  filter(colSums.t.cor_data..=="0")
cor_train[293,,]
## EUC ----------------------------------------------------------------------------------------
## train

class <- data.frame(kmeans.cluster$cluster)
class1 <- cbind(seq(1,nrow(class),by=1),class,y$AVG_REMOVAL_RATE)
colnames(class1)<-c("number","class","value")
set1 <- class1 %>% filter(class==1)
set2 <- class1 %>% filter(class==2)

summary(set2)
# number           class       value       
# Min.   :   2.0   Min.   :1   Min.   : 53.65  
# 1st Qu.: 257.8   1st Qu.:1   1st Qu.: 74.43  
# Median : 722.5   Median :1   Median : 83.40  
# Mean   : 826.5   Mean   :1   Mean   :102.87  
# 3rd Qu.:1359.8   3rd Qu.:1   3rd Qu.:148.98  
# Max.   :1974.0   Max.   :1   Max.   :162.56   

tmp <- set1 %>% mutate(dataset = ifelse(value<=74.43, "0", ifelse(value>74.43 & value<148.98 , "1",ifelse(value>=148.98,"2",0))))

# number         class       value       
# Min.   :   1   Min.   :2   Min.   : 53.43  
# 1st Qu.: 648   1st Qu.:2   1st Qu.: 71.54  
# Median :1088   Median :2   Median : 77.70  
# Mean   :1078   Mean   :2   Mean   : 83.43  
# 3rd Qu.:1528   3rd Qu.:2   3rd Qu.: 85.35  
# Max.   :1977   Max.   :2   Max.   :162.64

tmp <- set2 %>% mutate(dataset = ifelse(value<=71.54, "0", ifelse(value>71.54 & value<85.35 , "1",ifelse(value>=85.35,"2",0))))


## ABS ----------------------------------------------------------------------------------------
## train

class <- data.frame(kmeans.cluster2$cluster)
class2 <- cbind(seq(1,nrow(class),by=1),class,y$AVG_REMOVAL_RATE)
colnames(class2)<-c("number","class","value")
set1 <- class2 %>% filter(class==1)
set2 <- class2 %>% filter(class==2)
summary(set2)
# number           class       value       
# Min.   :   7.0   Min.   :1   Min.   : 53.65  
# 1st Qu.: 429.2   1st Qu.:1   1st Qu.: 72.96  
# Median : 954.5   Median :1   Median : 79.50  
# Mean   : 969.0   Mean   :1   Mean   : 91.86  
# 3rd Qu.:1492.8   3rd Qu.:1   3rd Qu.: 89.38  
# Max.   :1974.0   Max.   :1   Max.   :161.07     
tmp <- set1 %>% mutate(dataset = ifelse(value<=72.96, "0", ifelse(value>72.96 & value<89.38 , "1",ifelse(value>=89.38,"2",0))))

# number           class       value       
# Min.   :   1.0   Min.   :2   Min.   : 53.43  
# 1st Qu.: 520.5   1st Qu.:2   1st Qu.: 72.27  
# Median :1001.0   Median :2   Median : 78.82  
# Mean   : 997.2   Mean   :2   Mean   : 89.68  
# 3rd Qu.:1477.5   3rd Qu.:2   3rd Qu.: 88.47  
# Max.   :1977.0   Max.   :2   Max.   :162.64   
tmp <- set2 %>% mutate(dataset = ifelse(value<=72.27, "0", ifelse(value>72.27 & value<88.47 , "1",ifelse(value>=88.47,"2",0))))


# saveRDS(tmp,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/K-means/code/euc_class1_total_detail.rds")

# saveRDS(tmp,"C:/Users/User/Desktop/0415_data_preprocessing/dataset/K-means/code/abs_class2_total_detail.rds")



# tmp<- read_rds("C:/Users/User/Desktop/0415_data_preprocessing/dataset/K-means/code/euc_class1_total_detail.rds")
# cor_train <- read_rds("C:/Users/User/Desktop/0415_data_preprocessing/dataset/code/total_corrdata.rds")
cor_train <- read_rds("C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/total_corrdata.rds")

cor_train <- correlation_set
trainY <- to_categorical(tmp$dataset)
train_corset <-list()
train_set <-list()

for(i in 1: nrow(tmp))
{
  num <- tmp$number[i]
  wafer <-cor_train[num,,]
  wafer2<-x[num,,]
  train_corset[[i]] <- wafer
  train_set[[i]] <- wafer2
}
origin_train <- abind(train_set,along = 0) #718 316  19
dim(origin_train)
corr_train <- abind(train_corset,along = 0) #718 19  19
dim(corr_train)

smp_size <- floor(0.7 * dim(origin_train)[1])
## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(dim(origin_train)[1]), size = smp_size)
Xtrain <- origin_train[train_ind,, ] 
Xtest <- origin_train[-train_ind,,] 
Xtrain2 <- corr_train[train_ind,, ] 
Xtest2 <- corr_train[-train_ind,,] 


dim(Xtrain)
dim(Xtest)
dim(Xtrain2)
dim(Xtest2)
xtrain <- array_reshape(Xtrain,dim=c(dim(Xtrain)[1],dim(Xtrain)[2],dim(Xtrain)[3],1)) 
xtest <- array_reshape(Xtest,dim=c(dim(Xtest)[1],dim(Xtest)[2],dim(Xtest)[3],1)) 

xtrain2 <- array_reshape(Xtrain2,dim=c(dim(Xtrain2)[1],dim(Xtrain2)[2],dim(Xtrain2)[3],1)) 
xtest2 <- array_reshape(Xtest2,dim=c(dim(Xtest2)[1],dim(Xtest2)[2],dim(Xtest2)[3],1)) 

trainy <- trainY[train_ind, ] #502 3 
testy <- trainY[-train_ind, ] #216 3

trainy2 <- as.matrix(tmp[train_ind,3]) #502 1 
testy2 <- as.matrix(tmp[-train_ind,3]) #216 3

dim(xtrain)
dim(xtrain2)
dim(xtest)
class(trainy)
dim(trainy2)

saveRDS(tmp,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/abs_class2_total_detail.rds")
saveRDS(xtrain,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/abs_class2_xtrain.rds")
saveRDS(xtest,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/abs_class2_xtest.rds")
saveRDS(trainy,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/abs_class2_trainy.rds")
saveRDS(testy,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/abs_class2_testy.rds")
saveRDS(trainy2,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/abs_class2_trainy_removal.rds")
saveRDS(testy2,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/abs_class2_testy_removal.rds")

# saveRDS(tmp,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/K-means/abs_class2_total_detail.rds")
# saveRDS(xtrain,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/abs_class2_xtrain.rds")
# saveRDS(xtest,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/abs_class2_xtest.rds")
# saveRDS(trainy,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/abs_class2_trainy.rds")
# saveRDS(testy,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/abs_class2_testy.rds")
# saveRDS(trainy2,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/abs_class2_trainy_removal.rds")
# saveRDS(testy2,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/abs_class2_testy_removal.rds")


# saveRDS(tmp,"C:/Users/User/Desktop/0415_data_preprocessing/dataset/K-means/code/abs_class2_total_detail.rds")
# saveRDS(xtrain,"C:/Users/User/Desktop/0415_data_preprocessing/dataset/K-means/code/abs_class2_xtrain.rds")
# saveRDS(xtest,"C:/Users/User/Desktop/0415_data_preprocessing/dataset/K-means/code/abs_class2_xtest.rds")
# saveRDS(trainy,"C:/Users/User/Desktop/0415_data_preprocessing/dataset/K-means/code/abs_class2_trainy.rds")
# saveRDS(testy,"C:/Users/User/Desktop/0415_data_preprocessing/dataset/K-means/code/abs_class2_testy.rds")
# saveRDS(trainy2,"C:/Users/User/Desktop/0415_data_preprocessing/dataset/K-means/code/abs_class2_trainy_removal.rds")
# saveRDS(testy2,"C:/Users/User/Desktop/0415_data_preprocessing/dataset/K-means/code/abs_class2_testy_removal.rds")
# 
# saveRDS(xtrain2,"C:/Users/User/Desktop/0415_data_preprocessing/dataset/K-means/code/Corr/abs_class2_xtrain.rds")
# saveRDS(xtest2,"C:/Users/User/Desktop/0415_data_preprocessing/dataset/K-means/code/Corr/abs_class2_xtest.rds")
# saveRDS(trainy,"C:/Users/User/Desktop/0415_data_preprocessing/dataset/K-means/code/Corr/abs_class2_trainy.rds")
# saveRDS(testy,"C:/Users/User/Desktop/0415_data_preprocessing/dataset/K-means/code/Corr/abs_class2_testy.rds")
# saveRDS(trainy2,"C:/Users/User/Desktop/0415_data_preprocessing/dataset/K-means/code/Corr/abs_class2_trainy_removal.rds")
# saveRDS(testy2,"C:/Users/User/Desktop/0415_data_preprocessing/dataset/K-means/code/Corr/abs_class2_testy_removal.rds")

##########################################################################################################################
###################################################################################################################
#################################### CAE feature extraction　######################################################
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
xtrain<-read_rds("C:/Users/User/Desktop/2021_0415/dataset/normalize/total_xtrain.rds")
xtest<-read_rds("C:/Users/User/Desktop/2021_0415/dataset/normalize/total_xtest.rds")
y_removal<-read_rds("C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/only_scale/trainy.rds")
y_train_class<-read_rds("C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/only_scale/trainy_class.rds")


xtrain<-read_rds("C:/Users/User/Desktop/2021_0416/dataset/normalize/total_xtrain.rds")
xtest<-read_rds("C:/Users/User/Desktop/2021_0416/dataset/normalize/total_xtest.rds")
y_removal<-read_rds("C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/only_scale/trainy.rds")
y_train_class<-read_rds("C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/only_scale/trainy_class.rds")


ytrain<- data.frame(y_removal)
order <- ytrain[order(ytrain$AVG_REMOVAL_RATE,decreasing = T),]
defect <- order[1:4,]
# WAFER_ID STAGE AVG_REMOVAL_RATE
# 311 2058207580     A         4326.154
# 195 1834206730     A         4202.112
# 197 1834206944     A         4182.417
# 200 1834206972     A         4129.494
y <- ytrain[-as.numeric(defect$number),]#364   3
x <- xtrain[-as.numeric(defect$number),,]


######## paper 6 SVID --------------------------------------------------------------
var <- c(c(1:4),c(10,11))
tmp <- x[,,var]
tmp2 <-xtest[,,var]

xtrain <- array_reshape(tmp,dim=c(dim(tmp)[1],dim(tmp)[2],dim(tmp)[3],1))
xtest <- array_reshape(tmp2,dim=c(dim(tmp2)[1],dim(tmp2)[2],dim(tmp2)[3],1))
dim(xtrain)
dim(xtest)
######## -----------------------------------------------------------------------------
# ## encoder
# enc_input = layer_input(shape = c(316, 19, 1),name="input")
# enc_output = enc_input %>% 
#   layer_conv_2d(64,kernel_size=c(3,3), activation="relu", padding="same",name="encoder1") %>% 
#   layer_max_pooling_2d(c(3,3), padding="same",name="max_pool1")%>%
#   layer_conv_2d(32,kernel_size = c(3,3),activation="relu",padding="same",name="encoder2")%>%
#   layer_max_pooling_2d(c(3,3),padding="same",name="max_pool2")%>%
#   layer_conv_2d(16,kernel_size = c(3,3),activation="relu",padding="same",name="encoder3")%>%
#   layer_max_pooling_2d(c(3,3),padding="same",name="max_pool3")
# 
# encoder <- keras_model(enc_input,enc_output)
# summary(encoder)
# 
# 
# ## decoder 
# decoder <- encoder$output %>%
#   layer_conv_2d(16, kernel_size=c(3,3), activation="relu", padding="same",name="decoder1") %>% 
#   layer_upsampling_2d(c(3,3),name="up_samp1")%>%
#   layer_conv_2d(32, kernel_size=c(3,3), activation="relu", padding="same",name="decoder2") %>% 
#   layer_upsampling_2d(c(3,3),name="up_samp2")%>%
#   layer_conv_2d(64, kernel_size=c(3,3), activation="relu", padding="valid",name="decoder3") %>% 
#   layer_upsampling_2d(c(3,3),name="up_samp3")%>%
#   layer_conv_2d(1, kernel_size=c(3,3), activation="sigmoid",padding="valid",name="autoencoder")
# autoencoder <- keras_model(encoder$input,decoder)
# summary(autoencoder)

# callbacks = list(
#   callback_model_checkpoint("checkpoints.h5"), callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1))

##### CAE+ANN -------------------------------------------------------------------------
# ## encoder
# enc_input = layer_input(shape = c(318, 6, 1),name="input")
# enc_output = enc_input %>%
# layer_conv_2d(8,kernel_size=c(3,3), activation="relu", padding="same",name="encoder1") %>%
# layer_max_pooling_2d(c(2,2), padding="same",name="max_pool1")%>%
# layer_conv_2d(4,kernel_size = c(3,3),activation="relu",padding="same",name="encoder2")%>%
# layer_max_pooling_2d(c(2,2), padding="same",name="max_pool2")%>%
# layer_conv_2d(2,kernel_size = c(3,3),activation="relu",padding="same",name="encoder3")%>%
# layer_max_pooling_2d(c(2,2), padding="same",name="max_pool3")
# encoder <- keras_model(enc_input,enc_output)
# summary(encoder)
# 
# ## decoder
# decoder <- encoder$output %>%
#   layer_conv_2d(2, kernel_size=c(3,3), activation="relu", padding="same",name="decoder1") %>%
#   layer_upsampling_2d(c(2,2),name="up_samp1")%>%
#   layer_conv_2d(4, kernel_size=c(3,3), activation="relu", padding="same",name="decoder2")%>%
#   layer_upsampling_2d(c(2,2),name="up_samp2")%>%
#   layer_conv_2d(8, kernel_size=c(3,3), activation="relu", padding="same",name="decoder3")%>%
#   layer_upsampling_2d(c(2,2),name="up_samp3")%>%
#   layer_conv_2d(1, kernel_size=c(3,3), activation="sigmoid",padding="valid",name="autoencoder")
# autoencoder <- keras_model(encoder$input,decoder)
# summary(autoencoder)
# 
var <- c(2,3,5,6,7,8,9,12,15,16,17,19)
tmp <- x[,,var]
tmp2 <-xtest[,,var]
xtrain<- array_reshape(tmp,dim=c(dim(tmp)[1],dim(tmp)[2],dim(tmp)[3],1))
xtest<- array_reshape(tmp2,dim=c(dim(tmp2)[1],dim(tmp2)[2],dim(tmp2)[3],1))


dim(xtrain)
dim(xtest)

## encoder
enc_input = layer_input(shape = c(316, 12, 1),name="input")
#enc_input = layer_input(shape = c(19, 19, 1),name="input")
enc_output = enc_input %>% 
  layer_conv_2d(64,kernel_size=c(3,3), activation="relu", padding="same",name="encoder1") %>% 
  layer_max_pooling_2d(c(2,2), padding="same",name="max_pool1")%>%
  layer_conv_2d(32,kernel_size = c(3,3),activation="relu",padding="same",name="encoder2")%>%
  layer_max_pooling_2d(c(2,2),padding="same",name="max_pool2")%>%
  layer_conv_2d(16,kernel_size = c(3,3),activation="relu",padding="same",name="encoder3")%>%
  layer_max_pooling_2d(c(2,2),padding="same",name="max_pool3")%>%
  layer_conv_2d(8,kernel_size = c(3,3),activation="relu",padding="same",name="encoder4")%>%
  layer_max_pooling_2d(c(2,2),padding="same",name="max_pool4")

encoder <- keras_model(enc_input,enc_output)
summary(encoder)

## decoder 
decoder <- encoder$output %>%
  layer_conv_2d(8, kernel_size=c(3,3), activation="relu", padding="same",name="decoder1") %>% 
  layer_upsampling_2d(c(2,2),name="up_samp1")%>%
  layer_conv_2d(16, kernel_size=c(3,3), activation="relu", padding="same",name="decoder2") %>% 
  layer_upsampling_2d(c(2,2),name="up_samp2")%>%
  layer_conv_2d(32, kernel_size=c(3,3), activation="relu", padding="same",name="decoder3") %>% 
  layer_upsampling_2d(c(2,2),name="up_samp3")%>%
  layer_conv_2d(64, kernel_size=c(3,3), activation="relu", padding="valid",name="decoder4") %>% 
  layer_upsampling_2d(c(2,2),name="up_samp4")%>%
  layer_conv_2d(1, kernel_size=c(3,3), activation="sigmoid",padding="same",name="autoencoder")
autoencoder <- keras_model(encoder$input,decoder)
summary(autoencoder)

autoencoder %>% compile(optimizer="RMSprop", loss="mse")
autoencoder %>% fit(x= xtrain, y= xtrain,
                    validation_data=list(x=xtest,y=xtest),batch_size=10,epochs=100)


save_model_hdf5(autoencoder,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/CAE_featureextract.h5")
save_model_hdf5(autoencoder,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/CAE_featureextract.h5")

autoencoder <- load_model_hdf5("C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/CAE_featureextract.h5")
# autoencoder <- load_model_hdf5("C:/Users/User/Desktop/2021_0415/dataset/code/CAE_featureextract.h5")
summary(autoencoder)
## feature extraction----------------------------------------------------------------------------
# layer_name<-"max_pool3"
layer_name<-"max_pool4"
encoder <- keras_model(inputs=autoencoder$input,outputs=get_layer(autoencoder,layer_name)$output)
summary(encoder)

train_feature = encoder%>% predict(xtrain) # 1383   12    1   16
train_feature[1,,,]
dim(train_feature)
# train_matrix <- array_reshape(train_feature, c(nrow(train_feature),40*2), order = "F") #  1977  192
# saveRDS(train_matrix,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/origin_CAE_trainfeature.rds")

train_matrix <- array_reshape(train_feature, c(nrow(train_feature),dim(train_feature)[2]*dim(train_feature)[4]), order = "F") #  1977  192
# train_matrix <- array_reshape(train_feature, c(nrow(train_feature),12*16), order = "F") #  1977  192
dim(train_matrix)
# saveRDS(train_matrix,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/origin_CAE_trainfeature.rds")
# saveRDS(train_matrix,"C:/Users/User/Desktop/2021_0415/dataset/code/origin_CAE_trainfeature.rds")
saveRDS(train_matrix,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/origin_CAE_trainfeature.rds")
####### 
train_matrix<- read_rds("C:/Users/User/Desktop/2021_0415/dataset/code/origin_CAE_trainfeature.rds")

train_matrix<- read_rds("C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/origin_CAE_trainfeature.rds")
kmeans.cluster <- Kmeans(train_matrix,centers=2,nstart=25,method="euclidean")
kmeans.cluster2 <- Kmeans(train_matrix,centers=2,nstart=25,method="abspearson") ## 僅需兩群

str(kmeans.cluster)
str(kmeans.cluster_test)
library(factoextra)
plot <-fviz_cluster(kmeans.cluster,           # 分群結果
                    data = train_matrix,              # 資料
                    geom = c("point","text"), # 點和標籤(point & label)
                    frame.type = "norm")      # 框架型態
print(plot)
# Elbow Method 應用在 K-Means
# fviz_nbclust(train_matrix,
#              FUNcluster = kmeans,# K-Means
#              method = "wss",     # total within sum of square
#              k.max = 12          # max number of clusters to consider
# ) +
# 
#   labs(title="Elbow Method for K-Means") +
# 
#   geom_vline(xintercept = 3,        # 在 X=3的地方
#              linetype = 2)          # 畫一條垂直虛線
dim(train_matrix)
fviz_nbclust(train_matrix, kmeans, method = "silhouette")
saveRDS(kmeans.cluster,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/CAE_euclidean_cluster.rds")
saveRDS(kmeans.cluster2,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/CAE_abspearson_cluster.rds")

# saveRDS(kmeans.cluster,"C:/Users/User/Desktop/2021_0415/dataset/code/CAE_euclidean_cluster.rds")
# saveRDS(kmeans.cluster2,"C:/Users/User/Desktop/2021_0415/dataset/code/CAE_abspearson_cluster.rds")

# saveRDS(kmeans.cluster,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/CAE_euclidean_cluster.rds")
# saveRDS(kmeans.cluster2,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/CAE_abspearson_cluster.rds")

#################################################################################################

# cor_train <- read_rds("C:/Users/User/Desktop/2021_0416/dataset/finaldataset/total_corrdata.rds")
# cor_data<- read_rds("C:/Users/User/Desktop/2021_0416/dataset/finaldataset/total_kmeansdata.rds")

cor_train <- read_rds("C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/total_corrdata.rds")
cor_data<- read_rds("C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/total_kmeansdata.rds")
kmeans.cluster <- read_rds("C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/CAE_euclidean_cluster.rds")
kmeans.cluster2 <- read_rds("C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/CAE_abspearson_cluster.rds")

# cor_data <- read_rds("C:/Users/User/Desktop/0415_data_preprocessing/dataset/code/total_kmeansdata.rds")
# cor_train <- read_rds("C:/Users/User/Desktop/0415_data_preprocessing/dataset/code/total_corrdata.rds")
# kmeans.cluster <- read_rds("C:/Users/User/Desktop/2021_0415/dataset/code/CAE_euclidean_cluster.rds")
# kmeans.cluster2 <- read_rds("C:/Users/User/Desktop/2021_0415/dataset/code/CAE_abspearson_cluster.rds")
## EUC ----------------------------------------------------------------------------------------
## train
class <- data.frame(kmeans.cluster$cluster)
class1 <- cbind(seq(1,nrow(class),by=1),class,y$AVG_REMOVAL_RATE)
colnames(class1)<-c("number","class","value")
set1 <- class1 %>% filter(class==1)
set2 <- class1 %>% filter(class==2)

summary(set2)

# number           class       value       
# Min.   :   2.0   Min.   :1   Min.   : 67.27  
# 1st Qu.: 108.2   1st Qu.:1   1st Qu.:148.82  
# Median : 200.5   Median :1   Median :151.33  
# Mean   : 205.5   Mean   :1   Mean   :150.44  
# 3rd Qu.: 274.8   3rd Qu.:1   3rd Qu.:154.10  
# Max.   :1369.0   Max.   :1   Max.   :162.56

tmp <- set1 %>% mutate(dataset = ifelse(value<=148.82, "0", ifelse(value>148.82 & value<154.10 , "1",ifelse(value>=154.10,"2",0))))

# number           class       value       
# Min.   :   1.0   Min.   :2   Min.   : 53.43  
# 1st Qu.: 655.5   1st Qu.:2   1st Qu.: 71.50  
# Median :1097.0   Median :2   Median : 77.62  
# Mean   :1086.1   Mean   :2   Mean   : 82.86  
# 3rd Qu.:1537.5   3rd Qu.:2   3rd Qu.: 85.09  
# Max.   :1977.0   Max.   :2   Max.   :162.64 

tmp <- set2 %>% mutate(dataset = ifelse(value<=71.50, "0", ifelse(value>71.50 & value<85.09 , "1",ifelse(value>=85.09,"2",0))))


## ABS ----------------------------------------------------------------------------------------
## train

class <- data.frame(kmeans.cluster2$cluster)
class2 <- cbind(seq(1,nrow(class),by=1),class,y$AVG_REMOVAL_RATE)
colnames(class2)<-c("number","class","value")
set1 <- class2 %>% filter(class==1)
set2 <- class2 %>% filter(class==2)

summary(set2)
# number           class       value       
# Min.   :   1.0   Min.   :1   Min.   : 53.43  
# 1st Qu.: 658.2   1st Qu.:1   1st Qu.: 71.47  
# Median :1098.5   Median :1   Median : 77.60  
# Mean   :1087.8   Mean   :1   Mean   : 82.71  
# 3rd Qu.:1538.8   3rd Qu.:1   3rd Qu.: 85.05  
# Max.   :1977.0   Max.   :1   Max.   :162.64   
tmp <- set1 %>% mutate(dataset = ifelse(value<=71.47, "0", ifelse(value>71.47 & value<85.05 , "1",ifelse(value>=85.05,"2",0))))

# number           class       value       
# Min.   :   2.0   Min.   :2   Min.   : 67.27  
# 1st Qu.: 112.5   1st Qu.:2   1st Qu.:148.81  
# Median : 202.0   Median :2   Median :151.35  
# Mean   : 212.2   Mean   :2   Mean   :150.15  
# 3rd Qu.: 279.5   3rd Qu.:2   3rd Qu.:154.12  
# Max.   :1441.0   Max.   :2   Max.   :162.56
tmp <- set2 %>% mutate(dataset = ifelse(value<=148.81, "0", ifelse(value>148.81 & value<148.81 , "1",ifelse(value>=148.81,"2",0))))



saveRDS(tmp,"C:/Users/User/Desktop/K-means/code/CAE_abs_class2_total_detail.rds")

#saveRDS(tmp,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/code/abs_class2_total_detail.rds")



# cor_train <- read_rds("C:/Users/User/Desktop/2021_0415/dataset/code/total_corrdata.rds")
cor_train <- read_rds("C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/total_corrdata.rds")

cor_train <- read_rds("C:/Users/User/Desktop/2021_0416/dataset/finaldataset/total_corrdata.rds")
x <- xtrain# 12SVID
dim(x)

trainY <- to_categorical(tmp$dataset)
train_corset <-list()
train_set <-list()

for(i in 1: nrow(tmp))
{
  num <- tmp$number[i]
  wafer <-cor_train[num,,]
  # wafer2<-x[num,,]
  wafer2<-x[num,,,]
  train_corset[[i]] <- wafer
  train_set[[i]] <- wafer2
}
origin_train <- abind(train_set,along = 0) #718 316  19
dim(origin_train)
corr_train <- abind(train_corset,along = 0) #718 19  19
dim(corr_train)

smp_size <- floor(0.7 * dim(origin_train)[1])
## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(dim(origin_train)[1]), size = smp_size)
Xtrain <- origin_train[train_ind,, ] 
Xtest <- origin_train[-train_ind,,] 
Xtrain2 <- corr_train[train_ind,, ] 
Xtest2 <- corr_train[-train_ind,,] 


dim(Xtrain)
dim(Xtest)
dim(Xtrain2)
dim(Xtest2)
xtrain <- array_reshape(Xtrain,dim=c(dim(Xtrain)[1],dim(Xtrain)[2],dim(Xtrain)[3],1)) 
xtest <- array_reshape(Xtest,dim=c(dim(Xtest)[1],dim(Xtest)[2],dim(Xtest)[3],1)) 

xtrain2 <- array_reshape(Xtrain2,dim=c(dim(Xtrain2)[1],dim(Xtrain2)[2],dim(Xtrain2)[3],1)) 
xtest2 <- array_reshape(Xtest2,dim=c(dim(Xtest2)[1],dim(Xtest2)[2],dim(Xtest2)[3],1)) 

trainy <- trainY[train_ind, ] #502 3 
testy <- trainY[-train_ind, ] #216 3

trainy2 <- as.matrix(tmp[train_ind,3]) #502 1 
testy2 <- as.matrix(tmp[-train_ind,3]) #216 3

dim(xtrain)
dim(xtrain2)
dim(xtest)
class(trainy)
dim(trainy2)
dim(testy)

saveRDS(tmp,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/CAE_abs_class2_total_detail.rds")
saveRDS(xtrain,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/CAE_abs_class2_xtrain.rds")
saveRDS(xtest,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/CAE_abs_class2_xtest.rds")
saveRDS(trainy,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/CAE_abs_class2_trainy.rds")
saveRDS(testy,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/CAE_abs_class2_testy.rds")
saveRDS(trainy2,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/CAE_abs_class2_trainy_removal.rds")
saveRDS(testy2,"C:/Users/User/Desktop/2021_0415/dataset/K-means2/code/CAE_abs_class2_testy_removal.rds")



saveRDS(tmp,"C:/Users/User/Desktop/2021_0415/dataset/K-means/CAE_abs_class2_total_detail.rds")
saveRDS(xtrain,"C:/Users/User/Desktop/2021_0415/dataset/code/CAE_abs_class2_xtrain.rds")
saveRDS(xtest,"C:/Users/User/Desktop/2021_0415/dataset/code/CAE_abs_class2_xtest.rds")
saveRDS(trainy,"C:/Users/User/Desktop/2021_0415/dataset/code/CAE_abs_class2_trainy.rds")
saveRDS(testy,"C:/Users/User/Desktop/2021_0415/dataset/code/CAE_abs_class2_testy.rds")
saveRDS(trainy2,"C:/Users/User/Desktop/2021_0415/dataset/code/CAE_abs_class2_trainy_removal.rds")
saveRDS(testy2,"C:/Users/User/Desktop/2021_0415/dataset/code/CAE_abs_class2_testy_removal.rds")

saveRDS(xtrain2,"C:/Users/User/Desktop/2021_0415/dataset/code/Corr/CAE_abs_class2_xtrain.rds")
saveRDS(xtest2,"C:/Users/User/Desktop/2021_0415/dataset/code/Corr/CAE_abs_class2_xtest.rds")
saveRDS(trainy,"C:/Users/User/Desktop/2021_0415/dataset/code/Corr/CAE_abs_class2_trainy.rds")
saveRDS(testy,"C:/Users/User/Desktop/2021_0415/dataset/code/Corr/CAE_abs_class2_testy.rds")
saveRDS(trainy2,"C:/Users/User/Desktop/2021_0415/dataset/code/Corr/CAE_abs_class2_trainy_removal.rds")
saveRDS(testy2,"C:/Users/User/Desktop/2021_0415/dataset/code/Corr/CAE_abs_class2_testy_removal.rds")


# saveRDS(tmp,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/K-means/CAE_abs_class2_total_detail.rds")
# saveRDS(xtrain,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/K-means/CAE_abs_class2_xtrain.rds")
# saveRDS(xtest,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/K-means/CAE_abs_class2_xtest.rds")
# saveRDS(trainy,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/K-means/CAE_abs_class2_trainy.rds")
# saveRDS(testy,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/K-means/CAE_abs_class2_testy.rds")
# saveRDS(trainy2,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/K-means/CAE_abs_class2_trainy_removal.rds")
# saveRDS(testy2,"C:/Users/User/Desktop/2021_0416/dataset/finaldataset/K-means/CAE_abs_class2_testy_removal.rds")



