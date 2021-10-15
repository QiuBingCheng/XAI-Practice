###################################################################################################################
####################################資料前處理　##################################################################
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
###合併已經rescale 數據 ------------------------------------------------------------------------------------------
Xtrain_stageA456  <- read_rds("C:/Users/User/Desktop/Amber/PHM/PHM/進度/0104/final scaling code/xtrain_stageA456.rds")
Xtest_stageA456 <- read_rds("C:/Users/User/Desktop/Amber/PHM/PHM/進度/0104/final scaling code/xtest_stageA456.rds")
Xtrain_stageA123  <- read_rds("C:/Users/User/Desktop/Amber/PHM/PHM/進度/0104/final scaling code/xtrain_stageA123.rds")
Xtest_stageA123 <- read_rds("C:/Users/User/Desktop/Amber/PHM/PHM/進度/0104/final scaling code/xtest_stageA123.rds")
Xtrain_stageB456  <- read_rds("C:/Users/User/Desktop/Amber/PHM/PHM/進度/0104/final scaling code/xtrain_stageB456.rds")
Xtest_stageB456 <- read_rds("C:/Users/User/Desktop/Amber/PHM/PHM/進度/0104/final scaling code/xtest_stageB456.rds")

xtrain_set <- c(Xtrain_stageA123,Xtrain_stageA456,Xtrain_stageB456)
xtest_set <- c(Xtest_stageA123,Xtest_stageA456,Xtest_stageB456)
saveRDS(xtrain_set,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/only_scale/xtrain_set.rds")
saveRDS(xtest_set,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/only_scale/xtest_set.rds")

trainY_A456 <- read_rds("C:/Users/User/Desktop/Amber/PHM/PHM/進度/0104/final scaling code/trainY_A456.rds")
testY_A456 <- read_rds("C:/Users/User/Desktop/Amber/PHM/PHM/進度/0104/final scaling code/testY_A456.rds")
trainY_A123 <- read_rds("C:/Users/User/Desktop/Amber/PHM/PHM/進度/0104/final scaling code/trainY_A123.rds")
testY_A123 <- read_rds("C:/Users/User/Desktop/Amber/PHM/PHM/進度/0104/final scaling code/testY_A123.rds")
trainY_B456 <- read_rds("C:/Users/User/Desktop/Amber/PHM/PHM/進度/0104/final scaling code/trainY_B456.rds")
testY_B456 <- read_rds("C:/Users/User/Desktop/Amber/PHM/PHM/進度/0104/final scaling code/testY_B456.rds")
trainY_A456 <- cbind(c(1:nrow(trainY_A456)),trainY_A456)
colnames(trainY_A456)[1]<- "number"
trainY_A123 <- cbind(c(1:nrow(trainY_A123)),trainY_A123)
colnames(trainY_A123)[1]<- "number"
trainY_B456 <- cbind(c(1:nrow(trainY_B456)),trainY_B456)
colnames(trainY_B456)[1]<- "number"
y_removal <- rbind(data.table(trainY_A123),data.table(trainY_A456),data.table(trainY_B456))

trainY_A123[,4]<-0
trainY_A456[,4]<-1
trainY_B456[,4]<-2
trainY_set <- rbind(data.table(trainY_A123),data.table(trainY_A456),data.table(trainY_B456))
trainY_set<-data.frame(trainY_set)
trainY <- to_categorical(trainY_set[,4])

y_removal2 <- rbind(data.table(testY_A123),data.table(testY_A456),data.table(testY_B456))
testY_A123[,3]<-0
testY_A456[,3]<-1
testY_B456[,3]<-2
testY_set <- rbind(data.table(testY_A123),data.table(testY_A456),data.table(testY_B456))
testY_set<-data.frame(testY_set)


saveRDS(y_removal,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/only_scale/trainy.rds")
saveRDS(y_removal2,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/only_scale/testy.rds")


###### 資料正規化(19 SVID) mean==0 ; sd=1 ------------------------------------------------------------- 
lst <- list()
for(i in 1: length(xtrain_set))
{
  subset <- xtrain_set[[i]]
  k <- apply(subset, 2, function(x) round((x-mean(x))/sd(x), 3)) 
  k[is.na(k)]<-0
  lst[[i]] <- k
}

lst2 <- list()
for(i in 1: length(xtest_set))
{
  subset2 <- xtest_set[[i]]
  k2 <- apply(subset2, 2, function(x) round((x-mean(x))/sd(x), 3))
  k2[is.na(k2)]<-0
  lst2[[i]]<- k2
}
tmp <- lst[[1]]
# list dataset 轉成三維矩陣 -----------------------------------　
xtrain <- abind(lst , along = 0) # 1981 316 19 
xtest <- abind(lst2 , along = 0) # 424 316  19 
dim(xtrain)
dim(xtest)

saveRDS(xtrain,"C:/Users/User/Desktop/Amber/PHM/PHM/進度/0119 meeting/datacode/total data/xtrain.rds")
saveRDS(xtest,"C:/Users/User/Desktop/Amber/PHM/PHM/進度/0119 meeting/datacode/total data/xtest.rds")

saveRDS(xtrain,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/normalize/total_xtrain.rds")
saveRDS(xtest,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/normalize/total_xtest.rds")


### 剔除A123 4片outlier 、removalrate分三levels-------------------------------------------------------------
xtrain<-read_rds("C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/normalize/total_xtrain.rds")
xtest<-read_rds("C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/normalize/total_xtest.rds")

y_removal<-read_rds("C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/only_scale/trainy.rds")
y_removal2<-read_rds("C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/only_scale/testy.rds")

y_train_class<-read_rds("C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/only_scale/trainy_class.rds")
y_test_class <-read_rds("C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/only_scale/testy_class.rds")


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

summary(y)
# number         WAFER_ID          STAGE    AVG_REMOVAL_RATE
# Min.   :  1.0   Min.   :-4.230e+09   A:1162   Min.   : 53.43  
# 1st Qu.:165.0   1st Qu.:-8.772e+08   B: 815   1st Qu.: 72.37  
# Median :331.0   Median : 1.476e+09            Median : 79.15  
# Mean   :363.3   Mean   : 8.907e+08            Mean   : 90.31  
# 3rd Qu.:560.0   3rd Qu.: 2.981e+09            3rd Qu.: 88.63  
# Max.   :815.0   Max.   : 4.230e+09            Max.   :162.64  
tmp <- y %>% mutate(dataset = ifelse(AVG_REMOVAL_RATE<=72.37, "0", ifelse(AVG_REMOVAL_RATE>72.37 & AVG_REMOVAL_RATE<88.63 , "1",ifelse(AVG_REMOVAL_RATE>=88.63,"2",0))))
train_class <- to_categorical(tmp$dataset)

dim(x) #1977  316   19
dim(y) # 1977    4
dim(train_class)


### 原本train set splitted (70% 30%) ----------------------------------------------------------------------
smp_size <- floor(0.7 * dim(x)[1])
set.seed(123)
train_ind <- sample(seq_len(dim(x)[1]), size = smp_size)

Xtrain<- x[train_ind,,]
Xtest<- x[-train_ind,,]
trainy <- as.matrix(y[train_ind,4])
testy <- as.matrix(y[-train_ind,4])

trainy_class <- train_class[train_ind,]
testy_class <- train_class[-train_ind,]


xtrain <- array_reshape(Xtrain,dim=c(dim(Xtrain)[1],dim(Xtrain)[2],dim(Xtrain)[3],1)) ## 1977  316   19    1
xtest <- array_reshape(Xtest,dim=c(dim(Xtest)[1],dim(Xtest)[2],dim(Xtest)[3],1)) ## 424 316  19   1
dim(xtrain)
dim(xtest)
dim(trainy_class)
dim(trainy)
saveRDS(xtrain,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/code/xtrain.rds")
saveRDS(xtest,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/code/xtest.rds")
saveRDS(trainy,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/code/trainy.rds")
saveRDS(testy,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/code/testy.rds")
saveRDS(trainy_class,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/code/trainy_class.rds")
saveRDS(testy_class,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/code/testy_class.rds")

### 各dataset set splitted (70% 30%) ----------------------------------------------------------------------
A123<- x[1:364,,]
A456<- x[365:1162,,]
B456<- x[1163:1977,,]

A123_y<- y[1:364,]
A456_y<- y[365:1162,]
B456_y<- y[1163:1977,]

summary(A123_y)
## A123:
# number          WAFER_ID          STAGE   AVG_REMOVAL_RATE
# Min.   :  1.00   Min.   :-1.194e+09   A:364   Min.   :138.6   
# 1st Qu.: 91.75   1st Qu.: 3.309e+08   B:  0   1st Qu.:148.6   
# Median :182.50   Median : 1.836e+09           Median :151.1   
# Mean   :184.05   Mean   : 1.171e+09           Mean   :151.2   
# 3rd Qu.:276.25   3rd Qu.: 1.854e+09           3rd Qu.:153.7   
# Max.   :368.00   Max.   : 2.078e+09           Max.   :162.6  
tmp <- A123_y %>% mutate(dataset = ifelse(AVG_REMOVAL_RATE<=148.6, "0", ifelse(AVG_REMOVAL_RATE>148.6 & AVG_REMOVAL_RATE<153.7 , "1",ifelse(AVG_REMOVAL_RATE>=153.7,"2",0))))
A123_class <- to_categorical(tmp$dataset)
## A456:
# number         WAFER_ID          STAGE   AVG_REMOVAL_RATE
# Min.   :  1.0   Min.   :-4.230e+09   A:798   Min.   :53.43   
# 1st Qu.:200.2   1st Qu.:-9.032e+08   B:  0   1st Qu.:69.70   
# Median :399.5   Median : 1.452e+09           Median :74.06   
# Mean   :399.5   Mean   : 8.643e+08           Mean   :73.08   
# 3rd Qu.:598.8   3rd Qu.: 3.017e+09           3rd Qu.:77.71   
# Max.   :798.0   Max.   : 4.230e+09           Max.   :88.70   
tmp <- A456_y %>% mutate(dataset = ifelse(AVG_REMOVAL_RATE<=69.70, "0", ifelse(AVG_REMOVAL_RATE>69.70 & AVG_REMOVAL_RATE<77.71 , "1",ifelse(AVG_REMOVAL_RATE>=77.71,"2",0))))
A456_class <- to_categorical(tmp$dataset)

## B456:
# number         WAFER_ID          STAGE   AVG_REMOVAL_RATE
# Min.   :  1.0   Min.   :-4.230e+09   A:  0   Min.   : 54.31  
# 1st Qu.:204.5   1st Qu.:-1.720e+09   B:815   1st Qu.: 73.33  
# Median :408.0   Median : 1.462e+09           Median : 81.51  
# Mean   :408.0   Mean   : 7.915e+08           Mean   : 79.97  
# 3rd Qu.:611.5   3rd Qu.: 3.021e+09           3rd Qu.: 86.76  
# Max.   :815.0   Max.   : 4.230e+09           Max.   :101.46 
tmp <- B456_y %>% mutate(dataset = ifelse(AVG_REMOVAL_RATE<=73.33, "0", ifelse(AVG_REMOVAL_RATE>73.33 & AVG_REMOVAL_RATE<86.76 , "1",ifelse(AVG_REMOVAL_RATE>=86.76,"2",0))))
B456_class <- to_categorical(tmp$dataset)

smp_size <- floor(0.7 * dim(A123)[1])
set.seed(123)
train_ind <- sample(seq_len(dim(A123)[1]), size = smp_size)

Xtrain<- A123[train_ind,,]
Xtest<- A123[-train_ind,,]

trainy <- as.matrix(A123_y[train_ind,4])
testy <- as.matrix(A123_y[-train_ind,4])

trainy_class <- A123_class[train_ind,]
testy_class <- A123_class[-train_ind,]

xtrain <- array_reshape(Xtrain,dim=c(dim(Xtrain)[1],dim(Xtrain)[2],dim(Xtrain)[3],1)) ## 1977  316   19    1
xtest <- array_reshape(Xtest,dim=c(dim(Xtest)[1],dim(Xtest)[2],dim(Xtest)[3],1)) ## 424 316  19   1

dim(xtrain)
dim(xtest)
dim(trainy)
dim(testy)

saveRDS(xtrain,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/code/A123_xtrain.rds")
saveRDS(xtest,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/code/A123_xtest.rds")
saveRDS(trainy,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/code/A123_trainy.rds")
saveRDS(testy,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/code/A123_testy.rds")
saveRDS(trainy_class,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/code/A123_trainy_class.rds")
saveRDS(testy_class,"C:/Users/User/Desktop/0409_datapreprocessing_problem/dataset/code/A123_testy_class.rds")

