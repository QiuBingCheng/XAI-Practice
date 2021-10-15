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

data <- abind(xtrain,xtest,along=1)

zero_set <- list()
for(i in 1:dim(data)[1])
{
  wafer <- data.frame(data[i,,,])
  colnames(wafer)<- c(seq(1,19,by=1))
  ans = colnames(wafer[,colSums(wafer[])==0])
  zero_set[[i]]<-as.numeric(ans)
}

output_count <- table(unlist(zero_set))
saveRDS(output_count,"C:/Users/User/Desktop/2021_0527/SVIDprofile/count/B456.rds")


# Plot 
set <- data.table(output_count)
null <- data.table(seq(1,19,by=1),rep(0,19))
for(i in 1:dim(set)[1])
{
  num <- as.numeric(set[i,1])
  null[num,2]<-set[i,2]
}

colnames(null)<-c("SVID","count")

max_limit <- ceiling(max(null$count)/100)*100
null$SVID <- factor(null$SVID,levels =seq(1,19,by=1))
plot <- ggplot(null, aes(x =  SVID, y = count)) +
  geom_bar(stat = "identity",fill="steelblue",width = 0.5) +
  xlab("SVID")+ylab("frequency")+
  ggtitle("Zero value of B456 SVID ") +
  theme_minimal()+
  geom_text(aes(label=count), vjust=-0.3, size=4)+
  theme(axis.line.x = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 20,face="bold"), 
        axis.title = element_text(size = 20,face="bold"), 
        plot.title = element_text(size = 30, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        axis.title.x = element_text(size=25,face="bold"),
        axis.title.y = element_text(size=25,face="bold"),
        axis.text.y = element_text(size=15),
        axis.text.x = element_text(size=15,face="bold"),
        legend.title = element_text(size=12),
        legend.text = element_text(size=12),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank()) +
  ylim(0,max_limit)+scale_y_continuous(breaks=seq(0,max_limit,50), limits=c(0, max_limit))
print(plot)

ggsave(plot, file="C:/Users/User/Desktop/2021_0527/SVIDprofile/count/B456.png", width=15, height=10)

#####################################
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
A123<- read_rds("D:/AmberChu/Amber/PHM/PHM/턨プ/2021_0527/SVIDprofile/count/A123.rds")
A456<- read_rds("D:/AmberChu/Amber/PHM/PHM/턨プ/2021_0527/SVIDprofile/count/A456.rds")
B456<- read_rds("D:/AmberChu/Amber/PHM/PHM/턨プ/2021_0527/SVIDprofile/count/B456.rds")

A123<- data.table(A123)
A456<- data.table(A456)
B456<- data.table(B456)

null_a123 <- data.table(seq(1,19,by=1),rep(0,19))
for(i in 1:dim(A123)[1])
{
  num <- as.numeric(A123[i,1])
  null_a123[num,2]<-A123[i,2]
}

null_a456 <- data.table(seq(1,19,by=1),rep(0,19))
for(i in 1:dim(A456)[1])
{
  num <- as.numeric(A456[i,1])
  null_a456[num,2]<-A456[i,2]
}

null_b456 <- data.table(seq(1,19,by=1),rep(0,19))
for(i in 1:dim(B456)[1])
{
  num <- as.numeric(B456[i,1])
  null_b456[num,2]<-B456[i,2]
}

null_a123$V1<-paste0(rep("SVID",19),seq(1,19,1))
null_a456$V1<-paste0(rep("SVID",19),seq(1,19,1))
null_b456$V1<-paste0(rep("SVID",19),seq(1,19,1))
A123 <- data.table(rep("A123 dataset",19),null_a123)
A456 <- data.table(rep("A456 dataset",19),null_a456)
B456 <- data.table(rep("B456 dataset",19),null_b456)


total_set <- rbind(A123,A456,B456)
colnames(total_set)<-c("dataset","SVID","frequency")


max_limit <- ceiling(max(total_set$frequency)/100)*100
# total_set$SVID <- factor(total_set$SVID)
total_set$SVID <- factor(total_set$SVID,levels=paste0(rep("SVID",19),seq(1,19,1)))




plot <- ggplot(total_set, aes(x =  SVID, y = frequency)) + 
  geom_bar(stat = "identity",fill="#316A9E",width = 0.5) +
  geom_text(data=subset(total_set, frequency > 0),
            aes(SVID,frequency,label=frequency),vjust=-0.3,size=3.5)+
  facet_wrap(~dataset, scales="free_y",  ncol=1)+
  # geom_text(aes(label = Importance,hjust = ifelse(Importance > 0 , 1.2, 0)),size=4) +
  xlab("19 variable")+ylab("frequency of zero value")+
  theme_minimal()+
  theme(axis.line.x = element_line(size = .6, colour = "black"),
        axis.text = element_text(size = 12,face="bold"), 
        axis.title = element_text(size = 12,face="bold"),
        # panel.background = element_blank(),
        plot.title = element_text(size = 30, family = "Mongolian Baiti",hjust = 0.5,face = "bold"),
        # axis.title.x = element_text(size=15),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size=20),
        axis.text.y = element_text(size=15),
        axis.text.x = element_text(size=15),
        legend.title = element_text(size=25),
        legend.text = element_text(size=15),
        strip.text = element_text(size=30,face="bold",colour = "black"),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank())
print(plot)

# ggsave(plot, file="C:/Users/User/Desktop/2021_0527/SVIDprofile/count/three_set.png", width=15, height=10)
ggsave(plot, file="D:/AmberChu/Amber/PHM/PHM/턨プ/2021_0527/SVIDprofile/count/three_set2.png", width=15, height=10)

ggsave(plot, file="C:/Users/User/Desktop/three_set_SVID_zerovalue.png", width=18, height=13)



