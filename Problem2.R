#my working directory is set as follows: E:/UB/Spring2018/DIC/Labs/Lab1/GajendragadkarLab1Part1
#set this value to new value where the RAR is extracted

setwd("E:\\UB\\Spring2018\\DIC\\Labs\\Lab1\\GajendragadkarLab1Part1\\Problem2\\Data\\")
sales<-read.table(file = "./salesdata.txt", header=T) 
sales # to verify that data has been read
barplot(as.matrix(sales), main="Sales Data", ylab= "Total",beside=T, col=rainbow(5))