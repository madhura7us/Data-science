
setwd("yourPath")
sales<-read.table(file = "./salesdata.txt", header=T) 
sales # to verify that data has been read
barplot(as.matrix(sales), main="Sales Data", ylab= "Total",beside=T, col=rainbow(5))
