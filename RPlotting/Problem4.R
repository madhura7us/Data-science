#my working directory is set as follows: E:/UB/Spring2018/DIC/Labs/Lab1/GajendragadkarLab1Part1
#set this value to new value where the RAR is extracted

setwd("E:\\UB\\Spring2018\\DIC\\Labs\\Lab1\\GajendragadkarLab1Part1\\Problem4\\Data")


fb1<-read.csv(file= "./FB.csv")
aapl1<-read.csv(file= "./AAPL.csv")
par(bg="cornsilk")
plot(aapl1$Adj.Close, col="blue", type="o", ylim=c(0,200), xlab="Days", ylab="Price" )
lines(fb1$Adj.Close, type="o", pch=22, lty=2, col="red")
legend("topright", inset=.05, c("Apple","Facebook"), fill=c("blue","red"), horiz=TRUE)
hist(aapl1$Adj.Close, col=rainbow(8))