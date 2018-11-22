library("lattice")
library (datasets)
splom(mtcars[c(1,3,4,5,6)], main="MTCARS Data")


library("lattice")
library (datasets)

splom(mtcars[c(1,3,4,6)], col=rainbow(2),main="MTCARS Data")

library("lattice")
library (datasets)


splom(rock[c(1,2,3,4)], main="ROCK Data")
