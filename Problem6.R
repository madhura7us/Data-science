library("ggmap") 
library("maptools") 
library("maps") 
visited <- c("SFO", "Chennai", "London", "Melbourne", "Lima", "Peru","Johannesbury","SA") 
ll.visited <- geocode(visited) 
visit.x <- ll.visited$lon 
visit.y <- ll.visited$lat 
map("world", fill=TRUE, col="white", bg="lightblue", ylim=c(-60, 90), mar=c(0,0,0,0))
points(visit.x,visit.y, col="red", pch=36)


library("ggmap") 
library("maptools") 
library(maps) 
visited <- c("SFO", "New York", "Buffalo", "Dallas, TX") 
ll.visited <- geocode(visited) 
visit.x <- ll.visited$lon 
visit.y <- ll.visited$lat 
map("state", fill=TRUE, col=rainbow(50), bg="lightblue", mar=c(0,0,0,0)) 
points(visit.x,visit.y, col="yellow", pch=36)
