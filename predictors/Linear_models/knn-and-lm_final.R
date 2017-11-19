#  The MIT License (MIT)
#  Copyright (c) 2017 Anja Meunier
#  
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
#  OR OTHER DEALINGS IN THE SOFTWARE.

library(leaps) # contains subset selection for linear regression
library(kknn) # contains knn function
library(glmnet)

stockdata <- read.csv("./../../new_database/aapl/time_data.csv", header = TRUE)
#stockdata <- read.csv("./../../new_database/googl/time_data.csv", header = TRUE)
#stockdata <- read.csv("./../../new_database/amzn/time_data.csv", header = TRUE)
#stockdata <- read.csv("./../../new_database/vrx/time_data.csv", header = TRUE)
#stockdata <- read.csv("./../../new_database/wmt/time_data.csv", header = TRUE) 

# FEATURE EXPLANATION:

# Input
# Date                          Last date of which we know all the financial data closing price / relative intraday price change
# adj_low                       Adjusted Low of the stock price on day "Date"
# adj_high                      Adjusted High of the stock price on day "Date"
# adj_open_tomorrow             Adjusted Open one day after the day "Date"
# adj_vol                       Adjusted Volume traded of the stock on day "Date"
# vol_rel_m_ave_10              Volume on day "Date" / volume of the last 10 days
# relative_overnight_tomorrow   Relative overnight price change (adj. open tomorrow / adj close)
# relative_intraday             Relative intraday price change (adj. close / adj open)
# sentiment_n                   Negative sentiment score from news
# sentiment_p                   Positive sentiment score from news

# Output
# relative_intraday_tomorrow    Relative intraday price change (adj. close / adj open) of one day after the day "stated in "date"
# adj_close_tomorrow            Adjusted closing price of one day after the day stated in "date"

# Our assumed scenario is that we would run the model at the time of market opening on day "Date" + 1, so we can include that days Opening as input. We will predict either the absolute Closing Price or the relative price change.


head(stockdata)
summary(stockdata)
dim(stockdata)

a1 <- round(0.6*nrow(stockdata)) #nr of days for training price prediction
a2 <- round(0.25*nrow(stockdata)) #nr of days for training error prediction
a3 <- nrow(stockdata) - a1 - a2 - 1

a4 <- round(0.8*a1) #nr of days used as pseudo test data to determine optimal lambda and k


# Standardization not necessary for lm and done automatically by knn


# PRICE MODELS

# lm - lasso - determine optimal lambda

lambda_grid <- 10 ^ seq(-5, -1, length = 200) #lambda grid for lasso

predictions.lasso <- data.frame(matrix(rep(0, length(lambda_grid)*(a1-a4)),ncol=length(lambda_grid)))

errors.lasso <-  data.frame(matrix(rep(0, length(lambda_grid)*(a1-a4)),ncol=length(lambda_grid)))
errors.lasso2 <- matrix(rep(0, length(lambda_grid),ncol=1))


for (n in 0:(a1-a4-1)) {
  m <- a4+n
  
  x <- model.matrix(relative_intraday_tomorrow ~ +sentiment_p+sentiment_n+adj_low+adj_high+adj_open_tomorrow+adj_vol+relative_overnight_tomorrow+relative_intraday+vol_rel_m_ave_10, data = stockdata[1:m,])[,-1]
  y <- stockdata[1:m,]$relative_intraday_tomorrow
    
  lasso <- glmnet(x, y, alpha = 1, lambda = lambda_grid)
  x.test <- model.matrix(relative_intraday_tomorrow ~ +sentiment_p+sentiment_n+adj_low+adj_high+adj_open_tomorrow+adj_vol+relative_overnight_tomorrow+relative_intraday+vol_rel_m_ave_10, data = stockdata[m+1,])[,-1]
  
  for (l in 1:length(lambda_grid)) {
    predictions.lasso[n+1,l] <- c(1,x.test) %*% coef(lasso)[,l]
    errors.lasso[n+1,l] <- ((stockdata[m+1, ]$relative_intraday_tomorrow - predictions.lasso[n+1,l]) ^ 2) 
  }
}  

errors.lasso2<- colMeans(errors.lasso)
errors.lasso2 <- sqrt(errors.lasso2)

plot(lambda_grid,errors.lasso2,col=4,type="p",xlab="Lambda",ylab="RMSE")
points(lambda_grid[which.min(errors.lasso2)], min(errors.lasso2), col="red", cex = 2, pch = 20)

(lambda.min <- lambda_grid[which.min(errors.lasso2)])
which.min(errors.lasso2)

coef(lasso)[,which.min(errors.lasso2)]


# KNN full - determine optimal k

k_grid <- c(seq(1, 19, by = 1),seq(20, 100, by = 2))

predictions.knn <- data.frame(matrix(rep(0, length(k_grid)*(a1-a4)),ncol=length(k_grid)))

errors.knn <-  data.frame(matrix(rep(0, length(k_grid)*(a1-a4)),ncol=length(k_grid)))
errors.knn2 <- matrix(rep(0, length(k_grid),ncol=1))

for (n in 0:(a1-a4-1)) {
  m <- a4+n
  
  for (i in 1:length(k_grid)) {
    knn.pred <- kknn(relative_intraday_tomorrow ~ +sentiment_p+sentiment_n+adj_low+adj_high+adj_open_tomorrow+adj_vol+relative_overnight_tomorrow+relative_intraday+vol_rel_m_ave_10, stockdata[1:m, ],stockdata[m+1, ],k=k_grid[i],distance=1, kernel = "rectangular",scale=TRUE)
    predictions.knn[n+1,i] <- fitted(knn.pred)
    errors.knn[n+1,i] <- ((stockdata[m+1, ]$relative_intraday_tomorrow - fitted(knn.pred)) ^ 2)
  }
}  

errors.knn2<- colMeans(errors.knn)
errors.knn2 <- sqrt(errors.knn2)


plot(k_grid,errors.knn2,col=4,type="p",xlab="K",ylab="RMSE")
points(k_grid[which.min(errors.knn2)], min(errors.knn2), col="red", cex = 2, pch = 20)

(k.min <- k_grid[which.min(errors.knn2)])
which.min(errors.knn2)


# KNN selected predictors - determine optimal k

predictions.knn.2 <- data.frame(matrix(rep(0, length(k_grid)*(a1-a4)),ncol=length(k_grid)))

errors.knn.2 <-  data.frame(matrix(rep(0, length(k_grid)*(a1-a4)),ncol=length(k_grid)))
errors.knn2.2 <- matrix(rep(0, length(k_grid),ncol=1))

for (n in 0:(a1-a4-1)) {
  m <- a4+n
  
  for (i in 1:length(k_grid)) {
    knn.pred.2 <- kknn(relative_intraday_tomorrow ~ adj_low+adj_vol+relative_overnight_tomorrow+relative_intraday+vol_rel_m_ave_10, stockdata[1:m, ],stockdata[m+1, ],k=k_grid[i],distance=1, kernel = "rectangular",scale=TRUE)
    predictions.knn.2[n+1,i] <- fitted(knn.pred.2)
    errors.knn.2[n+1,i] <- ((stockdata[m+1, ]$relative_intraday_tomorrow - fitted(knn.pred.2)) ^ 2)
  }
}  

errors.knn2.2<- colMeans(errors.knn.2)
errors.knn2.2 <- sqrt(errors.knn2.2)


plot(k_grid,errors.knn2.2,col=4,type="p",xlab="K",ylab="RMSE")
points(k_grid[which.min(errors.knn2.2)], min(errors.knn2.2), col="red", cex = 2, pch = 20)

(k.min.2 <- k_grid[which.min(errors.knn2.2)])
which.min(errors.knn2.2)



# model comparison 

modnr <- 6  #nr of models to be compared

predictions.price <- data.frame(matrix(rep(0, modnr*(a2+a3)),ncol=modnr))
colnames(predictions.price)<- c("lmfull","lmrel","lmabs", "lasso.lmin","knn.kmin","knn.kmin2") 

errors.price <-  data.frame(matrix(rep(0, modnr*(a2+a3)),ncol=modnr))
colnames(errors.price)<- c("lmfull","lmrel","lmabs","lasso.lmin","knn.kmin","knn.kmin2") 


B <- a2 + a3
for (b in 0:(B-1)) {
  m <- a1+b
  
  #lm
   
  # predict absolute price change directly

  lm.fit.abs <- lm( adj_close_tomorrow ~+sentiment_p+sentiment_n+adj_low+adj_high+adj_open_tomorrow+adj_vol+relative_overnight_tomorrow+relative_intraday+vol_rel_m_ave_10, data = stockdata[1:m, ])
  predictions.price[b+1,]$lmabs<- predict(lm.fit.abs, newdata = stockdata[m+1, ])
  errors.price[b+1,]$lmabs<-((stockdata[m+1, ]$adj_close_tomorrow - predictions.price[b+1,]$lmabs) ^ 2)


  # predict relative price change, calculate error on absolute price

  lm.fit.full <- lm( relative_intraday_tomorrow ~+sentiment_p+sentiment_n+adj_low+adj_high+adj_open_tomorrow+adj_vol+relative_overnight_tomorrow+relative_intraday+vol_rel_m_ave_10, data = stockdata[1:m, ])
  predictions.price[b+1,]$lmfull<- stockdata[m+1, ]$adj_open_tomorrow*(1+predict(lm.fit.full, newdata = stockdata[m+1, ]))
  errors.price[b+1,]$lmfull<-((stockdata[m+1, ]$adj_close_tomorrow - predictions.price[b+1,]$lmfull) ^ 2)

  lm.fit.rel <- lm( relative_intraday_tomorrow ~ +sentiment_p+sentiment_n+relative_overnight_tomorrow+relative_intraday, data = stockdata[1:m, ])
  predictions.price[b+1,]$lmrel<- stockdata[m+1, ]$adj_open_tomorrow*(1+predict(lm.fit.rel, newdata = stockdata[m+1, ]))
  errors.price[b+1,]$lmrel<-((stockdata[m+1, ]$adj_close_tomorrow - predictions.price[b+1,]$lmrel) ^ 2)

  # lasso with lambda min
  x <- model.matrix(relative_intraday_tomorrow ~ +sentiment_p+sentiment_n+adj_low+adj_high+adj_open_tomorrow+adj_vol+relative_overnight_tomorrow+relative_intraday+vol_rel_m_ave_10, data = stockdata[1:m,])[,-1]
  y <- stockdata[1:m,]$relative_intraday_tomorrow

  lasso.lmin <- glmnet(x, y, alpha = 1, lambda = lambda.min)
  x.test <- model.matrix(relative_intraday_tomorrow ~ +sentiment_p+sentiment_n+adj_low+adj_high+adj_open_tomorrow+adj_vol+relative_overnight_tomorrow+relative_intraday+vol_rel_m_ave_10, data = stockdata[m+1,])[,-1]

  predictions.price[b+1,]$lasso.lmin <- stockdata[m+1, ]$adj_open_tomorrow*(1+(c(1,x.test) %*% coef(lasso.lmin)))
  errors.price[b+1,]$lasso.lmin <-((stockdata[m+1, ]$adj_close_tomorrow - predictions.price[b+1,]$lasso.lmin) ^ 2)


  # knn full
  knn.kmin <- kknn(relative_intraday_tomorrow ~ +sentiment_p+sentiment_n+adj_low+adj_high+adj_open_tomorrow+adj_vol+relative_overnight_tomorrow+relative_intraday+vol_rel_m_ave_10, stockdata[1:m, ],stockdata[m+1, ],k=k.min,distance=1, kernel = "rectangular",scale=TRUE)
  predictions.price[b+1,]$knn.kmin<- stockdata[m+1, ]$adj_open_tomorrow*(1+fitted(knn.kmin))
  errors.price[b+1,]$knn.kmin<-((stockdata[m+1, ]$adj_close_tomorrow - predictions.price[b+1,]$knn.kmin) ^ 2)

  # knn selected
  knn.kmin2 <- kknn(relative_intraday_tomorrow ~ adj_low+adj_vol+relative_overnight_tomorrow+relative_intraday+vol_rel_m_ave_10, stockdata[1:m, ],stockdata[m+1, ],k=k.min.2,distance=1, kernel = "rectangular",scale=TRUE)
  predictions.price[b+1,]$knn.kmin2<- stockdata[m+1, ]$adj_open_tomorrow*(1+fitted(knn.kmin2))
  errors.price[b+1,]$knn.kmin2<-((stockdata[m+1, ]$adj_close_tomorrow - predictions.price[b+1,]$knn.kmin2) ^ 2)

}

coef(lasso.lmin)



#root mean squared errors 
(test.error.full <- sqrt(mean(errors.price$lmfull)))
(test.error.rel <- sqrt(mean(errors.price$lmrel)))
(test.error.abs <- sqrt(mean(errors.price$lmabs)))
(test.error.lasso <- sqrt(mean(errors.price$lasso.lmin)))
(test.error.knn <- sqrt(mean(errors.price$knn.kmin)))
(test.error.knn2 <- sqrt(mean(errors.price$knn.kmin2)))

(baseline.error.abs <- sqrt(mean((stockdata[a1:(a1+B),]$adj_close_tomorrow-stockdata[a1:(a1+B),]$adj_open_tomorrow) ^ 2))) #always predict 0 change



#plot predictions

#full
par(mfrow = c(3,1))
plot(stockdata[(a1+1):(a1+a2+a3),]$adj_close_tomorrow, ylab="Adjusted Close Prediction LM",type="l",col=2)
lines(predictions.price$lmfull, type="l",col=4)
plot(stockdata[(a1+a2):(a1+a2+200),]$adj_close_tomorrow, ylab="Adjusted Close Prediction LM",type="l",col=2)
lines(predictions.price[a2:(a2+200),]$lmfull, type="l",col=4)
plot(stockdata[(a1+a2):(a1+a2+20),]$adj_close_tomorrow, ylab="Adjusted Close Prediction LM",type="l",col=2)
lines(predictions.price[a2:(a2+20),]$lmfull, type="l",col=4)
par(mfrow = c(1,1))

#rel
par(mfrow = c(3,1))
plot(stockdata[(a1+1):(a1+a2+a3),]$adj_close_tomorrow, ylab="Adjusted Close Prediction LM",type="l",col=2)
lines(predictions.price$lmrel, type="l",col=4)
plot(stockdata[(a1+a2):(a1+a2+200),]$adj_close_tomorrow, ylab="Adjusted Close Prediction LM",type="l",col=2)
lines(predictions.price[a2:(a2+200),]$lmrel, type="l",col=4)
plot(stockdata[(a1+a2):(a1+a2+20),]$adj_close_tomorrow, ylab="Adjusted Close Prediction LM",type="l",col=2)
lines(predictions.price[a2:(a2+20),]$lmrel, type="l",col=4)
par(mfrow = c(1,1))

#abs
par(mfrow = c(3,1))
plot(stockdata[(a1+1):(a1+a2+a3),]$adj_close_tomorrow, ylab="Adjusted Close Prediction LM",type="l",col=2)
lines(predictions.price$lmabs, type="l",col=4)
plot(stockdata[(a1+a2):(a1+a2+200),]$adj_close_tomorrow, ylab="Adjusted Close Prediction LM",type="l",col=2)
lines(predictions.price[a2:(a2+200),]$lmabs, type="l",col=4)
plot(stockdata[(a1+a2):(a1+a2+20),]$adj_close_tomorrow, ylab="Adjusted Close Prediction LM",type="l",col=2)
lines(predictions.price[a2:(a2+20),]$lmabs, type="l",col=4)
par(mfrow = c(1,1))

#lasso
par(mfrow = c(3,1))
plot(stockdata[(a1+1):(a1+a2+a3),]$adj_close_tomorrow, ylab="Adjusted Close Prediction LM",type="l",col=2)
lines(predictions.price$lasso.lmin, type="l",col=4)
plot(stockdata[(a1+a2):(a1+a2+200),]$adj_close_tomorrow, ylab="Adjusted Close Prediction LM",type="l",col=2)
lines(predictions.price[a2:(a2+200),]$lasso.lmin, type="l",col=4)
plot(stockdata[(a1+a2):(a1+a2+20),]$adj_close_tomorrow, ylab="Adjusted Close Prediction LM",type="l",col=2)
lines(predictions.price[a2:(a2+20),]$lasso.lmin, type="l",col=4)
par(mfrow = c(1,1))

#knn
par(mfrow = c(3,1))
plot(stockdata[(a1+1):(a1+a2+a3),]$adj_close_tomorrow, ylab="Adjusted Close Prediction LM",type="l",col=2)
lines(predictions.price$knn.kmin, type="l",col=4)
plot(stockdata[(a1+a2):(a1+a2+200),]$adj_close_tomorrow, ylab="Adjusted Close Prediction LM",type="l",col=2)
lines(predictions.price[a2:(a2+200),]$knn.kmin, type="l",col=4)
plot(stockdata[(a1+a2):(a1+a2+20),]$adj_close_tomorrow, ylab="Adjusted Close Prediction LM",type="l",col=2)
lines(predictions.price[a2:(a2+20),]$knn.kmin, type="l",col=4)
par(mfrow = c(1,1))


#save output
write.csv(predictions.price, file = "predictions-price.csv")
write.csv(errors.price, file = "errors-price.csv")
