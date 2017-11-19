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


# Run first the knn-and-lm_final.R script in order to generate the inputs for this one
rnn.results <- read.csv("RNN_1_results_AAPL.csv", header = TRUE)
stockdata <- read.csv("aapl_time_data.csv", header = TRUE)

rnn.results <- read.csv("RNN_1_results_GOOGL.csv", header = TRUE)
stockdata <- read.csv("googl_time_data.csv", header = TRUE) 

rnn.results <- read.csv("RNN_1_results_AMZN.csv", header = TRUE)
stockdata <- read.csv("amzn_time_data.csv", header = TRUE) 

rnn.results <- read.csv("RNN_1_results_VRX.csv", header = TRUE)
stockdata <- read.csv("vrx_time_data.csv", header = TRUE)

rnn.results <- read.csv("RNN_1_results_WMT.csv", header = TRUE)
stockdata <- read.csv("wmt_time_data.csv", header = TRUE) 



a1 <- round(0.6*nrow(stockdata)) #nr of days for training price prediction
a2 <- round(0.25*nrow(stockdata)) #nr of days for training error prediction
a3 <- nrow(stockdata) - a1 - a2 - 1



head(rnn.results)

rnn.results <- data.frame(rnn.results[6:nrow(rnn.results),1:2])
colnames(rnn.results)  <- c("adj_close_tomorrow_pred","adj_close_tomorrow_true")

rnn.results$adj_close_tomorrow_pred <- as.numeric(as.character(rnn.results$adj_close_tomorrow_pred))
rnn.results$adj_close_tomorrow_true <- as.numeric(as.character(rnn.results$adj_close_tomorrow_true))

rnn.errordata <- stockdata[1639:(nrow(stockdata)-2),]
rnn.errordata["error_abs"] <- sqrt((rnn.results$adj_close_tomorrow_pred - rnn.results$adj_close_tomorrow_true)^2)

head(rnn.errordata)




modnr.p <- 2    #nr of models 


rnn.error.pred<- data.frame(matrix(rep(0, modnr.p*a3),ncol=modnr.p))
colnames(rnn.error.pred)<- c("lm.abs","x")

rnn.error.error <-  data.frame(matrix(rep(0, modnr.p*a3),ncol=modnr.p))
colnames(rnn.error.error)<- c("lm.abs","x")


C <- a3
for (c in 0:(C-1)) {
  m <- a2+c
  
  #lm absolute error

  rnn.lm.error.abs <- lm(error_abs~ sentiment_n+sentiment_p+adj_close+adj_open_tomorrow+relative_overnight_tomorrow+vol_m_ave_10, data = rnn.errordata[1:m, ])
  
  
  rnn.error.pred[c+1,]$lm.abs<- predict(rnn.lm.error.abs, newdata = rnn.errordata[m+1, ])
  rnn.error.error[c+1,]$lm.abs<- ((rnn.errordata[m+1, ]$error_abs - rnn.error.pred[c+1,]$lm.abs) ^ 2)
  
}

summary(rnn.lm.error.abs)


(rnn.test.error.error <- sqrt(mean(rnn.error.error$lm.abs)))
(baseline.errorpred <- sqrt(mean((rnn.errordata[a2:(a2+a3), ]$error_abs - mean(rnn.errordata$error_abs[1:a2])) ^ 2)) )



#plot
par(mfrow = c(1,1))

plot(rnn.errordata[(a2+1):nrow(rnn.errordata),]$error_abs, ylab="Error Prediction LM",type="l",col=2)
lines(rnn.error.pred$lm.abs, type="l",col=4)
