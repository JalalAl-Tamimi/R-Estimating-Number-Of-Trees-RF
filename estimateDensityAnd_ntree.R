## Procedures to 
## 1) estimate density of the dataset 
## 2) estimating number of trees needed in a Random Forest from party package

### start by simulating a classification data with some correlated predictors. 
### X1 and X2 are informative predictors that have various levels of correlation
### with non-informtive ones (x3 to x7). X8 is informative and 
### there are no correlated predictors with it
### This complex dataframe with various levels of correlation is common in phonetic research. 
set.seed(1)

n <- 2000 #observations
p <- 10 #predictors
corr1 <- 0.9 #correlation
corr2 <- 0.5 #correlation
corr3 <- 0.1 #correlation
noise <- 0.7 #noise

#simulation of data
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- corr1*x1+rnorm(n)*sqrt(1-corr1^2)#x3 = high correlation with x1
x4 <- corr2*x1+rnorm(n)*sqrt(1-corr2^2)#x4 = medium correlation with x1
x5 <- corr3*x1+rnorm(n)*sqrt(1-corr3^2)#x5 = low correlation with x1
x6 <- corr1*x2+rnorm(n)*sqrt(1-corr1^2)#x6 = high correlation with x2
x7 <- corr3*x2+rnorm(n)*sqrt(1-corr3^2)#x7 = low correlation with x2
x8 = rnorm(n)
z <- x1 + x2 + x8 + noise*rnorm(n)#add some noise to the data. predictors x1, x2 and x8 are informative 
y <- as.factor(rbinom(n, 1, exp(z) / (1 + exp(z))))#create a binomial response for classification


DF <- as.data.frame(replicate(p-8,rnorm(n)))
names(DF) <- paste("x",seq(9,p),sep="")
DF1 <- data.frame(y=y,x1=x1,x2=x2,x3=x3,x4=x4,x5=x5,x6=x6,x7=x7,x8=x8)
DF <- cbind(DF1,DF)

## Density based metric: Adaptation of procedure as developped by:
## Oshiro, T. M., Perez, P. S., & Baranauskas, J. A. (2012). How many trees in a random forest? 
## In International Workshop on Machine Learning and Data Mining in Pattern Recognition (pp. 154–168)


## Three density messures can be computed, D1, D2 or D3
## p = number of predictors; n = number of observations and c = number of classes to classify
## D1 = log_p(n)
## D2 = log_p(n/c)
## D2 = log_p((n+1)/(c+1))

## c below is number of classes to classify
c <- 2
D1 <- log(n,base = p)
D2 <- log(n/c,base = p)
D3 <- log((n+1)/(c+1),base = p)

D1
D2
D3

## below is procedure to estimate number of trees needed
# but use the ones below as they allow for foreach to run properly

## this procedure uses parallel computing on a local machine. 
## This was tested on a Windows 10 machine with 32 GB RAMs and 4 Cores (8 logical through multithreading) 

## start by loading packges and install those that are not installed
requiredPackages = c('foreach','doSNOW','parallel','party','pROC')
for(p in requiredPackages){
  if(!require(p,character.only = TRUE)) install.packages(p)
  library(p,character.only = TRUE)
}

## below to prepare using parallel computing
#detectCores()
NumberOfCluster <- detectCores()
cl <- makeCluster(NumberOfCluster)
registerDoSNOW(cl)

## check number of ntrees. Default for mtry, i.e., round(sqrt(predictors))
treeNumb <- 20
mtry <-round(sqrt(p))
the_uber_formula1 <- as.formula('y ~ .')

## below computes 20 random forests using the "party" package and estimates the accuracy of the model. 
## Then estimates of the AUC (Area Under the Curve) and saves these in a dataframe.
system.time(AUCTestDF <-
              foreach (i=1:treeNumb,.combine=rbind.data.frame,.packages=c("party","pROC")) %dopar% {
                set.seed(1)
                cforest.mdl=cforest(the_uber_formula1,data=DF,
                                    controls = cforest_unbiased(mtry = mtry, ntree = i*100))
                cforest.cond = predict(cforest.mdl,OOB=TRUE)
                auc.cforest = auc(DF[,1],as.numeric(cforest.cond))
              })

## below computes the ROC (Receiver Operating Curve) curves for this dataframe 
## for the 20 random forests with a 100 trees iteration
## and saves these in a list that will be used later to compare ROC curves
system.time(AUCTest <-
              foreach (i=1:treeNumb,.packages=c("party","pROC")) %dopar% {
                set.seed(1)
                cforest.mdl=cforest(the_uber_formula1,data=DF,
                                    controls = cforest_unbiased(mtry = mtry, ntree = i*100))
                cforest.cond = predict(cforest.mdl,OOB=TRUE)
                roc <- roc(DF[,1],as.numeric(cforest.cond))
                })



##stop cluster. If not done R will still use all cores as defined above and the system will become too slow!!
stopCluster(cl)
####

## below provides the highest AUC value of all generated Random Forests. 
which(AUCTestDF == max(AUCTestDF), arr.ind = TRUE)## result is 100 trees (first row)

## below provides a comparison between subsequent curves. The result shows that 100 trees are enough
## to obtain a model with the highest level of accuracy as all subsequent models provide
## no statistical improvement. Even comparison between modell 1 and 3 (the two highest) provides 
## no statistical improvement

roc.test(AUCTest[[1]], AUCTest[[2]])#1 is best model
roc.test(AUCTest[[2]], AUCTest[[3]])
roc.test(AUCTest[[3]], AUCTest[[4]])
roc.test(AUCTest[[4]], AUCTest[[5]])
roc.test(AUCTest[[5]], AUCTest[[6]])
roc.test(AUCTest[[6]], AUCTest[[7]])
roc.test(AUCTest[[7]], AUCTest[[8]])
roc.test(AUCTest[[8]], AUCTest[[9]])
roc.test(AUCTest[[9]], AUCTest[[10]])
roc.test(AUCTest[[10]], AUCTest[[11]])
roc.test(AUCTest[[11]], AUCTest[[12]])
roc.test(AUCTest[[12]], AUCTest[[13]])
roc.test(AUCTest[[13]], AUCTest[[14]])
roc.test(AUCTest[[14]], AUCTest[[15]])
roc.test(AUCTest[[15]], AUCTest[[16]])
roc.test(AUCTest[[16]], AUCTest[[17]])
roc.test(AUCTest[[17]], AUCTest[[18]])
roc.test(AUCTest[[18]], AUCTest[[19]])
roc.test(AUCTest[[19]], AUCTest[[20]])

