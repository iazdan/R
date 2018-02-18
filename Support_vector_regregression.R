#Call data

rm(list=ls())
#Reading the data set
Data <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data")
Data <- as.matrix(Data, ncol=14)
Y <- Data[,14]
X <- as.matrix(Data[,1:13],ncol=13)
N <- length(Y)
print(c(N, dim(X)))

require('quadprog')
## Defining the Gaussian kernel
rbf_kernel <- function(x1,x2,gamma){
  K<-exp(-(1/gamma^2)*t(x1-x2)%*%(x1-x2))
  return(K)
}

gamma<-2.5
rbf_kernel(X[1,],X[2,],gamma)


svmtrain <- function(X,Y,C=Inf, gamma=1.5,esp=1e-10){
  #difining H matrix according to the description provided in the project document
  N<-length(Y)
  H<-matrix(0,N,N)
  X<-as.matrix(X);Y<-as.vector(Y)
  
  for(i in 1:N){
    for(j in 1:N){
      H[i,j]<-Y[i]*Y[j]*rbf_kernel(X[i,],X[j,],gamma)
    }
  }
  #defining Dm in terms of H, Dm=[[H, -H],[-H,H]]
  Dm<-matrix(0,2*N,2*N)
  Dm <- cbind(rbind(H, -H),rbind(-H, H))
  Dm<-Dm+diag(2*N)*1e-5 # adding a very small number to the diag, some trick
  
  #defining dv which contains a vector including 1 and -1
  epsilon =0.1e-5
  dv<-t(c(rep(1,N), rep(-1,N))) + epsilon
  #Number of equality constraints is
  meq<-1
  #defining Am and bv
  Am<-cbind(matrix(c(Y,-Y),2*N),diag(2*N))
  bv<-rep(0,1+2*N) # the 1 is for the sum(alpha)==0, others for each alpha_i >= 0
  #Defining Am and bv when C isnot infinity
  if(C!=Inf){
    # an upper bound is given
    Am<-cbind(Am,-1*diag(2*N))
    bv<-c(cbind(matrix(bv,1),matrix(rep(-C,2*N),1)))
  }
  #since we have alpha and plpah*, we subtract them to find the actual alphas
  alpha_org<-solve.QP(Dm,dv,Am,meq=meq,bvec=bv)$solution
  alpha_org <- alpha_org[1:N] - alpha_org[N+1:2*N]
  indx<-which(alpha_org>esp,arr.ind=TRUE)
  alpha<-alpha_org[indx]
  nSV<-length(indx)
  if(nSV==0){
    throw("QP is not able to give a solution for these data points")
  }
  Xv<-X[indx,]
  Yv<-Y[indx]
  Yv<-as.vector(Yv)
  ## choose one of the support vector to compute b. Instead of using an arbitrary Support 
  ##Vector xs, it is better to take an average over all of the Support Vectors in S

  b <- numeric(nSV)
  ayK <- numeric(nSV)
  for (i in 1:nSV){
    for (m in 1:nSV){
      ayK[m] <- alpha[m]*Yv[m]*rbf_kernel(Xv[m,],Xv[i,],gamma)
    }
    b[i]<-Yv[i]-sum(ayK)
    
  }
  w0 <- mean(b)
  
  #list(alpha=alpha, wstar=w, b=w0, nSV=nSV, Xv=Xv, Yv=Yv, gamma=gamma)
  list(alpha=alpha, b=w0, nSV=nSV, Xv=Xv, Yv=Yv, gamma=gamma)
}



### Predict the class of an object X


#predicting the result on the test sample
svmpredict <- function(x,model){
  alpha<-model$alpha
  b<-model$b
  Yv<-model$Yv
  Xv<-model$Xv
  nSV<-model$nSV
  gamma<-model$gamma
  #wstar<-model$wstar
  #result<-sign(rbf_kernel(wstar,x,gamma)+b)
  ayK <- numeric(nSV)
  for(i in 1:nSV){
    ayK[i]<-alpha[i]*Yv[i]*rbf_kernel(Xv[i,],x,gamma)
  }
  result <- sum(ayK)+b
  return(result)
}

model23 <- svmtrain(X,Y,C=1, gamma=5,esp=1e-10)
model23

#X[1,] is a sample from the data set
z <- X[1,]

svmpredict(z,model23)
