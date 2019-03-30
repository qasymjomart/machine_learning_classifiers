

###### First, make sure you have set the working directory to the directory that the data/functions are at
###### sourcing to the related functions. 
source("lda.R")
source("resub.lda.R")
######

###### loading data to workplace. R can read many data formats. This data is in Rdata format.
load("Data.Rdata")
######


###### t.test feature selection
# t.test.results=numeric()
# for (i in 2:71){
	# t.test.results=c(t.test.results,t.test(Data[1:180,i],Data[181:295,i])$p.value)
# }
######

Dat=Data[,c(41,50,72)]  #column 41 corrsponds to KIAA0175 gene and column 50 to ORC6L gene
Dat0=Dat[1:180,1:2]
Dat1=Dat[181:295,1:2]
plot(Dat0,xlim=c(-1,1),ylim=c(-0.75,0.75),col="red",pch=16,xlab="KIAA0175",ylab="ORC6L") #plots labled 0 data
points(Dat1,col="blue",pch=16) #plots labled 1 data on the previous plot


# Find LDA and plot
L = lda(Dat0,Dat1)

# in 2D problem
a = L[c(2,3)]
b = L[1]
slope = -a[1]/a[2]
intercept = -b/a[2]
abline(a=intercept,b=slope,lwd=2)

# Find apparent error
err = resub.lda(Dat0,Dat1,a,b)




