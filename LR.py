import math 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sklearn.metrics as mt
def sigmoid_function(z):
    return 1/(1+math.exp(-z))
def cost_function(ds,w):
    ttl_cst=0
    for x,y in ds:
        x=np.array(x)
        err=sigmoid_function(w.T.dot(x))
        ttl_cst+=abs(y-err)
    return ttl_cst
def SGD(ds,w):
    idx=random.randint(0,len(ds)-1)
    x,y=ds[idx]
    x=np.array(x)
    err=sigmoid_function(w.T.dot(x))
    return (err-y)*x
def LR(ds):
    w=np.zeros(4)
    lim=3000
    ur=0.1
    cst=[]
    for i in range(lim):
        curr_cst=cost_function(ds,w)
        if i%100==0:
            print("epoch = "+str(i/100 + 1)+" current_cost = ",curr_cst)
        cst.append(curr_cst)
        w=w-ur*SGD(ds,w)
        ur=ur*0.98
    return w,(lim,cst)
def main():
    df=pd.read_csv("data/data_banknote_authentication.txt",sep=",",header=None)
    X,Y=df.iloc[:,:-1].values,df.iloc[:,-1].values
    na=[]
    for idx,x in enumerate(X):
        na.append((x,Y[idx]))    
    na=np.array(na)
    w=LR(na)
    ps=[v[0] for v in na]
    lbl=[v[1] for v in na]
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    tpx=[]
    for idx,lbl_val in enumerate(lbl):
        px=ps[idx][0]
        py=ps[idx][1]
        tpx.append(px)
        if lbl_val==1:
            ax1.scatter(px,py,c='b',marker="o",label='O')
        else:
            ax1.scatter(px,py,c='r',marker="x",label='X')
    l=np.linspace(min(tpx),max(tpx))
    a,b=(-w[0][0]/w[0][1],w[0][0])
    ax1.plot(l,a*l+b,'g-')
    plt.show()
    lim = w[1][0]
    cst = w[1][1]
    w=w[0]
    predicted_Y=[]
    actual_Y=[]
    for X,Y in na:
        actual_Y.append(Y)
        predicted_Y.append(sigmoid_function(w.T.dot(X)))
    predicted_Y=np.asarray(predicted_Y)
    predicted_Y=predicted_Y > 0.5
    print ("Accuracy : ",str(mt.accuracy_score(actual_Y,predicted_Y)*100)[:5],"%")
if __name__ == '__main__':
    main()
