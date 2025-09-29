import numpy as np
import copy
x_axis=np.array([[2],[15],[20],[49],[59],[78],[61],[43],[49],[56]])
y_axis=np.array([0,0,0,0,1,1,1,1,0,1,1])
w=np.zeros(x_axis.shape[1])
# sigmoid function to convert the lenear to more better graph
def sigmoid(z):
    g=1/(1+np.exp(-z))
    return g
# lenear equation
def func(w,b):
    x=int(input("enter the size of tumar"))
    n=np.dot(w,x)+b
    return n
# cost function 
def function_cost(x,w,b,y):
    l=x.shape[0]
    cost=0
    for i in range(l):
        fun=np.dot(x,w)+b
        erro=sigmoid(fun)
        cost+= -y[i]*np.log(erro)-(1-y[i])*np.log(erro)
    cost=cost/l
    return cost
def grafient_desent(w,b,x,y):
    m,n=x.shape
    wj=np.zeros((n))
    bj=0
    for i in range(m):
        erro=sigmoid(np.dot(x[i],w)+b)-y[i]
        for j in range(n):
            wj[j]=wj[j]+(erro*x[i,j])/m
        bj=bj+(erro)/m
    return wj,bj
def gradient_desent(wj,bj,x,y,times,alpha):
    w=copy.deepcopy(wj)
    b=bj
    for i in range(times):
        jw,jb=grafient_desent(w,b,x,y)
        b=b-alpha*jb
        w=w-alpha*jw
    return w,b
w_final,b_final=gradient_desent(w,0,x_axis,y_axis,1000,1e-7)
print("final w: ",w_final,"final b: ",b_final)
print("Final cost: ",function_cost(x_axis,w_final,b_final,y_axis))
e=func(w_final,b_final)
print("your prediction is : ",sigmoid(e))