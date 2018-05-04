#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class wavelet(object):
    def __init__(self):
        pass

    def pas(self,h,kk=None):
        """根据滤波器h的值，通过kk次迭代计算尺度函数的采样值。初始值是一个常数"""
        if kk is None:
            kk=11
        '''为什么要乘以2？
        首先若满足加细方程，h(n)的和是sqrt(2),而且每一迭代公式中要乘以sqrt(2)，所以相当于对
        p乘以一个和为2的序列
        '''
        h2=h*2/np.sum(h)
        K = len(h2)-1
        S = 128
        p = np.zeros((1,3*S*K))+1
        p = np.append(p,0)/(3*K)
        hu=self.upsam(h2,S)
        for i in range(kk):
            p = self.dnsample(self.conv(hu,p))
            # plt.axis([0,K*S+2,-0.5,1.4])
            plt.plot(p)
            plt.show()
        p=p[0:K*S]
        L = len(p)
        x = np.linspace(1,L,L)/S
        plt.axis(0,3,-0.5,1.4)
        plt.plot(x,p)
        plt.show()

    def getPhi_Iteration(self,h,maxGen=15):
        '''采用迭代计算的方法求解小波的基函数'''
        timeInterval=0.01
        L=len(h)
        x=np.arange(0,L,timeInterval)
        intervalNum=1/timeInterval
        h1=h*2/sum(h)
        h1=self.upsample(h1,intervalNum-1)

        phi=(np.zeros(x.shape)+1)
        A0=4
        phi=phi*A0/sum(phi*timeInterval)

        for i in range(maxGen):
            phi=self.downsample(self.conv(h1,phi))
            L1=phi.shape[0]
            if L1<len(x):
                phi=np.hstack((phi,np.zeros(len(x)-L1)))
            plt.plot(x,phi)
            plt.show()

    def getPhi_Matirx(self,h,kk):
        '''采样矩阵求解的策略求解尺度函数各点上的值'''
        L=len(h)
        h2=h*2/sum(h)
        M=np.zeros((L,L))
        for i in range(L):
            index=0
            for j in range(i*2,-1,-1):
                if j>=L:
                    index+=1
                    continue
                else:
                    M[i,j]=h2[index]
                    index+=1
                    if(index>=L):
                        break
        M[L-1,:]=1
        b=np.zeros((L,1))
        b[L-1,0]=1
        p=np.linalg.inv(M).dot(b)












    def upsample(self,x,S):
        """上采样，各项之间添加S个0"""
        tmp = np.reshape(x,(1,-1))
        L=tmp.shape[1]
        y = np.row_stack((x, np.zeros((int(S), L)))).T
        y = np.reshape(y, (1, -1))
        return y[0]

    def downsample(self,x):
        """下采样，移除向量中的偶数项"""
        x=np.reshape(x,(1,-1))
        L=x.shape[1]
        y=x[0,0:L:2]
        return y

    def merge(self,d1,d2):
        """对x和y交错融合，例如x=[1,2],y=[3,4],返回[1,3,2,4]"""
        x=d1.copy()
        y=d2.copy()
        L_x=len(x)
        L_y=len(y)
        if L_x<L_y:
            p=np.zeros((L_y-L_x))
            x=np.hstack((x,p))
            x = np.row_stack((y, x)).T
        elif L_x>L_y:
            p=np.zeros((L_x-L_y))
            y=np.hstack((y,p))
            x = np.row_stack((x, y)).T

        x=np.reshape(x,(1,-1))
        x=x[0,0:L_x+L_y]
        return x

    def conv(self,x,y):
        x=np.reshape(x,(1,-1))
        y=np.reshape(y,(1,-1))
        L=y.shape[1]
        l1=x.shape[1]+y.shape[1]-1 #返回值长度
        ans = np.zeros((1,l1))
        for i in range(L):
            pre=np.zeros((1,i))
            tmp= np.hstack((pre,x))
            post=np.zeros((1,l1-tmp.shape[1]))
            tmp= np.hstack((tmp,post))
            ans=ans+tmp*y[0,i]
        return ans[0]

    def conv2(self,x,y):
        L1=len(x)
        L2=len(y)
        ans=np.zeros(L1+L2-1)
        for i in range(L1+L2-1):
            arr1=np.zeros(L1)
            for j in range(L1):
                n=i-j
                if(n>=0 and n<L2):
                    arr1[j]=y[n]
            ans[i]=sum(arr1*x)
        return ans





if __name__=="__main__":
    a=wavelet()
    # h=[0.2303778,0.7148465,0.6308807,-0.0279837,-0.18703481,0.03084138,0.032883011,-0.010597401]
    h=np.array([0.4829626,0.8365163,0.2241438,-0.1294095])
    print(a.getPhi_Matirx(h,14))
