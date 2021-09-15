import numpy as np
import math
# import scipy.stats as st
# import scipy
# import matplotlib as mpl
import matplotlib.pyplot as plt

class KalmanFilter:
    stateSize = 1
    measureSize = 1
    controlSize = 0
    x,z,A,B,G,u,Px,K,Hz,Qv,Rm = [0,0,0,0,0,0, 0,0,0,0,0]
    
    def __init__(self,n,m,nc):
        self.stateSize = n
        self.measureSize = m
        self.controlSize = nc
        self.x = np.zeros([n,1])
        self.z = np.zeros([m,1])
        
        self.In = np.eye(n)
        self.A = np.eye(n)         # state transition matrix, discrete
        if nc>0:
            self.u = np.zeros([nc,1])
            self.B = np.zeros(n,nc)
        
        self.G = np.eye(n)
        self.Px = np.eye(n)*1e-6
        self.Qv = 1e-6*np.eye(n)  # process variance
        self.K  = np.zeros([n,m]) # kalman gain
        self.Hz = np.zeros([m,n])
        self.Rm = 1e-6*np.eye(m)  # estimate of measurement variance,
    
    def init(self,x_,P_,Q_,R_):
        self.x = x_
        self.Px = P_
        self.Qv = Q_
        self.Rm = R_
        return self
    
    def predict(self,A_):
        self.A = A_
        self.x = self.A@self.x
#         print(self.x)
        self.P = self.A@self.Px@self.A.T + self.G@self.Qv@self.G.T 
        return self
    
    def update(self,H_,z_):
        self.Hz = H_
        Ht = self.Hz.T
        temp1 = self.Hz@self.P@Ht + self.Rm
#         temp2 = temp1.inverse() #(H*P*H'+R)^(-1)
        # if self.measureSize==1:
        #     self.K = self.P@Ht@np.linalg.inv(temp1)
        # else:
        self.K = self.P@Ht@(np.linalg.inv(temp1))
        self.z = z_
        z_predict = self.Hz@self.x
        self.x = self.x + self.K@(self.z - z_predict)
#         print(self.x)
        self.P = (self.In - self.K@self.Hz)@self.Px
        return self
    
# ////////////////////////////////////////////////////////////////////# 
# SEC 2, SYSTEM# 
# ////////////////////////////////////////////////////////////////////# 
def calExpAt(A,dt):
    n = len(A)
    I = np.eye(n)
    return I+A*dt+A@A/2*dt*dt

#########################################################################
# 系统设置，初值估计
#########################################################################
measureStep = 0.1;
tf = 30;
lth = 1 + int(tf/measureStep);
ts = np.linspace(0,tf,lth)

A = np.array([[0,1.0],[0,0]])           # system matrix
H = np.array([[0,1.0]])                 # 注意size

x0 = np.array([0,1.2])                    # 我这里随意设置了一个初值，与真实值相差较大
x0 = x0[:,np.newaxis]
# x0 = np.array([[x1],[x2]]) 
#########################################################################
# 先验统计误差，
#########################################################################
sigmaQv = (1,0.1)                     # estimated pertubation of dynamical system per step
                                        # 如果第一步的初始状态估计误差就很大，这里就需要设置稍大；否则，收敛将很慢
Q = np.eye(2)
Q[0,0] = Q[0,0]*sigmaQv[0]**2           # pertubation from white noise, may be from 
Q[1,1] = Q[1,1]*sigmaQv[1]**2
sigmaRm = 0.1                           # a priori precision of measurement
R = np.array([[sigmaRm**2]])            # covariance of measurement precision

P0 = np.eye(2)                          # error of state, estimated at start
P0[0,0] = 10*P0[0,0]
P0[1,1] = 3*P0[1,1]*sigmaRm**2                                 

#########################################################################
# 采用先验误差设定，模拟真实的、带有噪声的测量数据
#########################################################################
# z_ts = np.linspace(0,18,lth)+np.random.normal(0,sigmaRm,lth)  # 匀加速
v = 1.2
z_real = v*np.ones((lth))               # 一维数组，尚未扩充维度到2
z_ts = z_real+np.random.normal(0,sigmaRm*sigmaRm,lth)
z_ts = z_ts[np.newaxis, :]              # 作为行向量输入

# ////////////////////////////////////////////////////////////////////# 
# SEC 3, ESTIMATE/# 
# ////////////////////////////////////////////////////////////////////# 

# 创建并初始化
dopp = KalmanFilter(2,1,0)
# 设置先验误差矩阵
dopp.init(x0,P0,Q,R)

# 保存数据
# P_ts = P[:,:,np.newaxis]
x_ts = np.zeros([lth,2])

for it in range(0,lth):
    dopp.predict(calExpAt(A,measureStep))
    dopp.update(H,z_ts[:,it,np.newaxis])
    x_ts[it,:] = dopp.x.T[0]
    # x_ts[0,it] = dopp.x[0],x_ts[1,it] = dopp.x[1]
    
# 绘图
plt.subplot(2,1,1)
plt.plot(ts,x_ts[:,1])
plt.plot(ts,z_ts[0])
plt.legend(['filtered data', 'measure data'])
plt.title('Doppler velocity estimate result')

plt.subplot(2,1,2)
plt.plot(ts,x_ts[:,1]-z_real)
plt.xlabel('t/[s]')
plt.title('estimatation error')
plt.show()

