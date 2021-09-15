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
    IsNonLinearObserve , funHz = (0, '')
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
        for i in range(0,self.stateSize):
            self.x[i,:] = x_[i,:]
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

    def observe(self,x):
        if self.IsNonLinearObserve:
            z = self.funHz(x)
            z_predict = np.array( [ [z[0]], [z[1]] ] )
        else:
            z_predict = self.Hz@x

        return z_predict

    def update(self,H_,z_):
        self.Hz = H_
        Ht = self.Hz.T
        temp1 = self.Hz@self.P@Ht + self.Rm
#         temp2 = temp1.inverse() #(H*P*H'+R)^(-1)
        self.K = self.P@Ht@np.linalg.inv(temp1)

        self.z = self.observe(self.x)

        self.x = self.x + self.K@(z_ - self.z)
#         print(self.x)
        self.P = (self.In - self.K@self.Hz)@self.Px
        return self
   
def calA(X):
    n = len(X)
    A = np.zeros([n,n])
    A[0,2],A[0,3],A[1,2],A[1,3]=(math.cos(X[3]),-X[2]*math.sin(X[3]),math.sin(X[3]),X[2]*math.cos(X[3]))
    return A

def calH(X):
    n = len(X)
    m = 2
    H = np.zeros([m,n])
    rho2 = X[0]**2+X[1]**2
    rho = math.sqrt(rho2)
    H[0,0],H[0,1],H[1,0],H[1,1]=(X[0]/rho,X[1]/rho,-X[1]/rho2,X[0]/rho2)
    return H

def calCovXY(x,y,covRhoEl):
    # covXY =  np.zeros([2,2])
    rho2 = math.sqrt(x**2+y**2)
    rho = math.sqrt(rho2)
    PxyPre =  np.array([[x/rho,y/rho],[-y/rho2,x/rho2]])
    covXY = PxyPre@covRhoEl@PxyPre.T
    return covXY

def calExpAt(A,dt):
    n = len(A)
    I = np.eye(n)
    # return I+A*dt+A@A/2*dt*dt+A@A@A/6*dt*dt*dt
    return I+A*dt

#########################################################################
# 系统设置，初值估计
#########################################################################
measureStep = 0.01
tf = 10
lth = 1 + int(tf/measureStep)
ts = np.linspace(0,tf,lth)

x0_real = np.array([5.0,12.0,0.8,-0.64])     # 这是设定中的真实状态初始值
# x0 = np.array([[x1],[x2]]) 
dx0 = np.array([0.1,0.2,0.01,-0.04])         # 我这里随意设置了一个偏差
x0 = x0_real + dx0                           # 初值
x0 = x0[:,np.newaxis]


# A = calA([4,3,1,0.5])                        # system matrix
# H = np.array([])                             # 注意size
Gamma = np.zeros([4,2])
Gamma[2,0],Gamma[3,1] = (1,1)
# z0 = np.array([5,0.64])
# z0 = z0[:,np.newaxis]
#########################################################################
# 先验统计误差，
#########################################################################
dvth = np.array([0.8,0.1])              # pertubation 0.8[m/s]/s,0.1[rad]/s
sigmaQv = dvth*measureStep              # estimated pertubation of dynamical system per step, 
Q = np.eye(2)
Q[0,0] = Q[0,0]*sigmaQv[0]**2           # pertubation from white noise, may be from ？？？
Q[1,1] = Q[1,1]*sigmaQv[1]**2

# dxy = 0.05                            # 二维平面内的定位误差，$x,y$而非$v,th$
sigmaRm = np.array([0.05,1.0/57.3])      # a priori precision of measurement sigma = 0.05[m],1[deg]
R = np.array([[sigmaRm[0]**2,0],[0,sigmaRm[1]**2]])            # covariance of measurement precision
CovXY = calCovXY(x0[0,0],x0[1,0],R)

P0 =  np.eye(4)                         # error of state, estimated at start
P0[0,0] = 100*P0[0,0]*CovXY[0,0]        
P0[1,1] = 100*P0[1,1]*CovXY[1,1]
P0[2,2] = 10*P0[2,2]*sigmaQv[0]**2         
P0[3,3] = 10*P0[3,3]*sigmaQv[1]**2


#########################################################################
# 采用准确的系统方程，模拟真实的状态
#########################################################################

#-----------------------------------------------------------------------------
"""A variety of methods to solve first order ordinary differential equations.

AUTHOR:
    Jonathan Senning <jonathan.senning@gordon.edu>
    Gordon College
    Based Octave functions written in the spring of 1999
    Python version: March 2008, October 2008
"""
#-----------------------------------------------------------------------------

def euler( f, x0, t ):
    """Euler's method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = euler(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """

    n = len( t )
    x = np.array( [x0] * n )
    for i in range( n - 1 ):
        x[i+1] = x[i] + ( t[i+1] - t[i] ) * f( x[i], t[i] )

    return x

#-----------------------------------------------------------------------------

def system( X, t ):
    eta1, eta2 = np.random.normal(0,sigmaQv*sigmaQv,2)
    return np.array([X[2]*math.cos(X[3]),X[2]*math.sin(X[3]),eta1,eta2/X[2]])# no control 

# ts = np.linspace( 0,tf,1+int(tf/measureStep))

# compute various numerical solutions
x_euler = euler( system, x0_real, ts )


#########################################################################
# 采用先验的测量方程，模拟带有噪声的测量数据
#########################################################################
def radarRangeElevation(x):
    return np.array([math.sqrt(x[0]**2+x[1]**2),math.atan(x[1]/x[0])])

def measurement( ts , x_euler , z0):
    n = len( ts )
    z = np.array( [z0] * n )
    for i in range (n):
        w1, w2 = np.random.normal(0,sigmaRm*sigmaRm,2)
        z[i,0] = math.sqrt(x_euler[i,0]**2+x_euler[i,1]**2) + w1
        z[i,1] = math.atan(x_euler[i,1]/x_euler[i,0])+ w2

    return z

z0 = np.array([5.0,0.64])               # 对应于准确初始状态的准确测量结果
z_ts = measurement( ts , x_euler , z0)


plt.subplot(2,1,1)
plt.plot(ts,z_ts[:,0])
plt.ylabel('rho/m')
plt.title('radar observation data')

plt.subplot(2,1,2)
plt.plot(ts,z_ts[:,1])
plt.xlabel('t/s')
plt.ylabel('theta/rad')
plt.show()


#########################################################################
# EKF
#########################################################################

# 创建并初始化
radar = KalmanFilter(4,2,0)
# 设置先验误差矩阵
radar.init(x0,P0,Q,R)
radar.G  = Gamma
radar.IsNonLinearObserve = 1
radar.funHz = radarRangeElevation
# 保存数据
# P_ts = P[:,:,np.newaxis]
x_ts = np.zeros([lth,4])

for it in range(0,lth):
    x_ = radar.x.T[0]                     # 转化为行向量（1维Array）再输入计算矩阵
    A = calA(x_)
    H = calH(x_)
    radar.predict(calExpAt(A,measureStep))
    radar.update(H,z_ts[it,:,np.newaxis])
    x_ts[it,:] = radar.x.T[0]
    # x_ts[0,it] = dopp.x[0],x_ts[1,it] = dopp.x[1]
    

#########################################################################
# 绘图
#########################################################################
plt.plot(x_ts[:,0],x_ts[:,1])
plt.plot(x_euler[:,0],x_euler[:,1])
plt.ylabel('y/m')
plt.xlabel('x/m')
plt.legend(['estimated data','real data'])
plt.title('radar trajectory estimate result')
plt.show()

plt.subplot(2,1,1)
plt.plot(ts,x_ts[:,0]-x_euler[:,0])
plt.ylabel('dx/m')
plt.subplot(2,1,2)
plt.plot(ts,x_ts[:,1]-x_euler[:,1])
plt.ylabel('dy/m')
plt.xlabel('t/[s]')
plt.title('estimatation error')
plt.show()

