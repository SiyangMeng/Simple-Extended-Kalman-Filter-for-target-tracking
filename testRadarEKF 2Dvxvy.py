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
    A[0,2],A[1,3]=(1,1)
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
# ???????????????????????????
#########################################################################
measureStep = 0.01
tf = 10
lth = 1 + int(tf/measureStep)
ts = np.linspace(0,tf,lth)

x0 = np.array([1000.0,1500.0,5.0,-3.0])                 # ???????????????????????????????????????????????????????????????
# x0 = np.array([4.0,3.0,0.8000,0.35])         # ???????????????????????????????????????
x0 = x0[:,np.newaxis]
# x0 = np.array([[x1],[x2]]) 

# A = calA([4,3,1,0.5])                        # system matrix
# H = np.array([])                             # ??????size
# z0 = np.array([5,0.64])
# z0 = z0[:,np.newaxis]
#########################################################################
# ?????????????????????
#########################################################################
Gamma = np.eye(4)
# Gamma = np.zeros([4,2])
# Gamma[2,0],Gamma[3,1] = (1,1)

dvth = np.array([20.0,20.0,2.0,2.0])    # pertubation 0.1[m/s]/s,0.5[rad]/s
sigmaQv = dvth*measureStep              # estimated pertubation of dynamical system per step, 
Q = np.eye(4)
Q[0,0] = Q[0,0]*sigmaQv[0]**2           # pertubation from white noise, may be from ?????????
Q[1,1] = Q[1,1]*sigmaQv[1]**2
Q[2,2] = Q[2,2]*sigmaQv[2]**2
Q[3,3] = Q[3,3]*sigmaQv[3]**2

# dxy = 0.05                            # ?????????????????????????????????$x,y$??????$v,th$
sigmaRm = np.array([3.16,0.0316])       # a priori precision of measurement
R = np.array([[sigmaRm[0]**2,0],[0,sigmaRm[1]**2]])            # covariance of measurement precision
CovXY = calCovXY(x0[0,0],x0[1,0],R)
# R = CovXY

P0 =  np.eye(4)                         # error of state, estimated at start
# P0[0,0] = P0[0,0]*CovXY[0,0]        
# P0[1,1] = P0[1,1]*CovXY[1,1]
# P0[2,2] = 10*P0[2,2]*sigmaQv[0]**2         
# P0[3,3] = 10*P0[3,3]*sigmaQv[1]**2
P0[0,0] = P0[0,0]*2       
P0[1,1] = P0[1,1]*2
P0[2,2] = P0[2,2]*0.3        
P0[3,3] = P0[3,3]*0.2

#########################################################################
# ???????????????????????????????????????????????????
#########################################################################
def system( X, t ):
    # dx1, dy1, eta1, eta2 = np.random.normal(0,sigmaQv*sigmaQv,4)
    # return np.array([X[2]+dx1,X[3]+dy1,eta1,eta2])# no control 
    return np.array([X[2],X[3],0.,0.])# no control 

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

x0_real = np.array([x0[0,0],x0[1,0],x0[2,0],x0[3,0]])     # ???????????????????????????????????????
# ts = np.linspace( 0,tf,1+int(tf/measureStep))

# compute various numerical solutions
x_euler = euler( system, x0_real, ts )


#########################################################################
# ???????????????????????????????????????????????????????????????
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

z0 = np.array([13.0,1.2])               # ????????????????????????????????????????????????
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

# ??????????????????
radar = KalmanFilter(4,2,0)
# ????????????????????????
radar.init(x0,P0,Q,R)
radar.G  = Gamma
radar.IsNonLinearObserve = 1
radar.funHz = radarRangeElevation
# ????????????
# P_ts = P[:,:,np.newaxis]
x_ts = np.zeros([lth,4])

for it in range(0,lth):
    x_ = radar.x.T[0]                     # ?????????????????????1???Array????????????????????????
    A = calA(x_)
    H = calH(x_)
    radar.predict(calExpAt(A,measureStep))
    radar.update(H,z_ts[it,:,np.newaxis])
    x_ts[it,:] = radar.x.T[0]
    # x_ts[0,it] = dopp.x[0],x_ts[1,it] = dopp.x[1]
    

#########################################################################
# ??????
#########################################################################
plt.plot(x_euler[:,0],x_euler[:,1])
plt.plot(x_ts[:,0],x_ts[:,1])
plt.ylabel('y/m')
plt.xlabel('x/m')
plt.legend(['real data', 'estimated data'])
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

