import numpy as np
import string
import time

def CalJointprb(X,Y,Z,alpha,beta,gamma,theta,pho):
    """

    :param X: matrix X
    :param Y: row vector, val(Y)={0,1}
    :param Z: matrix Z
    :return: Joint prb P(X,Y,Z)
    """
    assert alpha>0 and beta>0 and gamma>0 and theta>0 and pho>0, "All parameters must be set > 0!"
    alpha_hat, beta_hat, gamma_hat, theta_hat, pho_hat = np.log(alpha),np.log(beta),np.log(gamma),\
                                                         np.log(theta),np.log(pho)

    A,B,C,D,F = CalA(Z),CalB(Z),CalC(Z),CalD(X,Z),CalF(X,Y)
    pow = A*alpha_hat + B*beta_hat + C*gamma_hat + D*theta_hat + F*pho_hat
    Joint_Prb = np.exp(pow)
    return Joint_Prb

def CalCondprb(X,Y,Z,alpha,beta,gamma,theta,pho):
    num = CalJointprb(X,Y,Z,alpha,beta,gamma,theta,pho)
    num_map = {'0': 0, '1': 1}
    N_y = Y.shape[0]
    M_z = Z.shape[0]
    N_z = Z.shape[1]
    den1,den2 = 0,0
    for Y_ in Gen_Y_Matrix(N_y,num_map):
        F = CalF(X,Y_)
        den1 += np.power(pho,F)
    for Z_ in Gen_Z_Matrix(M_z,N_z,num_map):
        A_,B_,C_,D_ = CalA(Z_), CalB(Z_), CalC(Z_), CalD(X,Z_)
        den2 += np.power(alpha,A_)*np.power(beta,B_)*np.power(gamma,C_)*np.power(theta,D_)
    return num/den1/den2
def CalA(Z):
    return np.sum(Z==1)

def CalB(Z):
    res = 0
    for currentCol in xrange(Z.shape[1]-1):
        res += np.sum(Z[:,currentCol]==Z[:,currentCol+1])
    return res

def CalC(Z):
    res = 0
    for currentRow in xrange(Z.shape[0]-1):
        res += np.sum(Z[currentRow]==Z[currentRow+1])
    return res

def CalD(X,Z):
    return np.sum(X==Z)

def CalF(X,Y):
    res = 0
    for row in xrange(Y.shape[0]):
        delta = Y[row] == 1 and np.sum(X[row,:] == 1) or 0
        res += delta
    return res

def Gen_Y_Matrix(n,num_map):
    for i in xrange(2**n):
        num = list(bin(i)[2:])
        Y_list = map(lambda x:num_map[x], num)
        if len(Y_list)<n:
            Y_list = Y_list[::-1]
            while len(Y_list) != n:
                Y_list.append(0)
            Y_list = Y_list[::-1]
        Y = np.array(Y_list)
        yield Y


def Gen_Z_Matrix(m,n,num_map):
    for i in xrange(2**(m*n)):
        num = list(bin(i)[2:])
        Z_list = map(lambda x:num_map[x], num)
        if len(Z_list)<m*n:
            Z_list = Z_list[::-1]
            while len(Z_list)!=m*n:
                Z_list.append(0)
            Z_list = Z_list[::-1]
        Z = np.array(Z_list).reshape((m,n))
        yield Z

X = np.array([[0,0,1],[0,1,1],[0,0,1]])
Y = np.array([1,0,0]) # Y.shape = (3,)
Z = np.array([[0,1,1],[1,1,1],[0,0,1]])
alpha, beta, gamma, theta, pho = 1.3, 1.3, 1.4, 1.0, 0.8
alpha1,beta1,gamma1,theta1,pho1 = 0.5,1.5,1.0,0.8,1.2


prb1 = CalCondprb(X,Y,Z,alpha,beta,gamma,theta,pho)
prb2 = CalCondprb(X,Y,Z,alpha1,beta1,gamma1,theta1,pho1)
print prb1, prb2