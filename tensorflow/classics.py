import tensorflow as tf
import numpy as np
from utils import *
#from gurobipy import *
import cvxpy as cp
from tqdm import tqdm


def zero_forcing(y, H):
    '''
    Inputs: 
    y.shape = [batch_size, N] = [batch_size, 2*NR]
    H.shape = [batch_size, N, K] = [batch_size, 2*NR, 2*NT]
    Outputs:
    s.shape = [batch_size, K] = [batch_size, 2*NT]
    '''
    # Projected channel output
    Hty = batch_matvec_mul(tf.transpose(H, perm=[0, 2, 1]), y)

    # Gramian of transposed channel matrix
    HtH = tf.matmul(H, H, transpose_a=True) 

    # Inverse Gramian
    HtHinv = tf.linalg.inv(HtH)

    #Zero-Forcing Detector
    s = batch_matvec_mul(HtHinv, Hty)

    return s


def MMSE(y, H, noise_sigma):
    '''
    Inputs:
    y.shape = [batch_size, N] = [batch_size, 2*NR]
    H.shape = [batch_size, N, K] = [batch_size, 2*NR, 2*NT]
    noise_sigma.shape = [batch_size]
    Outputs:
    s.shape = [batch_size, K] = [batch_size, 2*NT]
    '''
    # Projected channel output
    Hty = batch_matvec_mul(tf.transpose(H, perm=[0, 2, 1]), y)

    # Gramian of transposed channel matrix
    HtH = tf.matmul(H, H, transpose_a=True) 

    # Inverse Gramian
    HtHinv = tf.linalg.inv(HtH + tf.reshape(tf.math.sqrt(noise_sigma)/2, [-1, 1, 1]) * tf.eye(tf.shape(H)[-1], batch_shape=[tf.shape(H)[0]]))

    # MMSE detector
    s = batch_matvec_mul(HtHinv, Hty)

    return s


# def mlSolver(hBatch, yBatch, Symb):
#     results = []
#     status = []
#     m = len(hBatch[0, 0, :])
#     n = len(hBatch[0, :, 0])
#     for idx_, Y in tqdm(enumerate(yBatch)):
#         H = hBatch[idx_]
#         model = Model('mimo')
#         k = len(Symb)
#         Z = model.addVars(m, k, vtype=GRB.BINARY, name='Z')
#         S = model.addVars(m, ub=max(Symb)+.1, lb=min(Symb)-0.1, name='S')
#         E = model.addVars(n, ub=200.0, lb=-200.0,  vtype=GRB.CONTINUOUS, name='E')
#         model.update()

#         for i in range(m):
#             model.addConstr(S[i] == quicksum( Z[i,j]*Symb[i] for j in range(k)))
#         model.addConstrs( Z.sum(j, '*')==1 for j in range(m), name='Const1')
#         for i in range(n):
#             E[i] = quicksum(H[i][j]*S[j] for j in range(m)) - Y[i][0]

#         obj = E.prod(E)
#         model.setObjective(obj, GRB.MINIMIZE)
#         model.Params.logToConsole = 0
#         model.setParam('TimeLimit', 100)
#         model.update()

#         model.optimize()
#         solution = model.getAttr('X', S)
#         status.append(model.getAttr(GRB.Attr.Status) == GRB.OPTIMAL)
#         if model.getAttr(GRB.Attr.Status)==9:
#                 print(np.linalg.cond(H))
#             x_est = []
#         for nnn in solution:
#             x_est.append(solution[nnn])
#         results.append(x_est)
#     return results, np.array(status)


# def Gurobi_ML(batch_size, hBatch, yBatch, constellation):
#     sHatBatch, status = mlSolver(hBatch, yBatch, constellation)
#     xHatBatch = np.argmin(np.abs(np.reshape(sHatBatch, [-1,1]) - constellation.reshape([-1,1])), axis=0)
#     xHatBatch = xHatBatch.reshape([batch_size, -1])
#     return xHatBatch