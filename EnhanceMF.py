import numpy

numpy.random.seed(3)


def matrix_factorization(A, K, beta=0.04, landa=0.01, loop=5000):
    W = numpy.random.rand(user, K)
    H = numpy.random.rand(K, item)
    print(W, H)
    cur = 0
    while cur < loop:
        for u in range(0, user):
            for i in range(0, item):
                if A[u][i] > 0:
                    r3 = 0
                    for k in range(K):
                        r3 += numpy.dot(W[u][k], H[k][i])
                    eui = A[u][i] - r3
                    for k in range(K):
                        W[u][k] += beta * (2 * eui * H[k][i] - landa * W[u][k])
                        H[k][i] += beta * (2 * eui * W[u][k] - landa * H[k][i])
        cur += 1
    return W, H


A = numpy.array([
    [5, 3, 1, 0, 3],
    [3, 2, 1, 3, 2],
    [2, 4, 2, 2, 0],
    [0, 3, 0, 0, 4]
])

user = A.shape[0]
item = A.shape[1]

W, K = matrix_factorization(A, 3)
Y = numpy.dot(W, K)
C = numpy.array([[round(Y[i][j]) for j in range(item)] for i in range(user)])

print('Ma tran ban dau:\n ', A)
print('ma tran W:\n ', W)
print('ma tran H:\n', K)
print('ma tran A3:\n', C)
