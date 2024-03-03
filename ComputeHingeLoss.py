



import numpy as np




import numpy as np

def loss(z):
    if (z >= 1):
        return 0
    else:
        return 1 - z



#Squared erorrs loss
def loss2(z):
    return (z**2)/2

def empiricalRisk(features, labels, thetas, lossfunc):
    N = features.shape[1]
    mySumLoss = 0
    for n in range(N):
        mySumLoss += lossfunc(labels[n] - np.dot(thetas, features[:,n]))
        print("mysumloss", mySumLoss, N)
    averageEmpiricalRisk = mySumLoss / N





fo = [ [24,12,  6],
        [0, 0, 0],
        [12,    6  ,  3],
        [24 ,  12    ,6]]


    return averageEmpiricalRisk

features = np.array([[1, 0, 1],
                     [1, 1, 1],
                     [1, 1, -1],
                     [-1, 1, 1]])
labels = np.array([2, 2.7, -0.7, 2])
thetas = np.array([[0], [1], [2]])
risk = empiricalRisk(features.transpose(), labels, thetas.transpose(), loss)
risk2 = empiricalRisk(features.transpose(), labels, thetas.transpose(), loss2)


print(risk, risk2, "rsk")
