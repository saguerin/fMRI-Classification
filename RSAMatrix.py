import math
import numpy as np

#input:
#data: m x n numpy matrix, m is the number of data points, n is the number of features/voxels
#label m x 1 numpy matrix, consisting of the class label for each data point, labels should be in range of [0, n)
#zTransfer: whether to transfer correlation coefficient to z value using Fisher transform
def CalcRSAMatrix(data, label=None, zTransfer=False):
    dim = data.shape

    if label is None:
        classNum = dim[0]
        ret = np.corrcoef(data)
    else:
        classNum = label.max() + 1
        data1 = np.zeros((classNum, dim[1]))

        for i in range(dim[0]):
            data1[int(label[i]), :] += data[i, :]

        ret = np.corrcoef(data1)

    if zTransfer:
        for i in range(int(classNum)):
            ret[i, i] = 0

        oneM = np.ones(ret.shape)
        ret = np.divide((oneM + ret), (oneM - ret))
        ret = np.multiply(np.log(ret), 0.5 * math.sqrt(dim[1] - 3))

    return ret


