import numpy as np


def padding(data):

  indices = np.where(data == '?')

  for i in range(len(indices[0])):
    data[indices[0][i]][indices[1][i]] = -1

  return data

