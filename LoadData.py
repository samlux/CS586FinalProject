import pickle as pkl

path = 'Data/'
filename = 'NJ_cases'
fileObject = open(path + filename, 'rb')
NJ_cases = pkl.load(fileObject)
fileObject.close()
filename = 'NJ_pop'
fileObject = open(path + filename, 'rb')
NJ_pop = pkl.load(fileObject)
fileObject.close()
filename = 'NJ_dense'
fileObject = open(path + filename, 'rb')
NJ_dense = pkl.load(fileObject)
fileObject.close()
