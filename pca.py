import numpy as np
import pandas as pd

# input data size = (2,224)

class myPCA:
    def __init__(self, data, n_PC, type='similar'):
        self.data = data
        self.n_PC = n_PC
        self.type = type

    def zero_mean(self):
        data = self.data.T
        mean = np.mean(data, axis=1)
        zero_mean_data = []

        for i in range(data.shape[0]):
            zero_mean_row = data[i] - mean[i]
            zero_mean_data.append(zero_mean_row)

        zero_mean_data = np.array(zero_mean_data).reshape(data.shape[0], data.shape[1]).T

        return zero_mean_data


    def get_matrix(self):
        zero_mean_data = self.zero_mean()

        cov_matrix = np.cov(zero_mean_data.T[0],zero_mean_data.T[1])


        if self.type == 'homo':
            return cov_matrix

        elif self.type == 'heter':
            cor_matrix = np.corrcoef(zero_mean_data.T[0],zero_mean_data.T[1])
            return cor_matrix

        else:
            print('please choose "homo" for the same scale data, or "heter" for the different scale data.')


    def get_eigenfactors(self):
        matrix = self.get_matrix()
        eigval, eigvect = np.linalg.eig(matrix)

        eigen_dic = {eigval[i]:eigvect.T[i] for i in range(len(eigval))}
        eigen_dic = sorted(eigen_dic.items(), key=lambda item:item[0], reverse=True)

        eigen_value = np.array([eigen_dic[i][0] for i in range(len(eigen_dic))])
        eigen_vector = np.array([eigen_dic[i][1] for i in range(len(eigen_dic))])

        return eigen_value, eigen_vector


    def get_variance_ratio(self):
        eigen_value, _ = self.get_eigenfactors()
        ratio = np.sum(eigen_value[:self.n_PC])/np.sum(eigen_value)
        print('eigenvalues refer to {}'.format(eigen_value))
        print('your select {} principle component(PC) represent {}% variance.'.format(self.n_PC, np.round(ratio, 5)*100))

    def get_PC_loading(self):
        _, eigen_vector = self.get_eigenfactors()

        return eigen_vector

    def get_PC(self):
        zero_meam_data = self.zero_mean()
        _, eigen_vector = self.get_eigenfactors()
        PC = np.dot(eigen_vector[:self.n_PC], zero_meam_data.T).T
        PC_of_n_dim = PC

        return PC_of_n_dim


x=np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y=np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])
data=np.array([[x[i],y[i]] for i in range(len(x))])

pca = myPCA(data, 1, 'homo')
pca.get_variance_ratio()
PC = pca.get_PC()
print(PC)

