import numpy as np
import matplotlib.pyplot as plt
class Lasso:
    def __init__(self, parallel = False, T = None):
        """
        self.x: X matrix, m by n
        self.y: Y matrix, m by 1
        self.beta: beta, model parameters
        self.lam: lambda, regularization param
        self.diag_Xmatrix: [x1.T.dot(x1), x2.T.dot(x2),...] xi : ith column of the X matrix
        self.step_size
        T:  the numbers of the parameters of a set to update in parallel
        k:  the numbers of the sets of the parameters
        """
        with open('./input/y.mmt') as y_mat:
            tmp = []
            for line in y_mat.readlines()[2:]:
                line = line.strip()
                fields = line.split(" ")
                if len(fields) > 3:
                    fields.pop(1)
                tmp.append(fields[2])
                self.y = np.array(tmp, dtype='float32')

        with open('./input/x.mmt') as x_mat:
            lines = x_mat.readlines()
            l = lines[1].strip()
            field = l.split(" ")
            if len(field) > 3:
                field.pop(1)
            self.x = np.zeros([int(field[0]),int(field[1])], dtype='float32')
            for line in lines[2:]:
                line = line.strip()
                fields = line.split(" ")
                self.x[int(fields[0])-1][int(fields[1])-1] = (fields[2])
        if parallel == True:
            assert T != None, "T must be set"
            self.T = T
            self.k = self.x.shape[1] // T
        self.beta = np.zeros(self.x.shape[1], dtype='float32')
        self.delta = np.zeros_like(self.T, dtype='float32')
        self.lam = 1e-4
        self.step_size = 1
        self.diag_Xmatrix = np.diag(self.x.T.dot(self.x))
        self.loss = 0.5 * np.sum((self.y - self.x.dot(self.beta))**2) + self.lam * np.sum(np.abs(self.beta))
        self.loss_1 = 5e20
        self.Loss = [self.loss]

    def _train(self):
        for i in range(self.k - 1):
            # self.delta = self.x.T[i*self.T:i*self.T+self.T].dot(self.y - self.x.dot(self.beta)) + \
            #              self.diag_Xmatrix[i*self.T:i*self.T+self.T]*self.beta[i*self.T:i*self.T+self.T]
            self.delta = self.x.T[i*self.T:i*self.T+self.T].dot(self.y - self.x.dot(self.beta))
            self.delta = self.S_function(self.delta)
            self.beta[i*self.T:i*self.T+self.T] += self.step_size * self.delta
        # self.delta = self.x.T[(self.k-1)*self.T:self.x.shape[1]].dot(self.y - self.x.dot(self.beta)) + \
        #              self.diag_Xmatrix[(self.k-1)*self.T:self.x.shape[1]]*self.beta[(self.k-1)*self.T:self.x.shape[1]]
        self.delta = self.x.T[(self.k - 1) * self.T:self.x.shape[1]].dot(self.y - self.x.dot(self.beta))
        self.delta = self.S_function(self.delta)
        self.beta[(self.k-1)*self.T:self.x.shape[1]] += self.step_size * self.delta
        self.loss = 0.5 * np.sum((self.y - self.x.dot(self.beta))**2) + self.lam * np.sum(np.abs(self.beta))

    def train(self, print_every = 100,decay_every = 1e4):
        n = 1
        Flag = False
        while not Flag:
            self.loss_1 = self.loss
            n += 1
            self._train()
            self.Loss.append(self.loss)
            if n % print_every == 0:
                print(n, self.loss)
            if n % decay_every == 0:
                self.step_size *= 0.99
            Flag = np.abs(self.loss - self.loss_1) < 1e-6
        plt.plot(self.Loss)
        plt.title('convergence curve graph')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()

    def S_function(self,grad_hat):
        tmpflag = np.abs(grad_hat) > self.lam
        betaGreaterFlag = grad_hat > 0
        betaLessFlag = grad_hat < 0
        grad_hat = grad_hat + self.lam * tmpflag * betaLessFlag - self.lam * tmpflag * betaGreaterFlag
        grad_hat = grad_hat * tmpflag
        return grad_hat


if __name__ == "__main__":
    L = Lasso(parallel=True, T=4) # T = 4, 8, 16
    L.train()
