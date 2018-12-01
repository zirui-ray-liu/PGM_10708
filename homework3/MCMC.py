import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt

def cal_log_posterior_prob(x_samples, mu1, mu2):
    Guass_prior = np.exp(-(mu1**2+mu2**2)/200)
    tmp = 0.5*(np.exp(-0.5*(x_samples - mu1)**2) + np.exp(-0.5*(x_samples - mu2)**2))
    tmp = np.log(tmp)
    tmp = np.sum(tmp)
    return tmp + np.log(Guass_prior)
def cal_transition_prob(mu1,mu2,mu1_hat,mu2_hat ,sigma):
    return np.exp(-((mu1_hat-mu1)**2 + (mu2_hat-mu2)**2)/2/sigma**2)/2/np.pi/sigma**2

def cal_accept_rate(x_samples,mu1,mu2,mu1_hat,mu2_hat, sigma):
    alpha = cal_log_posterior_prob(x_samples,mu1_hat,mu2_hat) - cal_log_posterior_prob(x_samples,mu1,mu2)
    alpha = np.exp(alpha)
    alpha *= cal_transition_prob(mu1_hat,mu2_hat,mu1,mu2, sigma) / cal_transition_prob(mu1,mu2,mu1_hat,mu2_hat,sigma)
    assert alpha >= 0
    return alpha

def Metropolis_Hastings_sampling(x_samples):
    mu1, mu2 = 0.0, 0.0
    sigma = 0.5
    N = 11000
    n = 0
    accept_counts = 0
    MU1, MU2 = [], []
    while n < N:
        n += 1
        mu1_hat = np.random.normal(mu1, sigma)
        mu2_hat = np.random.normal(mu2, sigma)
        alpha = cal_accept_rate(x_samples,mu1,mu2,mu1_hat,mu2_hat,sigma)
        if alpha >= 1:
            mu1, mu2 = mu1_hat, mu2_hat
            accept_counts += 1
        else:
            tmp = np.random.uniform(0, 1)
            if tmp <= alpha:
                mu1, mu2 = mu1_hat, mu2_hat
                accept_counts += 1
        if n >= N - 1000:
            MU1.append(mu1)
            MU2.append(mu2)
    return MU1, MU2, accept_counts/N

def Gibbs_sampling(x_samples):
    mu1, mu2 = 0.0, 0.0
    N = 11000
    n = 0
    MU1, MU2 = [], []
    z = np.zeros_like(x_samples, dtype=np.int8)
    while n < N:
        n += 1
        for i in range(len(z)):
            prob = np.exp(-0.5 * (x_samples[i] - mu1) ** 2) / (np.exp(-0.5*(x_samples[i]-mu2)**2) + np.exp(-0.5*(x_samples[i]-mu1)**2))
            u = np.random.uniform(0, 1)
            if u < prob:
                z[i] = 1  # Represents the data generate from mu1
            else:
                z[i] = 0  # Represents the data generate from mu2
        m_1 = np.sum((z == 1))
        m_0 = np.sum((z == 0))
        mu1_hat = 100 * np.sum((z == 1)*x_samples)/(100*m_1+1)
        mu1 = np.random.normal(mu1_hat, np.sqrt(200/(100*m_1+1)))
        mu2_hat = 100 * np.sum((z == 0)*x_samples)/(100*m_0+1)
        mu2 = np.random.normal(mu2_hat, np.sqrt(200/(100*m_0+1)))
        if n >= N - 1000:
            MU1.append(mu1)
            MU2.append(mu2)
    return MU1, MU2

if __name__ == "__main__":
    sampleNo = 100
    np.random.seed(0)
    x_samples = np.zeros(sampleNo, dtype=np.float32)
    for i in range(sampleNo):
        u = np.random.uniform(0, 1)
        if u >= 0.5:
            x_samples[i] = np.random.normal(5.0, 1)
        else:
            x_samples[i] = np.random.normal(-5.0, 1)

    MU1, MU2, accept_rate = Metropolis_Hastings_sampling(x_samples)
    print(np.mean(MU1), np.mean(MU2))
    print(accept_rate)
    MU1_G, MU2_G = Gibbs_sampling(x_samples)
    print(np.mean(MU1_G),np.mean(MU2_G))
    plt.subplot(211)
    plt.plot(MU1, MU2, '+')
    plt.subplot(212)
    plt.plot(MU1_G, MU2_G, '+')
    plt.show()
