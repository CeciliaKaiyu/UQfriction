# This script define Kernel mean and initial error for Tsunami example
# sklearn-rbf kernel takes the form k(x,y)=exp(-(x-y)**2/(2*length-scale**2))
# thus: the hyperparameter "hyper" in Gauss kernel we define equals 2 * sklearn-rbf-lengthscale**2
# "hyper"  need consist with  "hyper"  in kernel function


import torch
import math as math
import numpy as np

# Gauss kernel Kernel mean and beta(2,5)
def KM_Gauss(X,hyper):
    '''

    :param X:  \omega_1 \omega_2 \omega_3
    :param hyper: length-scale**2
    :return: kernel mean
    '''

    x = X
    l = hyper
    term1 = (-2 * torch.exp(-((-1 + x)**2 / l)) * math.sqrt(l) * (4 * l**2 + 2 * (-1 + x)**3 * x + l * (4 - 13 * x + 9 * x**2)))
    term2 = (2 * torch.exp(-(x**2 / l)) * math.sqrt(l) * (4 * l**2 + 2 * (-1 + x)**4 + l * (12 + x * (-20 + 9 * x))))
    term3 = math.sqrt(math.pi) * (4 * (-1 + x)**4 * x + 3 * l**2 * (-4 + 5 * x) + 4 * l * (-1 + x)**2 * (-2 + 5 * x)) * torch.erf((1 - x) / math.sqrt(l))
    term4 = math.sqrt(math.pi) * (4 * (-1 + x)**4 * x + 3 * l**2 * (-4 + 5 * x) + 4 * l * (-1 + x)**2 * (-2 + 5 * x)) * torch.erf(x / math.sqrt(l))

    result = (15/4 * math.sqrt(l)) * (term1 + term2 + term3 + term4)

    return result



#Gauss kernel Initial Error
def IE_Gauss(hyper):
    '''

    :param hyper: length-scale**2
    :return: initial error
    '''
    l = hyper
    term1 = (5/77) * torch.exp(-1/l) * l * (28 + 3 * torch.exp(1/l) * l * (385 + 4 * l * (77 - 44 * l**2 + 50 * l**3)) - 2 * l * (117 + 2 * l * (190 + 3 * l * (-19 + 6 * l + 50 * l**2))))
    term2 = (5/77) * math.sqrt(l) * (-28 + 11 * l * (20 + 81 * l)) * math.sqrt(math.pi) * torch.erf(1 / torch.tensor(math.sqrt(l)))

    result = term1 - term2

    return result



def KM_Gauss33(X, hyper):
    x = X
    l = hyper**(1/2)

    term1 = torch.exp(-(1 + x ** 2) / l ** 2)
    term2a = 2 * torch.exp(1 / l ** 2) * l * (2 * (-1 + x) ** 2 * x + (-4 + 5 * x) * l ** 2)
    term2b = -2 * torch.exp((2 * x) / l ** 2) * l * (2 * (-1 + x) * x ** 2 + (-1 + 5 * x) * l ** 2)
    term2 = term1 * (term2a + term2b)

    term3 = math.sqrt(math.pi) * (4 * (-1 + x) ** 2 * x ** 2 + 2 * (1 + 6 * (-1 + x) * x) * l ** 2 + 3 * l ** 4)

    result = (15 / 4) * l * (term2 + term3 * (torch.erf((1 - x) / l) + torch.erf(x / l)))

    return result



def IE_Gauss33(hyper):
    l = hyper**(1/2)
    term1 = -2 * l * torch.exp(-1 / l**2)
    term2 = -2 + 7 * l**2 + 12 * (-3 + 7 * torch.exp(1 / l**2)) * l**4 + (24 - 72 * torch.exp(1 / l**2)) * l**6 + 48 * (-1 + torch.exp(1 / l**2)) * l**8
    term3 =  math.sqrt(math.pi) * (4 - 12 * l**2 + 63 * l**4) * torch.erf(1 / l)
    return (5/14) * l *(term1 * term2 + term3)











