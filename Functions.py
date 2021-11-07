

import vegas
import numpy as np
import scipy
from scipy.special import erf
import phasespace as ps
class Functions:


    def __init__(self, ndims, alpha, **kwargs):
        self.ndims = ndims
        self.alpha = alpha
        self.variables = kwargs
        self.calls = 0

    def gauss(self, x):

        pre = 1.0/(self.alpha * np.sqrt(np.pi))**self.ndims
        exponent = -1.0*np.sum(((x-0.5)**2)/self.alpha**2, axis=-1)
        #self.calls += 1
        return pre * np.exp(exponent)
    def gauss_interp(self, x):

        pre = 1.0/(self.alpha * np.sqrt(np.pi))**self.ndims
        exponent = -1.0*np.sum(((x-0.5)**2)/self.alpha**2, axis=0)
        #self.calls += 1
        return pre * np.exp(exponent)

    def camel(self, x):
        pre = 1.0/(self.alpha * np.sqrt(np.pi))**self.ndims
        exponent1 = -1.0*np.sum(((x-1/3)**2)/self.alpha**2, axis=-1)
        exponent2 = -1.0*np.sum(((x-2/3)**2)/self.alpha**2, axis=-1)
        #self.calls += 1
        return 0.5 * pre * (np.exp(exponent1) + np.exp(exponent2))

    def camel_interp(self, x):
        pre = 1.0/(self.alpha * np.sqrt(np.pi))**self.ndims
        exponent1 = -1.0*np.sum(((x-1/3)**2)/self.alpha**2, axis=0)
        exponent2 = -1.0*np.sum(((x-2/3)**2)/self.alpha**2, axis=0)
        #self.calls += 1
        return 0.5 * pre * (np.exp(exponent1) + np.exp(exponent2))

    @vegas.batchintegrand
    def poly(self, x):
        #print("Shape of poly = ", np.shape(x))
        # res = 0
        # for d in range(np.shape(x)[1]):
        #     res += -x[:,d]**2+x[:,d]
        # return res
        return np.sum(-x**2 + x, axis=-1)
    def poly_interp(self, x):
        return np.sum(-x**2 + x, axis=0)

    def periodic(self, x):
        return np.mean(x, axis = -1)*np.prod(np.sin(2*np.pi*x), axis = -1)
    def periodic_interp(self, x):
        return np.mean(x, axis = 0)*np.prod(np.sin(2*np.pi*x), axis = 0)

    def integral_gauss(self, dim, alpha = 0.2):
        return erf(1/(2*alpha))**dim

    def integral_camel(self, dim, alpha = 0.2):
        return (0.5*(erf(1/(3*alpha))+erf(2/(3*alpha))))**dim

    def integral_periodic(self, dim, alpha = 0.1):
        return 0

    def integral_poly(self, dim, alpha = 0.1):
        return dim/6

    def higgs_to_lep(self, x):
        m234, m34, c12, c23, phi = ps.x_to_var(x)
        m12_sq, m13_sq, m14_sq, m23_sq, m24_sq, m34_sq = ps.get_masses(m234, m34, c12, c23, phi)
        M_sq = ps.M_sq(m12_sq, m13_sq, m14_sq, m23_sq, m24_sq, m34_sq)
        return M_sq


    def D0(self, x):
        path = '/home/chahrour/Loops/D0000stmmmm/D0000stmmmm_data/D00001rssss_labels_5M.csv'
        return np.array(pd.read_csv(path, delimiter=',', nrows = x.shape[0]))
