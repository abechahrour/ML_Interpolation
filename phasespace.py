#import phasespace
#from phasespace import GenParticle
#from phasespace import kinematics as kin
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp
#import samplers as sp
tfd = tfp.distributions

pid = ['p_0', 'p_1']
mp1 = 0.
mp2 = 0.
mk1 = 0.105
mk2 = 0.105
pi = 3.14159265
e_sq = 4*pi/137
sw = 0.472
cw = 0.882
Iw3l = -0.5
Iw3r =  0.5
MZ = 91.19 #GeV
MW = 80.4
Mh = 125
GZ = 2.5
GW =  2.1
Qf = -1
gpf = - sw/cw * Qf
gnf = -sw/cw * Qf + Iw3l/(sw*cw)
gpfb = -sw/cw
gnfb = -sw/cw + Iw3r/(sw*cw)
m1 = m2 = m3 = m4 = 0.1
def bracket(A, B, C):
    return (A**2 + B**2 + C**2 - 2*(A*B + B*C + A*C)) / (4*B)

def Mpmpm(m12_sq, m13_sq, m24_sq, m34_sq):
    pre = e_sq**3 * (2* gpf*gpf/(cw**2 * sw))**2
    res = (m13_sq* m24_sq * np.sqrt(MW**4 + (MW*GW)**2))/ \
          ((m12_sq**2 + (GZ**2 - 2*m12_sq)*MZ**2 +MZ**4)* \
          (m34_sq**2 + (GZ**2 - 2*m34_sq)*MZ**2 +MZ**4))
    return pre * res

def Mpmmp(m12_sq, m14_sq, m23_sq, m34_sq):
    pre = e_sq**3 * (2* gpf*gnf/(cw**2 * sw))**2
    res = (m14_sq* m23_sq * np.sqrt(MW**4 + (MW*GW)**2))/ \
          ((m12_sq**2 + (GZ**2 - 2*m12_sq)*MZ**2 +MZ**4)* \
          (m34_sq**2 + (GZ**2 - 2*m34_sq)*MZ**2 +MZ**4))
    return pre * res

def Mmppm(m12_sq, m23_sq, m14_sq, m34_sq):
    pre = e_sq**3 * (2* gnf*gpf/(cw**2 * sw))**2
    res = (m23_sq* m14_sq * np.sqrt(MW**4 + (MW*GW)**2))/ \
          ((m12_sq**2 + (GZ**2 - 2*m12_sq)*MZ**2 +MZ**4)* \
          (m34_sq**2 + (GZ**2 - 2*m34_sq)*MZ**2 +MZ**4))
    return pre * res

def Mmpmp(m12_sq, m24_sq, m13_sq, m34_sq):
    pre = e_sq**3 * (2* gnf*gnf/(cw**2 * sw))**2
    res = (m13_sq* m24_sq * np.sqrt(MW**4 + (MW*GW)**2))/ \
          ((m12_sq**2 + (GZ**2 - 2*m12_sq)*MZ**2 +MZ**4)* \
          (m34_sq**2 + (GZ**2 - 2*m34_sq)*MZ**2 +MZ**4))
    return pre * res

def PS_density(x1, x2):
    return pi * Mh**4 *x1**3 *x2 * (1-x1**2)*(1-x2**2)
    #return pi *x1**3 *x2 * (1-x1**2)*(1-x2**2)

def M_sq(m12_sq, m13_sq, m14_sq, m23_sq, m24_sq, m34_sq):
    return Mpmpm(m12_sq, m13_sq, m24_sq, m34_sq) + \
           Mpmmp(m12_sq, m14_sq, m23_sq, m34_sq) + \
           Mmppm(m12_sq, m23_sq, m14_sq, m34_sq) + \
           Mmpmp(m12_sq, m24_sq, m13_sq, m34_sq)


def x_to_var(x):
    m234 = Mh* x[:,0]
    m34 = m234 * x[:,1]
    c12 = 2*x[:,2] - 1
    c23 = 2*x[:,3] - 1
    phi = 2*pi*x[:,4]
    return m234, m34, c12, c23, phi
def x_to_var_arr(x):
    m234 = Mh* x[:,0]
    m34 = m234 * x[:,1]
    c12 = 2*x[:,2] - 1
    c23 = 2*x[:,3] - 1
    phi = 2*pi*x[:,4]
    return np.column_stack((m234/Mh, m34/Mh, c12, c23, np.cos(phi)))

def get_masses(m234, m34, c12, c23, phi):

    ## (234) FRAME
    E1 = np.sqrt(bracket(0, m234**2, Mh**2))
    E2 = np.sqrt(bracket(0, m234**2, m34**2))
    m12_sq = 2*E1*E2*(1 - c12)
    m134_sq = Mh**2 - 2*E1*E2*(1 - c12)

    ## (34) FRAME
    E3 = E4 = np.sqrt(bracket(0, m34**2, 0))
    E2 = np.sqrt(bracket(0, m34**2, m234**2))
    E1 = np.sqrt(bracket(0, m34**2, m134_sq))
    c12_34 = -(m12_sq/2 - E1*E2)/(E1*E2)
    #c12_34 = m12_sq/(2*E1*E2)
    m23_sq = 2*E2*E3*(1-c23)
    m24_sq = 2*E2*E3*(1+c23)

    c13 = c12_34 * c23 + np.sqrt(1-c12_34**2)*np.sqrt(1-c23**2)*np.cos(phi)
    m13_sq = 2*E1*E3*(1-c13)
    m14_sq = 2*E1*E4*(1+c13)

    return m12_sq, m13_sq, m14_sq, m23_sq, m24_sq, m34**2

def kallen_func(a,b,c):
    return (a-b-c)**2 - 4*b*c
def two_particle_const(root_s, mk1 = mk1, mk2 = mk2):
    s = root_s**2
    kallen = np.sqrt(kallen_func(s, mk1, mk2))
    ps_const = 1/(8*s * (2*np.pi)**2) * kallen
    flux = 1/(2*s)
    const = flux * ps_const
    return const
def get_ps_points(root_s, M1, M2, size):
    E1 = (root_s**2+M1**2+M2**2)/(2*root_s)
    E2 = root_s - E1
    pz = np.sqrt(E1**2 - M1**2)
    P1 = kin.lorentz_vector([0, 0, pz], [E1])
    P2 = kin.lorentz_vector([0, 0, -pz], [E2])
    P1 = np.ones([size,1])*P1
    P2 = np.ones([size,1])*P2
    return P1, P2
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    norm = np.norm(vector, axis=-1, keepdims=True)
    normed_vec = vector / norm
    return normed_vec

def get_phi(v1):
    x = kin.x_component(v1)
    y = kin.y_component(v1)
    phi = np.arctan2(y, x)
    return phi
def cos_theta(v1):
    three_vec = kin.spatial_component(v1)
    v1_u = unit_vector(three_vec)
    return kin.z_component(v1_u)

def tens_to_numpy(a):
    x = tf.make_tensor_proto(a)
    x = tf.make_ndarray(x)
    return x

def getphasespace(root_s, mk1, mk2, n_events):
    weights, particles = phasespace.nbody_decay(root_s,
                                            [mk1, mk2]).generate(n_events=n_events)
    return weights, particles


def getfourmom(x):
    p1 = x.get(pid[0])
    p2 = x.get(pid[1])
    return p1, p2

def get2dphasespace(root_s, mk1, mk2, n_events):
    weights, particles = getphasespace(root_s, mk1, mk2, n_events)
    p1, p2 = getfourmom(particles)
    angles = cos_theta(p1)
    phi = get_phi(p1)
    x1 = (angles[:,0] + 1.)/2.
    x2 = (phi[:,0] + np.pi)/(2*np.pi)
    x = np.vstack([x1, x2]).T
    return x

def x_to_angle(x):
    x1 = x[:,0]
    x2 = x[:,1]
    costheta = 2*x1-1
    phi = 2*np.pi*x2
    angles = np.stack([costheta, phi], axis=1)
    return angles
def angle_to_fourmom(root_s, mp1, mp2, mk1, mk2, angles):
    costheta = angles[:,0]
    #print("this is costheta", costheta)
    phi = angles[:,1]
    size = np.size(costheta)
    #print("this is size", size)
    sintheta = np.sin(np.acos(costheta))
    #print("this is sintheta", sintheta)
    weights, particles = getphasespace(root_s, mk1, mk2, 1)
    k1, k2 = getfourmom(particles)
    #k1, k2 = np.cast(k1, dtype=np.float32), np.cast(k2, dtype=np.float32)
    #print("this is k1", k1)
    E1 = np.tile(kin.time_component(k1),(size,1))
    #print("this is E1 after tile", E1)
    E2 = np.tile(kin.time_component(k2), (size,1))
    E1 = np.reshape(E1, [-1])
    E2 = np.reshape(E2, [-1])
    k1_mag = np.norm(kin.spatial_component(k1))
    #print("this is k1_mag", k1_mag)
    E1, E2, k1_mag = np.cast(E1, dtype = np.float32), np.cast(E2, dtype = np.float32), np.cast(k1_mag, dtype = np.float32)
    #print("These are E1,E2", E1, E2)
    k1x = k1_mag*sintheta*np.cos(phi)
    k1y = k1_mag*sintheta*np.sin(phi)
    k1z = k1_mag*costheta
    #print("this is k1x", k1x)
    k1 = np.stack([k1x, k1y, k1z, E1], axis = 1)
    k1 = k1*np.ones_like(k1)
    k2 = np.stack([-k1x, -k1y, -k1z, E2], axis = 1)
    P1, P2 = get_ps_points(root_s, mp1, mp2, size)
    ps_point = np.stack([P1, P2, k1, k2], axis = 1)
    #print("this is P1", P1)
    #print("this is P2", P2)
    #print("this is k1", k1)
    #print("this is k2", k2)
    return  ps_point

#Create the training data using numpy. Uniformly distributed points in n-dim space.
def create_data_uniform(ndims, n_events):
    x_train = np.random.uniform(0,1, (n_events, ndims))
    y_train = np.zeros((n_events, ndims))
    return (x_train, y_train)
def create_data(ndims, n_events):
    x_train = sp.sample(ndims, n_events)
    #x_train = np.random.uniform(-1,1, (n_events, ndims))
    y_train = np.zeros((n_events, ndims))
    return (x_train, y_train)

#plt.hist(angles[:,0], 30, density=False, histtype='step')
#plt.hist(zcomp[:,0],30, density=False, histtype='step')
#plt.show()
##print(zcomp)
