import xalglib as alg
import numpy as np
import matplotlib.pyplot as plt
import time
def f(x):
    return np.sum(np.sin(2*np.pi*x), axis = 1)


s = alg.rbfcreate(9, 1)
x = np.random.rand(100000, 9)
y = f(x)
y = y.reshape(-1, 1)
z = np.concatenate((x, y), axis = 1)
alg.rbfsetpoints(s, z.tolist())
print("Points set", flush = True)
alg.rbfsetalgohierarchical(s, 0.3, 5, 0.0)
print("Hierarchy set", flush = True)
rep = alg.rbfbuildmodel(s)
print("Model built", flush = True)
print(rep.terminationtype, flush = True) # expected 1
x_test = np.random.rand(100000, 9)
y_test = f(x_test)
v = []

start = time.time()

for i in x_test:
    v.append(alg.rbfcalc(s, i.tolist()))

end = time.time()

print("Time for prediction = ", end - start)
v = np.array(v).squeeze()
relerr = (y_test - v)/y_test * 100


plt.figure()
plt.hist(relerr, range = (-200, 200), histtype = 'step')
plt.savefig("relerr.png")
