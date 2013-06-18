'''
Created on Jun 18, 2013

@author: delforge
'''
import sw
import numpy as np

#-----------------

space0 = sw.networks.Gain1(.5)
space1 = sw.networks.Gain1(.2)
networks = [space0, space1]
couplings = [set([1, 2])]
n = 4

presolver = sw.Solver(n, couplings) 
solver = presolver(networks)
ao = np.array([1, 0], dtype=complex)
b = solver(ao)
print np.abs(b[3]), np.angle(b[3], True)

b = space0.dot(np.array([1, 0], dtype=complex))
b = space1.dot(np.array([b[1], 0], dtype=complex))
print np.abs(b[1]), np.angle(b[1], True)


#---------------

frequencies = np.linspace(1000e9, 1004e9, 1001)
y = []
interface = sw.networks.SemiTransparentMirror1(.1, .9)
couplings = [{1, 2}, {3, 4}]
ao = np.array([1, 0], dtype=complex)
n = 6
presolver = sw.Solver(n, couplings)
for f in frequencies:
    distance = sw.networks.Distance1(1, 1, f)
    networks = [interface, distance, interface]
    solver = presolver(networks)
    b = solver(ao)
    y.append(np.abs(b[5]) ** 2)
print y
import matplotlib.pyplot as plt
plt.plot(frequencies / 1.e9, y)
plt.show()
