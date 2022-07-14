import scipy.interpolate
import numpy as np
from random import random as rd

m = np.array([12, 9.3, 6.2, 5.5, 2.8, 0.2])
e_stat = np.array([12.84, 11.19, 9.14, 8.63, 6.75, 4.814])
e_inf = np.array([7.42, 7.2, 5.93, 6.023, 5.503, 4.507])
t = np.array([0.611, 0.73, 0.8, 1, 2.28, 0.82])*10**-9
sigma = np.array([20.6, 23, 6.7, 5.15, 2.03, 0.606])*10**-3

es = scipy.interpolate.interp1d(m, e_stat)
ei = scipy.interpolate.interp1d(m, e_inf)
t0 = scipy.interpolate.interp1d(m, t)
si = scipy.interpolate.interp1d(m, sigma)

mv = 0.2 + 11.8*rd()

print("#material: {} {} 1 0 Concrete \n".format(ei(mv), si(mv)))
print("#add_dispersion_debye: 1 {} {} Concrete \n".format(es(mv)-ei(mv), t0(mv)))
print("#box: 0 0 0 0.5 0.001 0.3 Concrete \n")

r = 0.005 + 0.0245*rd()
z = (0.3 - r)*rd()
print("#cylinder: 0.25 0 {} 0.25 0.001 {} {} pec \n".format(z, z, r))

filename = "Concrete_labels"
f = open(filename, 'a')
f.write("{}: {} {} {}".format(current_model_run, r, z, mv))

from user_libs.antennas import GSSI
GSSI.antenna_like_GSSI_1500(0.25, 0.1, 0.3, resolution=0.001)