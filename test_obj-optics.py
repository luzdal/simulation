from objects_optics import OPT, beam_atoms, stats, thermodyn, beam_laser
#from objects_optics import OPT, oven, beam_atoms, stats, thermodyn, beam_laser, MOT
# % %
import numpy as np
import math
import matplotlib.pyplot as plt

pi = np.pi

T_arr1 = np.arange(450, 1350, 50)
T_arr2 = np.arange(1000, 1250, 50)
T_arr3 = np.arange(550, 1450, 100)
T_arr4 = np.arange(800, 1300, 100)

atoms_test  = beam_atoms(iso=164, T_hl=T_arr1, res='r')
laser_test  = beam_laser(diam=10e-2, powr=5e-3, detun=0, **{'trans':atoms_test})
opt_test    = OPT(light=laser_test, matter=atoms_test, **{'v':np.arange(-7, 7, 1e-1*7/2)})


thermo_test = thermodyn(matter=atoms_test)
thermo_test.get_press(T=[1200])
