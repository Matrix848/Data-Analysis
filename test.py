import csv
import math as mh
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.interpolate import make_interp_spline
from scipy import integrate
import pandas as pd


def funT(x):
    y = 2*mh.pi*((x/9.81)**(1/2))
    return y
  
def fun(x):
  y = x**2
  return y


df = pd.read_csv('/home/marco/Documents/Projects/data-analysis/data/pendulum/data.csv')

print(df)


#x = np.linspace(0, lenghts[len(lenghts)-1], 1000)
#plt.plot(np.log(lenghts), np.log(T))
l = np.array(df['l'].to_list())/1000

x = np.linspace(0, sorted(l)[len(l)-1], 10000)

plt.plot(np.log(l), np.log(np.array(df['P'].to_list())/10), 'o')

print(funT(0.10))
plt.plot(np.log(x), np.log(funT(x)))

plt.show()