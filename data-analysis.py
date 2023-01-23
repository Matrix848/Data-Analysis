import csv
import math as mh
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.interpolate import make_interp_spline
from scipy import integrate


# Read the data from a csv file and save them in a sorted list, which is then returned as the result of the function
def dataReader(path: str) -> list:
    data = []
    
    f = open(path, 'r')
    csvreader = csv.reader(f)
    
    for row in csvreader:
        data.append(float(row[0]))
    return sorted(data, key = float)

# Calculate the mean (len() returns the lenght of the given object)
def mean(data: list) -> float:

    mean_val = sum(data)/len(data) 
    
    return mean_val  


# Calculate the standard deviation
def stDev(data: list) -> float:
    sum_val = 0.0
    c = 0
    mean_val = mean(data)

    for val in data:
        sum_val += pow(val - mean_val, 2)
        c += 1
    
    stdev = mh.sqrt(sum_val/c)
    
    return stdev

    
# Count the occurences of each value in the data list and save it in a list of the type [[value, n of occurences]]
def ocsCounter(data: list) ->list:
    ocs = []
    prev_val = None
    
    for value in data:
        if prev_val != value or None:
            ocs.append([value, data.count(value)])
            prev_val=value
            
    return ocs

# Calculate the number of bins that must be used to create the histogram, if the size of the bins is not specified it will use half of the standard deviation
def binsCounter(data: list, size: float = None) -> int:
    stdev = stDev(data)
    mean_val = mean(data)
    
    if size == None:
        size = stdev/2

    bins = round((mean_val-data[0])/(size)) + round((data[len(data)-1]-mean_val)/(size)) +1
    
    return bins


# Calculate the data distribution in the varius bins(without normalizing it)
def distribution(data: list, size: float = None) -> list: #, bins: int = None):
    
    mean_val = mean(data)
    stdev = stDev(data)
    ocs = ocsCounter(data)
    
    if size == None:
        size = stdev/2
        
    n_bins = binsCounter(data, size)
    distOcs = []
    
    start_bin_pos = mean_val - round((mean_val-data[0])/(size))*size
    stop_bin_pos = mean_val - round((data[len(data)-1]-mean_val)/(size))*size
    
    bins_pos = []

    for i in range(n_bins):
        pos = start_bin_pos + size*i
        bins_pos.append(pos)
        i += 1
        
    distOcs.append([bins_pos[i] for i in range(n_bins)])
    distOcs.append([0 for i in range(n_bins)])
    
    for val in ocs:
        for i  in range(n_bins):
            if bins_pos[i] - size/2 < val[0] <= bins_pos[i] + size/2:
                distOcs[1][i] += val[1]
   
    return distOcs


# Calculate the normalized data distribution
def normalDistribution(data: list, size: float = None) -> list:
    
    stdev = stDev(data)
    
    if size == None: size = stdev/2
    
    distrData = distribution(data, size)
    
    for i in range(len(distrData[1])):
        distrData[1][i] = distrData[1][i]/(len(data)*size)
        
    return distrData


# DO NOT USE (it is needed only to convert the data from the distribution() or normalDistribution() functions so that matplotlib.pylot.hist() can run properly using them)
def distrMtplt(data: list) -> list:
    distrData = distribution(data)

    distrmtplt = []
    
    for i in range(len(distrData[0])):
        
        for j in range(distrData[1][i]):
            
            distrmtplt.append(distrData[0][i])
            
    return distrmtplt


# Gaussian function 
def gaussianFunction(x, stdev: float, mean_val: float()):
    
    y = (1 / (stdev * mh.sqrt(2 * mh.pi))) * np.exp(-0.5 * (((x - mean_val)/ stdev) ** 2))
    
    return y

# Probability distribution function(PDF)
def chi2PDF(x, ddof: int):
        
    y = (x**(ddof/2-1)) * np.exp(-x/2) / ((2**(ddof/2))*scipy.special.gamma(ddof/2))
    
    return y

# Plot the gaussian function(gaussianFunction)
def gaussianPlotter(data: list, center: float, color: str = None, label: str = None, ax: np.ndarray = None):
    if ax is not None:
        plt.sca(ax)

    # Create a sample of numbers to later be able to properly print the Gaussian function
    x = np.linspace(data[0], data[len(data)-1] + stDev(data)/2, 100)
  
    # Create the gaussian function defined in gaussianFunction() method using the above sample
    plt.plot(x, gaussianFunction(x, stDev(data), center), color = color, label = label)
    
    
# Plot the PDF
def chi2PdfPlotter(data: list, ddof: int, alpha: float, size: float,color: str = None, label: str = None, ax: np.ndarray = None):
    if ax is not None:
        plt.sca(ax)

    x = np.linspace(0, 3*ddof, 10000)
    
    plt.xlim(0, x.max())
    plt.ylim(0, chi2PDF(x, ddof).max()*2)
    
    cycle = True
    x_ = 0.0 + ddof
    c = 0.1
    
    while cycle:
        v =  round(-integrate.quad(chi2PDF, np.inf , x_, args=(ddof))[0], 10)
        
        if v == alpha:
            a_pos = x_
            cycle = False
        elif v < alpha:
            x_ -= c
            c = c / 10
        elif v > alpha:
            x_ += c 
        
    scipy.special.gammainc(ddof, alpha)
    chi2 = chiSquare(data, size)
    
    plt.fill_between(x, chi2PDF(x, ddof), where = (x>=a_pos), alpha=0.35, color='#37de01')
    plt.plot([a_pos, a_pos], [0, chi2PDF(a_pos, ddof)], linestyle='--', color='#37de01')
    plt.text(a_pos, chi2PDF(a_pos, ddof)/2, 'α ', ha='right', color='#37de01', fontsize =12)
    
    plt.plot([chi2, chi2], [0, chi2PDF(chi2, ddof)], linestyle='--', color='#ee0000')
    plt.text(chi2, chi2PDF(chi2, ddof)/2, 'X² ', ha='right', color='#ee0000', fontsize =12)
    plt.text((chi2 + a_pos)/2, chi2PDF((chi2 + a_pos)/2, ddof)+0.02, 'p value: ' + str(round(pValue(data, size), 3)), va='top', color='#ee0000',fontsize =12)
    plt.fill_between(x, chi2PDF(x, ddof), alpha = 0.30, where = (x>=chi2)&(x<=a_pos), color='#ff0000')
    
    plt.plot([ddof-2, ddof-2], [0, chi2PDF(ddof-2, ddof)], linestyle='--')
    plt.plot(x, chi2PDF(x, ddof), color = color, label = label)
   
 
def chiSquare(data: list, size: float = None):
    
    stdev = stDev(data)
    mean_val = mean(data)
    
    if size == None: size = stdev
    
    distr = normalDistribution(data, size)
    chisquare = 0.0
    
    for i in range(len(distr[1])):
        val = distr[1][i]
        pos = distr[0][i]
    
        exp_val = (mh.erf((pos - mean_val + size/2)/(stdev*mh.sqrt(2))) - mh.erf((pos - mean_val - size/2)/(stdev*mh.sqrt(2))))/(2*size)
        chisquare += ((val-exp_val)**2)/exp_val
    
    return chisquare

# Plot the histogram of the distribution/normal distribution
def distrPlotter(data: list, size: float = None, error_bars: bool = False, bicolor: bool = False, color_1: str = '#ffb200', color_2: str = '#e67505', err_color: str = '#900C3F', normal: bool = False, label: str = None, ax = None):
    
    stdev = stDev(data)
    
    if size == None: size = stdev/2
    
    distrData = normalDistribution(data, size = size) if normal else distribution(data, size = size)
        
                
    x_dis, y_dis = distrData

    if ax is not None: plt.sca(ax)
        
    if bicolor:
        for i in range(len(x_dis)):
            plt.bar(x_dis[i], y_dis[i], width=size, color = color_1 if i % 2 else color_2)
    else:
        plt.bar(x_dis, y_dis, width=size, color = color_1, label = label)

    
    if error_bars:
        for i in range(len(x_dis)):
            plt.errorbar(x_dis[i], y_dis[i] , yerr = mh.sqrt(y_dis[i]), uplims = True, color = err_color)
            plt.errorbar(x_dis[i], y_dis[i] , yerr = mh.sqrt(y_dis[i]), lolims = True, color = err_color)
    
    # Set which values to show on x ax
    plt.xticks(x_dis, [round(x_dis[i], 3) for i in range(len(x_dis))])

# Plot the expected expected normal distribution of data based on the fitted gaussian curve 
def expHistPlotter(data: list, size: float = None, normal: bool = False, width: float = None, alpha: float = None, color: str = None, label: str = None, ax: np.ndarray = None):
    stdev = stDev(data)
    mean_val = mean(data)
    
    if ax is not None: plt.sca(ax)
    
    if size == None: 
        size = stdev/2
    elif size <= 0:
        raise ValueError("size parameter must be greater than 0")
    
    if width == None:
        width = size
    elif width > size:
        raise ValueError("width parameter value can't be greater than the size parameter value")
    
    if alpha == None: alpha = 1
    
    if alpha < 0:
        raise ValueError("alpha parameter must be greater or equals to zero")
    elif alpha > 1:
        raise ValueError("alpha parameter must be smaller or equals to one")
    
    n_distr = normalDistribution(data, size)
    
    l = []
    
    for i in range(len(n_distr[0])):
        val = n_distr[0][i]
        exp_val = (mh.erf((val - mean_val + size/2)/(stdev*mh.sqrt(2))) - mh.erf((val - mean_val - size/2)/(stdev*mh.sqrt(2))))*len(data)/2
        
        if i == 0:
            l.append([val])
            l.append([exp_val])
        else:
            l[0].append(val)
            l[1].append(exp_val)

    if normal:
        for i in range(len(l[1])):
            l[1][i] = l[1][i]/(len(data)*size)
    
    return plt.bar(l[0], l[1], width = width, color = color, alpha = alpha, label = label)

# calculate the p-value
def pValue(data: list, size: float) -> float:
    ddof = binsCounter(data, size=size) - 2
    p_value = -integrate.quad(chi2PDF, np.inf , chiSquare(data,size=size), args=(ddof))[0]
    return p_value

# calculate the standard error
def errStd(data: list) -> float:
    stdev = stDev(data)
    return stdev/mh.sqrt(len(data))



if __name__ == '__main__':
    
    # Path to data file
    path = '/home/marco/Documents/Projects/data-analysis/data/pendulum/measures_A1.csv'
    
    data = dataReader(path)
    
    stdev = stDev(data)    
    size = stdev/2
    
    mean_val = mean(data)
    chiSquare(data, size=size)

    fig, ax = plt.subplots(1, 2)
    
    p1 = ax[0]
    
    distrPlotter(data, error_bars = True, normal=True, label='Obs Norm Distr', size = size, err_color='#fffe00', ax=p1) 
    gaussianPlotter(data, center = mean_val, label = 'Observed Gaussian', color='#6b0000', ax=p1)
    gaussianPlotter(data, center = 1.32, color='#00CC00', label = 'Expected Gaussian', ax=p1)
    expHistPlotter(data, normal = True, color='#ca03fc', width=size/6*5, alpha= 0.5, label='Exp Norm Distr', size = size, ax=p1)

    p2 = ax[1]
    
    # degrees of freedom(number of bins - number of parameters(sigma and mean))
    ddof = binsCounter(data, size) - 2
    chi2PdfPlotter(data, 7, 0.05, size , ax=p2 ,color='#ffbd00')

    print('Occurences: ' + str(ocsCounter(data)))
    print('Mean value: ' + str(mean_val))    
    print('Standard deviation: ' + str(stdev))
    print('Bins size: ' + str(size))
    print('Chi squared: ' + str(chiSquare(data, size)))
    print('P value: ' + str(pValue(data, size)))
    print('Standard error ' + str(errStd(data)))
    
    
    # Show the graph and the legend
    ax[0].legend(loc='upper right')
    plt.show()