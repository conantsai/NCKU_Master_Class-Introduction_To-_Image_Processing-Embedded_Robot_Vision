import numpy as np
import pylab as P

mu, sigma = 200, 25
x = mu + sigma*P.randn(10000)
bins = [100,125,150,160,170,180,190,200,210,220,230,240,250,275,300]
# the histogram of the data with histtype='step'
n, bins, patches = P.hist(x, bins, normed=1, histtype='bar', rwidth=0.8)

#
# now we create a cumulative histogram of the data
#
P.figure()



P.show()