from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import os
import pylab

from sklearn.metrics import mean_absolute_error




s='cat  y_real_1_test.txt y_real_2_test.txt y_real_3_test.txt y_real_4_test.txt y_real_5_test.txt y_real_6_test.txt y_real_7_test.txt y_real_8_test.txt y_real_9_test.txt y_real_10_test.txt  > real_all_test.txt'


os.system(s)

s='cat  y_RFR_1_test.txt y_RFR_2_test.txt y_RFR_3_test.txt y_RFR_4_test.txt y_RFR_5_test.txt y_RFR_6_test.txt y_RFR_7_test.txt y_RFR_8_test.txt y_RFR_9_test.txt y_RFR_10_test.txt  > pred_all_test.txt'


os.system(s)




real=pylab.loadtxt('real_all_test.txt')
pred=pylab.loadtxt('pred_all_test.txt')




mae=round(mean_absolute_error(real, pred),3)


r2_score=round(r2_score(real, pred),3)



print (mae, r2_score)



