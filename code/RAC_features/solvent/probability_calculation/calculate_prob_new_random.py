from __future__ import division
import pylab
import numpy as np

for N in range(0,6):    
    data=pylab.loadtxt('random_solvent1.dat')
    l=len(data)

    data_sol=pylab.loadtxt('scaled_five_parameter_local_solvent.dat')
    l1=len(data_sol)

    mae=np.zeros((l,l1))
    for i in range(0,l):
        for j in range(0,l1):
            mae[i][j]=(((data[i][0]-data_sol[j][0])**2)+((data[i][1]-data_sol[j][1])**2)+((data[i][2]-data_sol[j][2])**2)+((data[i][3]-data_sol[j][3])**2)+((data[i][4]-data_sol[j][4])**2))**0.5


    diff=np.zeros(l)
#print (mae)
    for i in range(0,l):
     
        a=mae[i]
    #print (a)
        b=sorted(a)
    #print (b)
        diff[i]=b[N]

    data_actual=pylab.loadtxt('one_solvent_scaled_five_parameter_solvent1.dat')

    count=0

    for i in range(0,l):
        mae=(((data[i][0]-data_actual[i][0])**2)+((data[i][1]-data_actual[i][1])**2)+((data[i][2]-data_actual[i][2])**2)+((data[i][3]-data_actual[i][3])**2)+((data[i][4]-data_actual[i][4])**2))**0.5
        #print (i)
        #print mae, diff[i]
        if mae <= diff[i]:
            count=count+1

    print (N+1, count*100/l)















