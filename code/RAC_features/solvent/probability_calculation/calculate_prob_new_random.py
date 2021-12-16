#First, We import the python libraries necesary to run this calculation

#############Standard_Python_Libraries############
from __future__ import division
import pylab
import numpy as np
##################################################

for N in range(0,6):    

    ##Read all the solvent properties randomly sampled from uniform distribution
    data=pylab.loadtxt('random_solvent1.dat')
    l=len(data)
    ##Here we read the precomputed properties of all the 31 different solvents found in our database
    data_sol=pylab.loadtxt('scaled_five_parameter_local_solvent.dat')
    l1=len(data_sol)


    mae=np.zeros((l,l1))
    #Now we calculate the distance between the randomly predicted 
    #solvents (represented by their five properties) to the all 31 solvents in the database
    #The distance calculations are done in the properties space


    for i in range(0,l):
        for j in range(0,l1):
            mae[i][j]=(((data[i][0]-data_sol[j][0])**2)+((data[i][1]-data_sol[j][1])**2)+((data[i][2]-data_sol[j][2])**2)+((data[i][3]-data_sol[j][3])**2)+((data[i][4]-data_sol[j][4])**2))**0.5


    diff=np.zeros(l)
    ## For each of the random prediction 
    ## We now sort the distances calculated above to identify
    ## the neighbours of the randomly predicted solvents in the property space

    for i in range(0,l):
     
        a=mae[i]
    
        b=sorted(a)
    
        diff[i]=b[N] ###Here we take a note of the distance of the Nth neighbour to the randomly predicted solvent

    ##Read all the actual solvent properties
    data_actual=pylab.loadtxt('one_solvent_scaled_five_parameter_solvent1.dat')

    count=0

    #Now we calculate the distance between the actual solvents 
    ##(represented by their five properties) to randomly predicted solvents for all the data points

    for i in range(0,l):
        mae=(((data[i][0]-data_actual[i][0])**2)+((data[i][1]-data_actual[i][1])**2)+((data[i][2]-data_actual[i][2])**2)+((data[i][3]-data_actual[i][3])**2)+((data[i][4]-data_actual[i][4])**2))**0.5
        
        #is this distance(random prediction to the actual values) is 
        ##less than the Nth neighbour distance of the randomly predicted solvent

        if mae <= diff[i]:
            count=count+1

    print (N+1, count*100/l)















