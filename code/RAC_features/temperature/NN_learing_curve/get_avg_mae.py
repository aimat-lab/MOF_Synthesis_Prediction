import numpy as np
import statistics




for i in range(1,7):
    j=(100*i)+1

    f=open('summary_'+str(j)+'.dat','r')
    l_f=f.readlines()

    l=len(l_f)

    a=[]
    for k in range(0,l):
        if "Testing" in l_f[k]:
            a.append(float(l_f[k].split()[5]))

    avg=(sum(a)/len(a))

    dev=statistics.stdev(a)

        
    print (j-1, avg, dev)

