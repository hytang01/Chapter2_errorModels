import os
from shutil import copyfile
import random #Added by Haoyu Tang
import numpy as np

def checkIfHighPerm(x,y,perm,num_well,alpha=1): # Added by Haoyu Tang, to check if the well location selected is on the high perm range
    tag = True
    for well in range(num_well):
#         print(perm[y[well],x[well]])
#         print(np.mean(perm))
        if perm[y[well],x[well]] > alpha*np.mean(perm):
            tag = tag&True
        else:
            tag = tag&False
#     print(tag)
    return tag

def checkIfDist(x,y,num_well,size_x,size_y,dist=2): # Added by Haoyu Tang, to check if the well distance between each other is ok    
    tag = True
    for well1 in range(num_well): # check distance between any two wells
        for well2 in range(num_well-well1-1): #since we not compare itself but the next well
#             print(well1)
#             print(well1+well2+1)
            #print((x[well1]-x[well1+well2+1])**2 + (y[well1]-y[well1+well2+1])**2)
            if (x[well1]-x[well1+well2+1])**2 + (y[well1]-y[well1+well2+1])**2 > dist**2:
                tag = tag&True
            else:
                tag = tag&False
            #print(tag)
    dist = dist/2    
    for well in range(num_well): # check boundary distance
        if (size_x-x[well]) > dist and (x[well]) > dist and (size_y-y[well]) > dist and (y[well]) > dist:
            tag = tag&True
        else:
            tag = tag&False
    return tag    
    

def randGenWell(realNumber,size_x,size_y,num_inj,num_prod): #Added by Haoyu Tang, generate random well locations for use
    wellLoc = {}
    num_well = num_inj + num_prod
    alpha = 0.8 #hyperparameter for deciding what range of perm acceptable for well
    dist = 10 #hyperparameter for minmum distance required between wells
    wellLoc_array = np.zeros((realNumber,2*(num_well)))
    np.random.seed(916) # can be changed~
# ##################################################
#     fileName = "PERMMULTIPLIER.in"#"PERMX.DAT"
#     perm = np.zeros(size_x*size_y)
#     with open(fileName) as f:
#         lines = f.readlines()
#     for i in range(size_x*size_y):
# #     for i in range(len(lines) - 2):
#         perm[i] = float(lines[i+1])
#     perm = perm.reshape(size_y,size_x)
##################################################    
    for real in range(realNumber):
        Inj = [] 
        Prod = []
        flag = True 
        while(flag):          
            x = np.random.randint(2, size_x-1, size = num_well)
            y = np.random.randint(2, size_y-1, size = num_well)
            for inj in range(num_inj):
                x[inj] = x[inj] 
                y[inj] = y[inj]    
            for prd in range(num_prod):
                x[prd + num_inj] = x[prd + num_inj]
                y[prd + num_inj] = y[prd + num_inj]
            x = x.astype(np.int64)
            y = y.astype(np.int64)
            tag_loc = True #checkIfHighPerm(x,y,perm,num_well,alpha)
            tag_dist = checkIfDist(x,y,num_well,size_x,size_y,dist)
            if tag_dist and tag_loc:
                flag = False
        #input x,y into wellLoc and wellLoc_array
        for inj in range(num_inj):
            Inj.append([int(x[inj]),int(y[inj])])
        for prod in range(num_prod):
            Prod.append([int(x[num_inj+prod]),int(y[num_inj+prod])])
        wellLoc[real+1] = [Inj, Prod]
        for well in range(num_well):
            wellLoc_array[real,well*2] = x[well]
            wellLoc_array[real,well*2+1] = y[well]
   # print(wellLoc)
    return wellLoc, wellLoc_array