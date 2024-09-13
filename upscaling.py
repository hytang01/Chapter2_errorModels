###########################################################################
#imports
###########################################################################

import numpy as np
import matplotlib.pyplot as plt
import math

###########################################################################
#module global variables
###########################################################################



###########################################################################
#functions:
###########################################################################

###########################################################################

def detUpsclGrids(grids, gridDims, coarseFact):
  cGrids = [grids[coord]//coarseFact[coord] for coord in range(len(grids))]
  cGDims = [gridDims[coord]*coarseFact[coord] for coord in 
            range(len(grids))]

  return cGrids, cGDims
    
###########################################################################

def upscaleProp(fGrids, cGrids, fact, poroVec, target = None):
#Can make this extremely general to account for the pressure averaging
#as well, which requires porosity weighting. Using np.average(), you can
#automatically apply a weight, e.g. the fine-scale porosities. Can make a
#default of ones, or something, or give a boolean that, if false, will 
#neglect the weighting in np.average().

  if target is None:
    target = poroVec
    w  = np.ones(fGrids)
  else:
    w = np.reshape(poroVec, fGrids, 'F')

  strucVar = np.reshape(target, fGrids, 'F')
  #toView = np.reshape(target, (fGrids[0], fGrids[1]))
  #fig1 = plt.imshow(toView)
  #plt.colorbar()
  #plt.show()
  
  coarseProp = [np.average(strucVar[i*fact[0]:(i*fact[0]+fact[0]), 
                                    j*fact[1]:(j*fact[1]+fact[1]), 
                                    k*fact[2]:(k*fact[2]+fact[2])
                                   ],
                           weights = w[i*fact[0]:(i*fact[0]+fact[0]), 
                                       j*fact[1]:(j*fact[1]+fact[1]), 
                                       k*fact[2]:(k*fact[2]+fact[2])
                                   ]
                          )
                for k in range(cGrids[2])
                for j in range(cGrids[1])
                for i in range(cGrids[0])
                ]

  #toView = np.reshape(coarsePoro, (cGrids[0], cGrids[1]))
  #fig2 = plt.imshow(toView)
  #plt.colorbar()
  #plt.show()
  return coarseProp

###########################################################################

def shiftLocations(matrix):
  leftSlices = []
  rightSlices = []
  dimensions = matrix.shape

  for dims in range(3):
    # primBound = matrix.shape[dims]
    # sliShort = slice(0, primBound - 1)
    # sliLong = slice(1, primBound)
    # sliAll = slice(0, primBound)

    # sliceLeft = tuple([sliShort if x == dims else sliAll for x in range(3)])
    # sliceRight = tuple([sliLong if y == dims else sliAll for y in range(3)])

    primBound = dimensions[dims]
    sliShort = slice(0, primBound - 1)
    sliLong = slice(1, primBound)

    sliceLeft = tuple([sliShort if x == dims else slice(0, dimensions[x])
                       for x in range(3)])
    sliceRight = tuple([sliLong if y == dims else slice(0, dimensions[y])
                        for y in range(3)])
    
    leftSlices.append(sliceLeft)
    rightSlices.append(sliceRight)

  return leftSlices, rightSlices

###########################################################################

def fineScaleFlows(valVec, grids, gridDims, trans = 1, prevDat = None):
  #Assumed isotropic, so kMatrix represents kx, ky, and kz in each grid
  matrix = np.reshape(valVec, grids, 'F')
  
  lft, rgt = shiftLocations(matrix)
  interfaces = getInterfaces(matrix, grids, gridDims, lft, rgt, trans)
  fineFlowVal = mkFineProp(interfaces, grids[0]*grids[1]*grids[2],
                           gridDims, trans, prevDat
                          )
  #plt.imshow(kXInterface[:, :, 0])
  #plt.colorbar()
  #plt.show()
  return fineFlowVal

		###################### SUPPORT FUNC #####################
  
def getInterfaces(matrix, grids, gDim, lftSlices, rtSlices, trans):
  interfaces = [np.zeros(grids) for dims in range(3)]

  for dims in range(3):

    if trans == 1 :

      if dims == 2:
        use_m = 0.1 * matrix
      else:
        use_m = matrix

      interfaces[dims][lftSlices[dims]] = np.divide(2 * gDim[dims], 
                                                  np.divide(gDim[dims], 
                                                  use_m[lftSlices[dims]])+ 
                                                  np.divide(gDim[dims],
                                                  use_m[rtSlices[dims]])
                                                   )
    else:
      interfaces[dims][lftSlices[dims]] = np.subtract(matrix[rtSlices[dims]], 
                                                      matrix[lftSlices[dims]]
                                                     )

  return interfaces

def mkFineProp(interfaces, allGrids, gridDims, transCheck, valList):
  #Conversion factor for md*m to (cp*m^3)/(bar*day)
  #I checked this thoroughly. 
  CONV = 0.008527017312 
  
  if transCheck == 1:
    areas = [gridDims[x] * gridDims[y] for x in range(1, -1, -1)
                                       for y in range(2, 0, -1) 
                                       if  x != y
            ]
    finalProp = [(np.reshape(interfaces[dim], allGrids, 'F') * areas[dim])/ 
                 gridDims[dim] for dim in range(3)
               ]
    finalProp = np.multiply(finalProp, CONV)
  elif transCheck == 2:
    finalProp = [np.multiply(valList[dim],
                 np.reshape(interfaces[dim], allGrids, 'F'))
                 for dim in range(3)
                ]
  else:
    #To deal with exceesive nan's in 2D case
    np.seterr(divide='ignore', invalid='ignore')    
    THRESH_HOLD = 10**-6

    finalProp = [np.divide(valList[dim], 
                 np.reshape(interfaces[dim], allGrids, 'F')) 
                 for dim in range(3)
                ]
    #Eliminate for loops in the future. 
    for x in range(3):
      interfaceVec = np.reshape(interfaces[x], allGrids, 'F')
      for y in range(allGrids):
        if abs(interfaceVec[y]) < THRESH_HOLD:
          finalProp[x][y] = -1

  return finalProp

###########################################################################

def getCoarseFlux(fineQ, grids,cGrids, fact):
  fineQMat = [np.reshape(fineQ[n], grids, 'F') for n in range(3)]
  QCoarseMat = [np.zeros(cGrids) for m in range(3)]

  #toView = fineQMat[0][:, :, 0]
  #fig1 = plt.imshow(toView)
  #plt.colorbar()
  #plt.show()

  for dir in range(3):
    isX = (dir == 0)
    isY = (dir == 1)
    isZ = (dir == 2)

    QCoarseMat[dir] = np.array([np.sum(fineQMat[dir][
                                slice(isX*(fact[0]*i+fact[0]-1) 
                                      +fact[0]*(not isX)*i,((not isX)
                                      *(i+1)*fact[0] + isX*((i+1)*fact[0]))),
                                slice(isY*(fact[1]*j+fact[1]-1) + fact[1]
                                     *(not isY)*j,((not isY)*(j+1)*fact[1]
                                     + isY*((j+1)*fact[1]))), 
                                slice(isZ*(fact[2]*k+fact[2]-1)  
                                      +fact[2]*(not isZ)*k,((not isZ)*(k+1)
                                      *fact[2] + isZ*((k+1)*fact[2])))
                                                    ]  
                                       )  
                                for k in range(cGrids[2]) 
                                for j in range(cGrids[1]) 
                                for i in range(cGrids[0])
                             ])
                              
  #toView = np.reshape(QCoarseMat[0], (cGrids[0], cGrids[1]), 'F')
  #print('\nCoarse Qs\n')
  #print(toView[0:3, 0:3])
  #fig2 = plt.imshow(toView)
  #plt.colorbar()
  #plt.show()
  return np.array(QCoarseMat)

###########################################################################

#def globalWI(fact, cGrids, nInj, nPrd, wellData, cPVec, cWellLoc):
def globalWI(fact, cGrids, nInj, nPrd, wellRate, wellBHP, cPVec, cWellLoc): # Haoyu modify for ResSimAD
  cPMatrix = np.reshape(cPVec, cGrids, 'F')
  # print(cPMatrix.shape)

  wiInj = np.zeros((cGrids[-1], nInj))
  wiPrd = np.zeros((cGrids[-1], nPrd))
  Wis = [wiInj, wiPrd]

  for type in range(2):
    if not type:
      wellUse = nInj
    else:
      wellUse = nPrd

    for wNum in range(wellUse):
      iInd = cWellLoc[type][wNum][0] - 1
      jInd = cWellLoc[type][wNum][1] - 1
      for perfs in range(cGrids[-1]):
#         cRate = np.sum(wellData[type][(perfs*fact[-1]):(perfs+1)*fact[-1], wNum, 1])
#         cBhp = np.mean(wellData[type][(perfs*fact[-1]):(perfs+1)*fact[-1], wNum, 0])
        cRate = np.sum(wellRate[type][wNum,(perfs*fact[-1]):(perfs+1)*fact[-1]])# Haoyu modify for ResSimAD
        cBhp = np.mean(wellBHP[type][wNum,(perfs*fact[-1]):(perfs+1)*fact[-1]])# Haoyu modify for ResSimAD
        
        Wis[type][perfs][wNum] = cRate / (cPMatrix[iInd, jInd, perfs] - cBhp)
  return Wis

###########################################################################

def geoMean(iterable):
    a = np.log(iterable)
    return np.exp(a.sum()/a.size)

def geometricTrans(fact, grid ,cGrid, cDims, fKVec):
  #ASSUMES ISOTROPIC PERMS
  fKMat = np.reshape(fKVec, grid, 'F')

  cKVec = [geoMean(fKMat[i*fact[0]:(i+1)*fact[0],
                             j*fact[1]:(j+1)*fact[1],
                             k*fact[2]:(k+1)*fact[2]
                            ] 
                       )
            for k in range(cGrid[2])
            for j in range(cGrid[1])
            for i in range(cGrid[0])
           ]
  #plt.imshow(np.log(fKMat[:, :, 0]))
  #plt.colorbar()
  #plt.show()

  #plt.imshow(np.log(np.reshape(cKMat, cGrid, 'F')[:, :, 0]))
  #plt.colorbar()
  #plt.show()
  geoTrans = fineScaleFlows(cKVec, cGrid, cDims)
  geoTrans = np.array(geoTrans)

  return geoTrans, cKVec

###########################################################################

def geometricWiIso(nInj, nPrd, cWellLoc, cGrids, cDims, cKMat, rw):
  #Simplified version of ro, since assumption of isotropic perm in grid
  PEACE = 0.28
  CONV = 0.008527017312 
  
  #print(nInj)
  #print(nPrd)

  cKMatUse = np.reshape(cKMat, cGrids, 'F')
  wiInjGeo = np.zeros((cGrids[-1], nInj))
  wiPrdGeo = np.zeros((cGrids[-1], nPrd))
  WisGeo = [wiInjGeo, wiPrdGeo]

  ro = PEACE * ((cDims[0]**2 + cDims[1]**2)**0.5) / 2
  
  #print(cKMatUse.shape)
  #print(rw)

  for type in range(2):
    if not type:
      wellUse = nInj
    else:
      wellUse = nPrd

    for wNum in range(wellUse):
      #print(wNum)
      iInd = cWellLoc[type][wNum][0] - 1
      jInd = cWellLoc[type][wNum][1] - 1
      for perfs in range(cGrids[-1]):
        WisGeo[type][perfs][wNum] = CONV*(2*math.pi*cDims[-1]*
                                    cKMatUse[iInd, jInd, perfs])/np.log(ro/rw[type][wNum])

  return WisGeo
  
###########################################################################

def correctTrans(listOfGlobal, listOfGeo, wells = True):
  solution = []
  if wells:
    levels = 2
  else:
    levels = 3

  for type in range(levels):
    globalUse = listOfGlobal[type]
    geoUse = listOfGeo[type]

    #Negative values
    finalVal = np.zeros(globalUse.shape)

    checkNeg = globalUse > 0
    checkNegInd = np.nonzero(checkNeg)
    checkNegInv = np.invert(checkNeg)

    finalVal[checkNegInd] = globalUse[checkNegInd]
    finalVal[checkNegInv] = geoUse[checkNegInv]

    #Nan values
    checkNan = np.isnan(finalVal)
    checkNanInd = np.nonzero(checkNan)
    
    finalVal[checkNanInd] = geoUse[checkNanInd]

    #inf values
    checkInf = np.isinf(finalVal)
    checkInfInd = np.nonzero(checkInf)
    
    finalVal[checkInfInd] = geoUse[checkInfInd] 

    solution.append(finalVal)
  return solution
