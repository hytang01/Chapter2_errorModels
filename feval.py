################################################ import libraries #########################################################
import os
from shutil import copyfile
import upscaling as upsc
import primFuncs as prim
import numpy as np
import random
import math
import tqdm
import copy

from julia.api import Julia
jl = Julia(compiled_modules=False)
import julia
from julia import Base
from julia import Main
from julia import ResSimAD
import ResSimADMain as sim

import ResSimADMain_upscale as sim_up

import pandas as pd
import sklearn
from sklearn.neighbors import KernelDensity

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# pip install lightgbm
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor


from pickle import dump,load
import sys
####################### calculate BHP for different well perforations 3D only ###################
def calc_bhp_perf(bhp,num_well,num_layers,ρ,h):
    # the formula is bhp = P0+ ρgh
    # here we use units of SI and switch back to bar in the end
    # we assume each well has same bhp value for inj or producer themselves
    pa2bar = 1e-5
    bhp_perf = np.zeros((num_well,num_layers))
    bhp_perf[:,0] = bhp
    ρgh = ρ*9.8*h*pa2bar
    for i in range(num_layers-1):
        bhp_perf[:,i+1] = bhp_perf[:,i] + ρgh
    return bhp_perf
####################### make prediction based on fitted models ###################
def predict(x_test, models, num_well):  # X_test[num_well][num_test][num_features], models[num_well]
    # may add more functionalities here if need further processing
    y_pred = []
    for idx in range(num_well):
        y_pred.append(models[idx].predict(x_test[idx]))
    y_pred = np.array(y_pred)
    return y_pred # y_pred[num_well][num_test]
###################### pre-process dataX to extract features for X ################
###################### pre-process dataX to extract features for X ################
def svd_preprocess_feature(dataX, use_PCA, num_inj, num_prod, coarse_wellLoc, 
                           coarseLevel_x, coarseLevel_y, coarseLevel_z, grids,num_reals,scale_factor,
                          saved_scaler,saved_pca): # dataX[3types]['var_type'][1][refer to notes abv]  
    features = []   
    x_dim = int(grids[0]/coarseLevel_x)
    y_dim = int(grids[1]/coarseLevel_y)
    z_dim = int(grids[2]/coarseLevel_z)
    
########################### Producers ################################################      
    for prd in range(num_prod):
        ################################## fine space data ###########################
        curr_feature1 = []       
        data = dataX[0] # fine space
        for real in range(num_reals):
            curr_real_feature1 = []
            for key in data.keys():
                x_start = coarse_wellLoc[real+1][1][prd][0] - 1
                y_start = coarse_wellLoc[real+1][1][prd][1] - 1
                avg = 0
                for z in range(z_dim):
                    z_start = z #coarse_wellLoc[real+1][1][prd][2] - 1
                    for j in range(coarseLevel_x):
                        for k in range(coarseLevel_y):
                            for l in range(coarseLevel_z):
                                cell_idx = (j+x_start+1) + (k+y_start)*x_dim + (l+z_start)*x_dim*y_dim # 1-based
                                avg += data[key][0][real][cell_idx-1]
                avg = avg/coarseLevel_x/coarseLevel_y/coarseLevel_z/z_dim
                curr_real_feature1.append(avg)
            curr_real_feature1 = np.array(curr_real_feature1)
            curr_real_feature1 = curr_real_feature1.reshape(curr_real_feature1.shape[0])
            curr_feature1.append(curr_real_feature1) 
        curr_feature1 = np.array(curr_feature1) # [num_reals][num_features]
        ################################## coarse space data with space and time ###########################
        curr_feature2 = []
        database_all = []
        data = dataX[1] # coarse space and time 
        num_var = len(data.keys())
        for real in range(num_reals): 
            database = []
            for key in data.keys():
                xloc = coarse_wellLoc[real+1][1][prd][0]
                yloc = coarse_wellLoc[real+1][1][prd][1]
                avg = 0
                for z in range(z_dim):     
                    loc_idx = (xloc - 1) + (yloc - 1)*x_dim + z*x_dim*y_dim
                    avg += data[key][0][real,:,loc_idx]
                temp = avg
                database.append(temp)
            database = np.array(database) #[num_variables][num_timestep]
            database_all.append(database)
        database_all = np.array(database_all) #[num_reals][num_variables][num_timestep]
        
        for i in range(num_var):
            data_curr = database_all[:,i,:] #[num_reals][num_timestep]
            scaler = MinMaxScaler()  
            scaler.fit(data_curr)
            feature_curr = scaler.transform(data_curr) # scale the feature
            saved_scaler[data.keys()[i]+'_prd'+str(prd+1)] = scaler
            if(use_PCA):
                pca = saved_pca[data.keys()[i]+'_prd'+str(prd+1)]
                feature_curr = pca.transform(feature_curr) # PCA to shrink dimension for the feature   
            curr_feature2.append(feature_curr)
        curr_feature2 = np.array(curr_feature2) # [num_variables][num_reals][num_feature]   
        curr_feature2 = np.einsum('ijk->jik', curr_feature2)
        curr_feature2 = curr_feature2.reshape((curr_feature2.shape[0],curr_feature2.shape[1]*curr_feature2.shape[2]))
#        ################################ coarse space data with only time ###########################       
        curr_feature3 = []
        data = dataX[2] # time      
        for key in data.keys():
            temp = data[key][0] 
            scaler = MinMaxScaler()  
            scaler.fit(temp)
            temp = scaler.transform(temp) # scale the feature
            saved_scaler[key+'_prd'+str(prd+1)] = scaler
            if(use_PCA):
                pca = saved_pca[key+'_prd'+str(prd+1)]
                feature_curr = pca.transform(temp) # PCA to shrink dimension for the feature
            curr_feature3.append(feature_curr)
        curr_feature3 = np.array(curr_feature3) #[num_variables][num_reals][num_feature]
        curr_feature3 = np.einsum('ijk->jik', curr_feature3)
        curr_feature3 = curr_feature3.reshape((curr_feature3.shape[0],curr_feature3.shape[1]*curr_feature3.shape[2]))
#        ################################## coarse space data with only space ###########################
        curr_feature4 = []
        data = dataX[3] # coarse space only 
        for i in range(num_reals):
            feature_curr = []
            for key in data.keys():
                xloc = coarse_wellLoc[real+1][1][prd][0]
                yloc = coarse_wellLoc[real+1][1][prd][1]
                avg = 0
                for z in range(z_dim):     
                    loc_idx = (xloc - 1) + (yloc - 1)*x_dim + z*x_dim*y_dim
                    avg += data[key][0][real,loc_idx]
                temp = avg
                feature_curr.append(temp)
            feature_curr = np.array(feature_curr)
            curr_feature4.append(feature_curr)
        curr_feature4 = np.array(curr_feature4)
#        ################################## coarse space data with only well-info ###########################
        curr_feature5 = []
        data = dataX[4] # coarse space only 
        for i in range(num_reals):
            feature_curr = []
            for key in data.keys():
                temp = data[key][0][real][1][prd]
                feature_curr.append(temp/scale_factor)
            feature_curr = np.array(feature_curr)
            curr_feature5.append(feature_curr)
        curr_feature5 = np.array(curr_feature5)
        curr_feature5  = curr_feature5.reshape((num_reals,-1))

        # post-process to reshape all feature for current real into 1D
        temp = np.concatenate((curr_feature1,curr_feature2),axis=1)
        temp2 = np.concatenate((temp,curr_feature3),axis=1)
        temp3 = np.concatenate((temp2,curr_feature4),axis=1)
        curr_feature = np.concatenate((temp3,curr_feature5),axis=1)
        curr_feature = np.array(curr_feature) # [num_reals][num_features_all]
#         print(curr_feature.shape)
        features.append(curr_feature)
    
###########################  Injectors ################################################  
    for inj in range(num_inj):
        ################################## fine space data ###########################
        curr_feature1 = []       
        data = dataX[0] # fine space
        for real in range(num_reals):
            curr_real_feature1 = []
            for key in data.keys():
                x_start = coarse_wellLoc[real+1][0][inj][0] - 1
                y_start = coarse_wellLoc[real+1][0][inj][1] - 1
                avg = 0
                for z in range(z_dim):
                    z_start = z #coarse_wellLoc[real+1][1][prd][2] - 1
                    for j in range(coarseLevel_x):
                        for k in range(coarseLevel_y):
                            for l in range(coarseLevel_z):
                                cell_idx = (j+x_start+1) + (k+y_start)*x_dim + (l+z_start)*x_dim*y_dim # 1-based
                                avg += data[key][0][real][cell_idx-1]
                avg = avg/coarseLevel_x/coarseLevel_y/coarseLevel_z/z_dim
                curr_real_feature1.append(avg)
            curr_real_feature1 = np.array(curr_real_feature1)
            curr_real_feature1 = curr_real_feature1.reshape(curr_real_feature1.shape[0])
            curr_feature1.append(curr_real_feature1) 
        curr_feature1 = np.array(curr_feature1) # [num_reals][num_features]
        ################################## coarse space data with space and time ###########################
        curr_feature2 = []
        database_all = []
        data = dataX[1] # coarse space and time 
        num_var = len(data.keys())
        for real in range(num_reals): 
            database = []
            for key in data.keys():
                xloc = coarse_wellLoc[real+1][0][inj][0]
                yloc = coarse_wellLoc[real+1][0][inj][1]
                avg = 0
                for z in range(z_dim):     
                    loc_idx = (xloc - 1) + (yloc - 1)*x_dim + z*x_dim*y_dim
                    avg += data[key][0][real,:,loc_idx]
                temp = avg
                database.append(temp)
            database = np.array(database) #[num_variables][num_timestep]
            database_all.append(database)
        database_all = np.array(database_all) #[num_reals][num_variables][num_timestep]
        
        for i in range(num_var):
            data_curr = database_all[:,i,:] #[num_reals][num_timestep]
            scaler = MinMaxScaler()  
            scaler.fit(data_curr)
            feature_curr = scaler.transform(data_curr) # scale the feature
            saved_scaler[data.keys()[i]+'_inj'+str(inj+1)] = scaler
            if(use_PCA):
                pca = saved_pca[data.keys()[i]+'_inj'+str(inj+1)]
                feature_curr = pca.transform(feature_curr) # PCA to shrink dimension for the feature
            curr_feature2.append(feature_curr)
        curr_feature2 = np.array(curr_feature2) # [num_variables][num_reals][num_feature]   
        curr_feature2 = np.einsum('ijk->jik', curr_feature2)
        curr_feature2 = curr_feature2.reshape((curr_feature2.shape[0],curr_feature2.shape[1]*curr_feature2.shape[2]))
#        ################################ coarse space data with only time ###########################       
        curr_feature3 = []
        data = dataX[2] # time      
        for key in data.keys():
            temp = data[key][0] 
            scaler = MinMaxScaler()  
            scaler.fit(temp)
            temp = scaler.transform(temp) # scale the feature
            saved_scaler[key+'_inj'+str(inj+1)] = scaler
            if(use_PCA):
                pca = saved_pca[key+'_inj'+str(inj+1)]
                feature_curr = pca.transform(temp) # PCA to shrink dimension for the feature
            curr_feature3.append(feature_curr)
        curr_feature3 = np.array(curr_feature3) #[num_variables][num_reals][num_feature]
        curr_feature3 = np.einsum('ijk->jik', curr_feature3)
        curr_feature3 = curr_feature3.reshape((curr_feature3.shape[0],curr_feature3.shape[1]*curr_feature3.shape[2]))
#        ################################## coarse space data with only space ###########################
        curr_feature4 = []
        data = dataX[3] # coarse space only 
        for i in range(num_reals):
            feature_curr = []
            for key in data.keys():
                xloc = coarse_wellLoc[real+1][0][inj][0]
                yloc = coarse_wellLoc[real+1][0][inj][1]
                avg = 0
                for z in range(z_dim):     
                    loc_idx = (xloc - 1) + (yloc - 1)*x_dim + z*x_dim*y_dim
                    avg += data[key][0][real,loc_idx]
                temp = avg
                feature_curr.append(temp)
            feature_curr = np.array(feature_curr)
            curr_feature4.append(feature_curr)
        curr_feature4 = np.array(curr_feature4)
#        ################################## coarse space data with only well-info ###########################
        curr_feature5 = []
        data = dataX[4] # coarse space only 
        for i in range(num_reals):
            feature_curr = []
            for key in data.keys():
                temp = data[key][0][real][0][inj]
                feature_curr.append(temp/scale_factor)
            feature_curr = np.array(feature_curr)
            curr_feature5.append(feature_curr)
        curr_feature5 = np.array(curr_feature5)
        curr_feature5  = curr_feature5.reshape((num_reals,-1))

        # post-process to reshape all feature for current real into 1D
        temp = np.concatenate((curr_feature1,curr_feature2),axis=1)
        temp2 = np.concatenate((temp,curr_feature3),axis=1)
        temp3 = np.concatenate((temp2,curr_feature4),axis=1)
        curr_feature = np.concatenate((temp3,curr_feature5),axis=1)
        curr_feature = np.array(curr_feature) # [num_reals][num_features_all]
#         print(curr_feature.shape)
        features.append(curr_feature)
    
    features = np.array(features)
    return features # features[num_well][num_reals][num_features] producers first then injectors in first dimension 'num_well'
################################################## main function #########################################################
def main(argv):
    infile = sys.argv[1] # decision variables each variable is own line
    outfile = sys.argv[2] # OBJ and constraints
    ################################################ define parameters #########################################################
    # Unit transformation (* -> metric to field)
    bar2psi = 14.50377
    cm2bbl = 6.289811
    cm2cft = 35.31466
    m2ft = 3.2808
    lb2kg = 0.453592
    beta = cm2bbl / bar2psi

    num_inj=4
    inj_bhp=250 #unit: bar
    num_prod=4
    prod_bhp=150 #unit: bar
    initial_pres= 200 #unit: bar
    total_time = 2000.0# 500.0 #unit: day
    intial_wat_sat = 0.1#0.0 #
    well_radius = 0.3048/2 #unit: m
    poro = 0.3
    grids = [60,60,30]
    gridSize = [20,20,5]
    perf_z1 = 1
    perf_z2 = grids[2]
    
    inj_rad = []
    for inj in range(num_inj):
        inj_rad.append(well_radius)
    prd_rad = []
    for prd in range(num_prod):
        prd_rad.append(well_radius)
    wellRadius = []
    wellRadius.append(inj_rad);wellRadius.append(prd_rad);

    coarseLevels = [[6],
                    [6],
                    [3]]
    num_lvl = len(coarseLevels[0])

    num_reals = 1 # coz we treat only realization now from UoF
    ################################################ input from UoF ############################################################
    wellLoc = {}
    wellLoc_load = np.loadtxt(infile)
    wellLoc[1] = [[[round(wellLoc_load[0]),round(wellLoc_load[1])],
                          [round(wellLoc_load[2]),round(wellLoc_load[3])],
                    [round(wellLoc_load[4]),round(wellLoc_load[5])],
                          [round(wellLoc_load[6]),round(wellLoc_load[7])]],
                         [[round(wellLoc_load[8]),round(wellLoc_load[9])],
                          [round(wellLoc_load[10]),round(wellLoc_load[11])],
                         [round(wellLoc_load[12]),round(wellLoc_load[13])],
                          [round(wellLoc_load[14]),round(wellLoc_load[15])]]]
    ################################################ calc upscaled wellLoc ######################################################
    # we can directly calculate the upscaled well locations based on levels
    # Also store the well location of coarse model of interests
    x = 0; y = 1;z=2;
    wellLoc_upscale = {} # dict of dict of lists
    for lvl in range(num_lvl):
       # print(wellLoc)
        wellLoc_temp = copy.deepcopy(wellLoc)
        for real in range(num_reals):
            for inj in range(num_inj):
                wellLoc_temp[real+1][0][inj][x] = math.ceil(wellLoc[real+1][0][inj][x]/coarseLevels[x][lvl])
                wellLoc_temp[real+1][0][inj][y] = math.ceil(wellLoc[real+1][0][inj][y]/coarseLevels[y][lvl])
            for prd in range(num_prod):
                wellLoc_temp[real+1][1][prd][x] = math.ceil(wellLoc[real+1][1][prd][x]/coarseLevels[x][lvl])
                wellLoc_temp[real+1][1][prd][y] = math.ceil(wellLoc[real+1][1][prd][y]/coarseLevels[y][lvl])
            wellLoc_upscale["level_"+str(lvl+1)] = wellLoc_temp
    ################################################ upscaling procedure #################################################
    cwd = os.getcwd()
    dataPath = cwd+"/SINGLE_PHASE_RUN/"
    single_time = 1.0
    single_Sw0= 1.0 # inject water into water (use water density for BHP calc in upscaling part)
    singlePhaseData = []
    for real in range(num_reals):
        wellLoc_curr = wellLoc[real+1]
        options,Main.sim = sim.ResSimADMain(dataPath,grids,gridSize,poro,wellLoc_curr,num_inj,inj_bhp,num_prod,prod_bhp,
                                            initial_pres,single_time,single_Sw0,well_radius,perf_z1,perf_z2,dt0=1.0)
        ResSimAD.runsim(Main.sim)
        t = ResSimAD.get_well_rates(Main.sim, "P1", "TIME")
        po = ResSimAD.get_state_map(Main.sim, "po", t[-1])
        pw = ResSimAD.get_state_map(Main.sim, "pw", t[-1])
        sw = ResSimAD.get_state_map(Main.sim, "sw", t[-1])
        inj = []
        for ind in range(num_inj):
            qi = ResSimAD.get_well_rates(Main.sim, "I"+str(ind+1), "WRAT")
            inj.append(qi)
        inj = np.array(inj)
        prod_qo = []
        prod_qw = []
        for ind in range(num_prod):
            qo = ResSimAD.get_well_rates(Main.sim, "P"+str(ind+1), "ORAT")
            qw = ResSimAD.get_well_rates(Main.sim, "P"+str(ind+1), "WRAT")
            prod_qo.append(qo)
            prod_qw.append(qw)
        prod_qo = np.array(prod_qo)
        prod_qw = np.array(prod_qw)
        prod_qw_perf= []
        inj_perf = []
        prod_qw_perf.append(Main.eval('''ResSimAD.value(sim.facility["P1"].qw)'''))
        prod_qw_perf.append(Main.eval('''ResSimAD.value(sim.facility["P2"].qw)'''))
        prod_qw_perf.append(Main.eval('''ResSimAD.value(sim.facility["P3"].qw)'''))
        prod_qw_perf.append(Main.eval('''ResSimAD.value(sim.facility["P4"].qw)'''))
        inj_perf.append(Main.eval('''ResSimAD.value(sim.facility["I1"].qw)'''))
        inj_perf.append(Main.eval('''ResSimAD.value(sim.facility["I2"].qw)'''))
        inj_perf.append(Main.eval('''ResSimAD.value(sim.facility["I3"].qw)'''))
        inj_perf.append(Main.eval('''ResSimAD.value(sim.facility["I4"].qw)'''))
        prod_qw_perf = np.array(prod_qw_perf)
        inj_perf = np.array(inj_perf)
        ρw = ResSimAD.get_data(Main.sim,"ρw") 
        ρo = ResSimAD.get_data(Main.sim,"ρo") 
        temp = {'time':t,'po':po,'pw':pw,'sw':sw,'inj_rate':inj,'prod_orat':prod_qo,'prod_wrat':prod_qw
               ,'inj_rate_perf':inj_perf,'prod_wrat_perf':prod_qw_perf,'wat_density':ρw,'oil_density':ρo}
        singlePhaseData.append(temp)
    # upscaling procedures
    dataPath = cwd+"/SINGLE_PHASE_RUN/" 
    num_well = num_inj+num_prod
    WI_all = []
    TransX_all = []
    TransY_all = []
    TransZ_all = []

    for real in range(num_reals):
        TransX_temp = {}
        TransY_temp = {}
        TransZ_temp = {}
        WI_temp = {}
        for lvl in range(num_lvl):
            cperf_z1 = (perf_z1-1)//coarseLevels[2][lvl]+1
            cperf_z2 = (perf_z2-1)//coarseLevels[2][lvl]+1
             #Gather fine-scale single-phase incompressible data. 
            fact = [coarseLevels[coord][lvl] for coord in range(3)]
            ssPresVec = singlePhaseData[real]['pw']/bar2psi #
            finePermVec = np.loadtxt(dataPath+'/PERM_NO_HEAD.DAT')
            finePoroVec = np.ones(finePermVec.shape)*poro
            wellRate = []
            prod_qw = singlePhaseData[real]['prod_wrat_perf']/cm2bbl
            inj = singlePhaseData[real]['inj_rate_perf']/cm2bbl
            wellRate.append(inj);wellRate.append(prod_qw)
            wellBHP = []
            # need a function to calulcate BHP layer by layer here
            ρw = np.mean(singlePhaseData[real]['wat_density'])*lb2kg *cm2cft # kg/m^3 for water density (from lbs/ft^3)
            prod_bhp_all = calc_bhp_perf(prod_bhp,num_prod,grids[2],ρw,gridSize[2])
            inj_bhp_all = calc_bhp_perf(inj_bhp,num_inj,grids[2],ρw,gridSize[2])
            wellBHP.append(inj_bhp_all); wellBHP.append(prod_bhp_all);
    ############################### core calculations of upscaling procuder -- adapted from Dylan's code for ADGPRS #############
            #coarse rock data calculated.
            cgrids, cgridSize = upsc.detUpsclGrids(grids, gridSize, fact)
            upPoroVec = upsc.upscaleProp(grids, cgrids, fact, finePoroVec)
            upPressVec = upsc.upscaleProp(grids, cgrids, fact, finePoroVec, ssPresVec)
            fineTrans = upsc.fineScaleFlows(finePermVec, grids, gridSize) 
            fineRates = upsc.fineScaleFlows(ssPresVec, grids, gridSize, trans = 2, 
                                          prevDat = fineTrans)
            #Upscaling rates and transmissibilites here. 
            coarseRates = upsc.getCoarseFlux(fineRates, grids, cgrids, fact)
            gridGlobalT = upsc.fineScaleFlows(upPressVec, cgrids, cgridSize, trans = 3, 
                                            prevDat = coarseRates)
            Wis = upsc.globalWI(fact, cgrids, num_inj, num_prod, wellRate, wellBHP,
                                upPressVec, wellLoc_upscale["level_"+str(lvl+1)][real+1])
            TGeoC, geoAvgK = upsc.geometricTrans(fact, grids ,cgrids, cgridSize, finePermVec)
            geoWis = upsc.geometricWiIso(num_inj, num_prod, wellLoc_upscale["level_"+str(lvl+1)][real+1],
                                         cgrids, cgridSize, geoAvgK, wellRadius)
            finalT = upsc.correctTrans(gridGlobalT, TGeoC, wells = False)            
            finalWi = upsc.correctTrans(Wis, geoWis, wells = True)
    ############################### core calculations of upscaling procuder -- adapted from Dylan's code for ADGPRS #############    
            #Save all trans & wi results
            TransX_temp["level_"+str(lvl+1)] = finalT[0]
            TransY_temp["level_"+str(lvl+1)] = finalT[1]
            TransZ_temp["level_"+str(lvl+1)] = finalT[2]
            inj_WI = []; prod_WI = [];well_WI = [];
            for inj in range(num_inj):
                inj_WI.append(finalWi[0][cperf_z1-1:cperf_z2,inj])# post process -- make sure only perforated well index stored and get rid of zeros
            for prod in range(num_prod):
                prod_WI.append(finalWi[1][cperf_z1-1:cperf_z2,prod])# post process -- make sure only perforated well index stored and get rid of zeros
            well_WI.append(inj_WI);well_WI.append(prod_WI);
            WI_temp["level_"+str(lvl+1)] = well_WI

        TransX_all.append(TransX_temp)
        TransY_all.append(TransY_temp)
        TransZ_all.append(TransZ_temp)
        WI_all.append(WI_temp)
################################################ upscaled simulation #################################################        
    # upscaled simulation runs
    dataPath = cwd+"/UPSCALED_RUN/" 
    upscaledData_original = {}
    for lvl in range(num_lvl):
        upscaledDataLvl = []
        for real in range(num_reals):
            cperf_z1 = (perf_z1-1)//coarseLevels[2][lvl]+1
            cperf_z2 = (perf_z2-1)//coarseLevels[2][lvl]+1
            TransX = TransX_all[real]["level_"+str(lvl+1)]
            TransY = TransY_all[real]["level_"+str(lvl+1)] 
            TransZ = TransZ_all[real]["level_"+str(lvl+1)] 
            wellIndex = WI_all[real]["level_"+str(lvl+1)]
            wellLoc_curr = wellLoc_upscale["level_"+str(lvl+1)][real+1]
            options,Main.sim = sim_up.ResSimADMain_upscale(dataPath,grids,gridSize,poro,wellLoc_curr,
                                                   num_inj,inj_bhp,num_prod,prod_bhp,
                                                   initial_pres,total_time,intial_wat_sat,wellIndex,
                                                   TransX,TransY,TransZ,coarseLevels,lvl,cperf_z1,cperf_z2,dt0=0.01)
            ResSimAD.runsim(Main.sim)
            t = ResSimAD.get_well_rates(Main.sim, "P1", "TIME")
            po = ResSimAD.get_state_map(Main.sim, "po", t[-1])
            sw = ResSimAD.get_state_map(Main.sim, "sw", t[-1])
            inj = []
            for ind in range(num_inj):
                qi = ResSimAD.get_well_rates(Main.sim, "I"+str(ind+1), "WRAT")
                inj.append(qi)
            inj = np.array(inj)
            prod_qo = []
            prod_qw = []
            for ind in range(num_prod):
                qo = ResSimAD.get_well_rates(Main.sim, "P"+str(ind+1), "ORAT")
                qw = ResSimAD.get_well_rates(Main.sim, "P"+str(ind+1), "WRAT")
                prod_qo.append(qo)
                prod_qw.append(qw)
            prod_qo = np.array(prod_qo)
            prod_qw = np.array(prod_qw)
            temp = {'time':t,'po':po,'sw':sw,'inj_rate':inj,'prod_orat':prod_qo,'prod_wrat':prod_qw}
            upscaledDataLvl.append(temp)
        upscaledData_original["level_"+str(lvl+1)] = upscaledDataLvl
################################################ overall NPV calculation #################################################
    # Step 1. define basic price, unit conversion, discount rate variables
    stb2m3 = 0.158987
    Po = 60#/stb2m3
    Pw = 3#/stb2m3
    Pinj = 2#/stb2m3
    r = 0.1 #discount rate

    # Step 2. calculate the overall NPV for upscaled model with upscaled simulation results
    NPV_upscaled = np.zeros((num_lvl, num_reals))
    for lvl in range(num_lvl):
        for i in range(num_reals):
            npv = 0
            rate_time_coarse = upscaledData_original["level_"+str(lvl+1)][i]['time']
            rate_PRD_OPR_coarse = upscaledData_original["level_"+str(lvl+1)][i]['prod_orat']
            rate_PRD_WPR_coarse = upscaledData_original["level_"+str(lvl+1)][i]['prod_wrat']
            rate_INJ_WIR_coarse = upscaledData_original["level_"+str(lvl+1)][i]['inj_rate']
            for n in range(rate_time_coarse.shape[0]):
                val = 0
                if n == 0:
                    dt = rate_time_coarse[0]
                else:
                    dt = rate_time_coarse[n] - rate_time_coarse[n-1]
                for j in range(num_prod):
                    val += dt*Po * rate_PRD_OPR_coarse[j,n] - dt*Pw * rate_PRD_WPR_coarse[j,n]
                for k in range(num_inj):
                    val += dt*Pinj * rate_INJ_WIR_coarse[k,n]
                npv += val/(1+r)**(rate_time_coarse[n]//365)
                #print(npv)
            NPV_upscaled[lvl,i] = int(npv)
################################################ feature collection #################################################
    lvl_chosen = 0
    dataPath = cwd+"/SINGLE_PHASE_RUN/"  

    #################################### vx1p & vy1p & vz1p #############################################
    # in order to retrieve useful information during the single phase stage, we need to run step by step
    # we want to get single phase velocity here
    vx1p = [] #[num_reals][timestep] only one step here # this is the actual velocity
    vy1p = [] #[num_reals][timestep] only one step here # this is the actual velocity
    vz1p = [] #[num_reals][timestep] only one step here # this is the actual velocity

    for real in range(num_reals):
        wellLoc_curr = wellLoc[real+1]
        options,Main.sim = sim.ResSimADMain(dataPath,grids,gridSize,poro,wellLoc_curr,num_inj,inj_bhp,num_prod,prod_bhp,
                                            initial_pres,single_time,single_Sw0,well_radius,perf_z1,perf_z2,dt0=1.0)
        # run step by step
        var1 = []
        var2 = []
        var3 = []

        for t in np.linspace(1.0,single_time,1):
            ResSimAD.step_to(Main.sim,t)
            # retrive \lambda_o for each time step

            vxw1p = ResSimAD.get_velocity(Main.sim,"x","w")
            vyw1p = ResSimAD.get_velocity(Main.sim,"y","w")
            vzw1p = ResSimAD.get_velocity(Main.sim,"z","w")
            var1.append(vxw1p)
            var2.append(vyw1p)
            var3.append(vzw1p)

        var1 = np.array(var1)
        var2 = np.array(var2)
        var3 = np.array(var3)

        vx1p.append(var1)
        vy1p.append(var2)
        vz1p.append(var3)

    vx1p = np.array(vx1p)
    vy1p = np.array(vy1p)
    vz1p = np.array(vz1p)

    vx1p = vx1p.reshape((num_reals,grids[0]*grids[1]*grids[2]))
    vy1p = vy1p.reshape((num_reals,grids[0]*grids[1]*grids[2]))
    vz1p = vz1p.reshape((num_reals,grids[0]*grids[1]*grids[2]))

    dataPath = cwd+"/UPSCALED_RUN/"  
    lvl = lvl_chosen
    ############################ coarse pres sat flux velocity conn_flux ###############################    
    # in order to retrieve useful information, we need to run step by step and retrieve useful info during each run
    pc = [] #[num_reals][timestep][num_cell]
    sc = [] #[num_reals][timestep][num_cell]
    fc = [] #[num_reals][timestep] # BL flux function
    vxc = [] #[num_reals][timestep]
    vyc = [] #[num_reals][timestep]
    vzc = [] #[num_reals][timestep]
    foc_conn = [] #[num_reals][timestep][conn_list]
    fwc_conn = [] #[num_reals][timestep][conn_list]

    for real in range(num_reals):
        cperf_z1 = (perf_z1-1)//coarseLevels[2][lvl]+1
        cperf_z2 = (perf_z2-1)//coarseLevels[2][lvl]+1
        TransX = TransX_all[real]["level_"+str(lvl+1)]
        TransY = TransY_all[real]["level_"+str(lvl+1)] 
        TransZ = TransZ_all[real]["level_"+str(lvl+1)] 
        wellIndex = WI_all[real]["level_"+str(lvl+1)]
        wellLoc_curr = wellLoc_upscale["level_"+str(lvl+1)][real+1]
        options,Main.sim = sim_up.ResSimADMain_upscale(dataPath,grids,gridSize,poro,wellLoc_curr,
                                               num_inj,inj_bhp,num_prod,prod_bhp,
                                               initial_pres,total_time,intial_wat_sat,wellIndex,
                                               TransX,TransY,TransZ,coarseLevels,lvl,cperf_z1,cperf_z2,dt0=0.01)
        # run step by step
        varpc = []
        varsc = []
        var1 = []
        var2 = []
        var3 = []
        var32 = []
        var4 = []
        var5 = []

        for t in upscaledData_original["level_"+str(lvl+1)][real]["time"]:
            ResSimAD.step_to(Main.sim,t)

            # retrive info for each time step
            pt = ResSimAD.get_data(Main.sim,"po")
            st = ResSimAD.get_data(Main.sim,"sw")
            varpc.append(pt)
            varsc.append(st)

            λo = ResSimAD.get_data(Main.sim,"λo")
            λw = ResSimAD.get_data(Main.sim,"λw")
            var1.append(λw/(λw+λo))

            vxoc = ResSimAD.get_velocity(Main.sim,"x","o")
            vxwc = ResSimAD.get_velocity(Main.sim,"x","w")
            var2.append(vxoc+vxwc) # calculate total velocity

            vyoc = ResSimAD.get_velocity(Main.sim,"y","o")
            vywc = ResSimAD.get_velocity(Main.sim,"y","w")
            var3.append(vyoc+vywc) # calculate total velocity

            vzoc = ResSimAD.get_velocity(Main.sim,"z","o")
            vzwc = ResSimAD.get_velocity(Main.sim,"z","w")
            var32.append(vzoc+vzwc) # calculate total velocity

            foc = ResSimAD.get_data(Main.sim,"fo")
            fwc = ResSimAD.get_data(Main.sim,"fw")
            var4.append(foc)
            var5.append(fwc)

        varpc = np.array(varpc)
        varsc = np.array(varsc)
        var1 = np.array(var1)
        var2 = np.array(var2)
        var3 = np.array(var3)
        var32 = np.array(var32)
        var4 = np.array(var4)
        var5 = np.array(var5)

        pc.append(varpc)
        sc.append(varsc)
        fc.append(var1)
        vxc.append(var2)
        vyc.append(var3)
        vzc.append(var32)
        foc_conn.append(var4)
        fwc_conn.append(var5)

    pc = np.array(pc)
    sc = np.array(sc)
    fc = np.array(fc)
    vxc = np.array(vxc)
    vyc = np.array(vyc)
    vzc = np.array(vzc)
    foc_conn = np.array(foc_conn)
    fwc_conn = np.array(fwc_conn)

    ######################################### connup ###############################################
    connup_l=Main.eval("""sim.connlist.l""") # run julia code using python
    connup_r=Main.eval("""sim.connlist.r""")

    ######################################### coarse block flux ###############################################
    vvxc = np.zeros(pc.shape) # vvxc is just diff from vxc since it is temporary used against the bug,volume flux not velocity
    vvyc = np.zeros(pc.shape) # [num_reals][num_TS][num_cells_coarse]
    vvzc = np.zeros(pc.shape)

    for real in range(num_reals):
        for time in range(pc[0].shape[0]):
            for i in range(connup_l.shape[0]):
                diff = connup_r[i] - connup_l[i]
                if(diff==1): # means it is in x direction
                    temp = foc_conn[real,time,i] + fwc_conn[real,time,i]
                    vvxc[real,time,connup_r[i]-1] += temp
                    vvxc[real,time,connup_l[i]-1] -= temp
                elif(diff==int(grids[0]/coarseLevels[0][lvl_chosen])): # means it is in y direction
                    temp = foc_conn[real,time,i] + fwc_conn[real,time,i]
                    vvyc[real,time,connup_r[i]-1] += temp
                    vvyc[real,time,connup_l[i]-1] -= temp
                else: # means it is in z direction
                    temp = foc_conn[real,time,i] + fwc_conn[real,time,i]
                    vvzc[real,time,connup_r[i]-1] += temp
                    vvzc[real,time,connup_l[i]-1] -= temp
################################################ calcualte NPV specifically #################################################
    upscaledData = upscaledData_original["level_"+str(lvl_chosen+1)]

    # Step 1. for chosen level only, get the npv for each well on the upscaled scale
    # Calculate upscaled NPV for each producer
    NPV_upscaled_prod = np.zeros((num_reals,num_prod))               
    for i in range(num_reals):
        rate_time_coarse = upscaledData[i]['time']
        rate_PRD_OPR_coarse = upscaledData[i]['prod_orat']
        rate_PRD_WPR_coarse = upscaledData[i]['prod_wrat']
        for j in range(num_prod):
            npv = 0
            for n in range(rate_time_coarse.shape[0]):
                val = 0
                if n == 0:
                    dt = rate_time_coarse[0]
                else:
                    dt = rate_time_coarse[n] - rate_time_coarse[n-1]
                val = dt*Po * rate_PRD_OPR_coarse[j,n] - dt*Pw * rate_PRD_WPR_coarse[j,n]
                npv += val/(1+r)**(rate_time_coarse[n]//365)           
            NPV_upscaled_prod[i,j] = int(npv)

    # Calculate upscaled NPV for each injector
    NPV_upscaled_inj = np.zeros((num_reals,num_inj))      
    for i in range(num_reals):
        rate_time_coarse = upscaledData[i]['time']
        rate_INJ_WIR_coarse = upscaledData[i]['inj_rate']
        for k in range(num_inj):
            npv = 0
            for n in range(rate_time_coarse.shape[0]):
                val = 0
                if n == 0:
                    dt = rate_time_coarse[0]
                else:
                    dt = rate_time_coarse[n] - rate_time_coarse[n-1]
                val = dt*Pinj * rate_INJ_WIR_coarse[k,n]
                npv += val/(1+r)**(rate_time_coarse[n]//365)           
            NPV_upscaled_inj[i,k] = int(npv)
################################################ preprocess feature and target #################################################
    # first, we will extract basic variables from fine, upscaling, upscaled simulation results
    tc = []
    PVI = [] # careful here, for current case we have water volume factor to be 1 so use rate injector directly!
    num_cell = grids[0]*grids[1]*grids[2]
    cell_size = gridSize[0]*gridSize[1]*gridSize[2]

    for i in range(num_reals):
        time = upscaledData[i]['time']
        tc.append(time)
        prod_wrat = upscaledData[i]['inj_rate']
        pore_volume = (cell_size*poro) * (num_cell)
        sum_rate = np.zeros(time.shape[0])
        for inj in range(num_inj):
            sum_rate += prod_wrat[inj]
        cumsum_rate = np.zeros(time.shape[0])
        for tic in range(time.shape[0]):
            cumsum_rate[tic] += sum_rate[tic]*time[tic]
        PVI.append(abs(cumsum_rate/pore_volume))
    tc = np.array(tc)
    PVI = np.array(PVI)

    # second, we will calculate some combinations based on literatures to generate some well-defined features for ML (x)
    cgrids_x = int(grids[0]/coarseLevels[0][lvl_chosen])
    cgrids_y = int(grids[1]/coarseLevels[1][lvl_chosen])
    cgrids_z = int(grids[2]/coarseLevels[2][lvl_chosen])
    ######################################### vx1p_bar/prime #############################################
    vx1p_bar = []
    vx1p_prime = []

    # step 1. compute v1p_bar the average at coarse scale
    for i in range(num_reals):
        temp1 = []
        for x in range(cgrids_x):
            for y in range(cgrids_y):
                for z in range(cgrids_z):
                    avg = 0
                    x_start = x*coarseLevels[0][lvl_chosen]
                    y_start = y*coarseLevels[1][lvl_chosen]
                    z_start = z*coarseLevels[2][lvl_chosen]
                    for j in range(coarseLevels[0][lvl_chosen]):
                        for k in range(coarseLevels[1][lvl_chosen]):
                            for l in range(coarseLevels[2][lvl_chosen]):
                                cell_idx = (j+x_start+1) + \
                                (k+y_start)*int(grids[0]/coarseLevels[0][lvl_chosen]) +\
                                (l+z_start)*int(grids[0]/coarseLevels[0][lvl_chosen])*\
                                int(grids[1]/coarseLevels[1][lvl_chosen])# 1-based
                                avg += vx1p[i][cell_idx-1]
                    temp1.append(avg/coarseLevels[0][lvl_chosen]/\
                                 coarseLevels[1][lvl_chosen]/coarseLevels[2][lvl_chosen])
        vx1p_bar.append(temp1)

    # step 2. get v1p_prime finished
    for i in range(num_reals):
        temp1 = []
        for x in range(grids[0]):
            for y in range(grids[1]):   
                for z in range(grids[2]):   
                    cell_idx = (x+1) + y*grids[0] + z*grids[0]*grids[1]
                    x_coarse = int(x/coarseLevels[0][lvl_chosen])
                    y_coarse = int(y/coarseLevels[1][lvl_chosen])
                    z_coarse = int(z/coarseLevels[2][lvl_chosen])
                    cell_idx_coarse = (x_coarse+1) + int(y_coarse*grids[0]/coarseLevels[0][lvl_chosen])+\
                                        int(z_coarse*grids[0]/coarseLevels[0][lvl_chosen]*grids[1]/coarseLevels[1][lvl_chosen])        
                    coarse_avg = vx1p_bar[i][cell_idx_coarse-1] 
                    temp1.append(vx1p[i][cell_idx-1] - coarse_avg)
        vx1p_prime.append(temp1)
    vx1p_prime = np.array(vx1p_prime)
    ######################################### vy1p_bar/prime #############################################
    vy1p_bar = []
    vy1p_prime = []

    # step 1. compute v1p_bar the average at coarse scale
    for i in range(num_reals):
        temp1 = []
        for x in range(cgrids_x):
            for y in range(cgrids_y):
                for z in range(cgrids_z):
                    avg = 0
                    x_start = x*coarseLevels[0][lvl_chosen]
                    y_start = y*coarseLevels[1][lvl_chosen]
                    z_start = z*coarseLevels[2][lvl_chosen]
                    for j in range(coarseLevels[0][lvl_chosen]):
                        for k in range(coarseLevels[1][lvl_chosen]):
                            for l in range(coarseLevels[2][lvl_chosen]):
                                cell_idx = (j+x_start+1) + \
                                (k+y_start)*int(grids[0]/coarseLevels[0][lvl_chosen]) +\
                                (l+z_start)*int(grids[0]/coarseLevels[0][lvl_chosen])*\
                                int(grids[1]/coarseLevels[1][lvl_chosen])# 1-based
                                avg += vy1p[i][cell_idx-1]
                    temp1.append(avg/coarseLevels[0][lvl_chosen]/\
                                 coarseLevels[1][lvl_chosen]/coarseLevels[2][lvl_chosen])
        vy1p_bar.append(temp1)

    # step 2. get v1p_prime finished
    for i in range(num_reals):
        temp1 = []
        for x in range(grids[0]):
            for y in range(grids[1]):   
                for z in range(grids[2]):   
                    cell_idx = (x+1) + y*grids[0] + z*grids[0]*grids[1]
                    x_coarse = int(x/coarseLevels[0][lvl_chosen])
                    y_coarse = int(y/coarseLevels[1][lvl_chosen])
                    z_coarse = int(z/coarseLevels[2][lvl_chosen])
                    cell_idx_coarse = (x_coarse+1) + int(y_coarse*grids[0]/coarseLevels[0][lvl_chosen])+\
                                        int(z_coarse*grids[0]/coarseLevels[0][lvl_chosen]*grids[1]/coarseLevels[1][lvl_chosen])        
                    coarse_avg = vy1p_bar[i][cell_idx_coarse-1] 
                    temp1.append(vy1p[i][cell_idx-1] - coarse_avg)
        vy1p_prime.append(temp1)
    vy1p_prime = np.array(vy1p_prime)
    ######################################### vz1p_bar/prime #############################################
    vz1p_bar = []
    vz1p_prime = []

    # step 1. compute v1p_bar the average at coarse scale
    for i in range(num_reals):
        temp1 = []
        for x in range(cgrids_x):
            for y in range(cgrids_y):
                for z in range(cgrids_z):
                    avg = 0
                    x_start = x*coarseLevels[0][lvl_chosen]
                    y_start = y*coarseLevels[1][lvl_chosen]
                    z_start = z*coarseLevels[2][lvl_chosen]
                    for j in range(coarseLevels[0][lvl_chosen]):
                        for k in range(coarseLevels[1][lvl_chosen]):
                            for l in range(coarseLevels[2][lvl_chosen]):
                                cell_idx = (j+x_start+1) + \
                                (k+y_start)*int(grids[0]/coarseLevels[0][lvl_chosen]) +\
                                (l+z_start)*int(grids[0]/coarseLevels[0][lvl_chosen])*\
                                int(grids[1]/coarseLevels[1][lvl_chosen])# 1-based
                                avg += vz1p[i][cell_idx-1]
                    temp1.append(avg/coarseLevels[0][lvl_chosen]/\
                                 coarseLevels[1][lvl_chosen]/coarseLevels[2][lvl_chosen])
        vz1p_bar.append(temp1)

    # step 2. get v1p_prime finished
    for i in range(num_reals):
        temp1 = []
        for x in range(grids[0]):
            for y in range(grids[1]):   
                for z in range(grids[2]):   
                    cell_idx = (x+1) + y*grids[0] + z*grids[0]*grids[1]
                    x_coarse = int(x/coarseLevels[0][lvl_chosen])
                    y_coarse = int(y/coarseLevels[1][lvl_chosen])
                    z_coarse = int(z/coarseLevels[2][lvl_chosen])
                    cell_idx_coarse = (x_coarse+1) + int(y_coarse*grids[0]/coarseLevels[0][lvl_chosen])+\
                                        int(z_coarse*grids[0]/coarseLevels[0][lvl_chosen]*grids[1]/coarseLevels[1][lvl_chosen])        
                    coarse_avg = vz1p_bar[i][cell_idx_coarse-1] 
                    temp1.append(vz1p[i][cell_idx-1] - coarse_avg)
        vz1p_prime.append(temp1)
    vz1p_prime = np.array(vz1p_prime)
    ################################################ error prediction #########################################################
    saved_scaler = load(open('error_model_scaler.pkl','rb'))
    saved_pca = load(open('error_model_pca.pkl','rb'))
    saved_model = load(open('error_model_model.pkl','rb'))

    ######################################### Trans X Y Z ################################################
    TransX = []
    for i in range(num_reals):
        TransX.append(TransX_all[i]["level_"+str(lvl+1)])
    TransX = np.array(TransX)
    TransY = []
    for i in range(num_reals):
        TransY.append(TransY_all[i]["level_"+str(lvl+1)])
    TransY = np.array(TransY)
    TransZ = []
    for i in range(num_reals):
        TransZ.append(TransZ_all[i]["level_"+str(lvl+1)])
    TransZ = np.array(TransZ)
    ######################################### WI #########################################################
    WI = []
    for i in range(num_reals):
        WI.append(WI_all[i]['level_'+str(lvl_chosen+1)])
    WI = np.array(WI)
    
    fileName = "PERM.DAT"
    size_x = grids[0]
    size_y = grids[1]
    size_z = grids[2]
    perm = np.zeros(size_x*size_y*size_z)
    with open(fileName) as f:
        lines = f.readlines()
    for i in range(len(lines)-2):
        perm[i] = float(lines[i+1])
    perm = perm.reshape((size_x,size_y,size_z),order='F')

    # scale variables that won't do SVD
    scaler =saved_scaler['v1p_x']
    vx1p_scaled = scaler.transform(vx1p) # scale the feature
    
    scaler =saved_scaler['v1p_y']
    vy1p_scaled = scaler.transform(vy1p) # scale the feature
    
    scaler =saved_scaler['v1p_z']
    vz1p_scaled = scaler.transform(vz1p) # scale the feature
    
    scaler =saved_scaler['v1p`_x']
    vx1p_prime_scaled = scaler.transform(vx1p_prime) # scale the feature
    
    scaler =saved_scaler['v1p`_y']
    vy1p_prime_scaled = scaler.transform(vy1p_prime) # scale the feature
    
    scaler =saved_scaler['v1p`_z']
    vz1p_prime_scaled = scaler.transform(vz1p_prime) # scale the feature
    
    scaler =saved_scaler['perm']
    perm_2d = perm.reshape(perm.shape[0]*perm.shape[1]*perm.shape[2]).reshape(-1,1)
    perm_scaled = scaler.transform(perm_2d)
    perm_1d = perm_scaled.reshape(perm.shape[0]*perm.shape[1]*perm.shape[2])
    perm_stack = []
    for i in range(num_reals):
        perm_stack.append(perm_1d)
    perm_stack = np.array(perm_stack)
    
    scaler =saved_scaler['TRANSX']
    TransX_scaled = scaler.transform(TransX)
    
    scaler =saved_scaler['TRANSY']
    TransY_scaled = scaler.transform(TransY)
    
    scaler =saved_scaler['TRANSZ']
    TransZ_scaled = scaler.transform(TransZ)

    # Step 1. build up dataX, dataY, test_num, train_num for basic setup
    scale_factor = 100
    ################# PREPARE DATAX ###################################
    dataX = []
    data = {'v1p_x':[vx1p_scaled],'v1p_y':[vy1p_scaled],'v1p_z':[vz1p_scaled],'v1p`_x':[vx1p_prime_scaled],\
            'v1p`_y':[vy1p_prime_scaled],'v1p`_z':[vz1p_prime_scaled],'perm':[perm_stack]}
    dataX_1 = pd.DataFrame(data) # type 1 is fine scale data
    dataX.append(dataX_1)
    data = {'vc_x':[vvxc],'vc_y':[vvyc],'vc_z':[vvzc],'Sc':[sc],'Pc':[pc]}
    dataX_2 = pd.DataFrame(data) # type 2 is coarse scale data with spatial and temporal info
    dataX.append(dataX_2)
    data = {'PVI':[PVI]}#{'tc':[tc],'PVI':[PVI]}
    dataX_3 = pd.DataFrame(data) # type 3 is coarse scale data with temporal info
    dataX.append(dataX_3)
    data = {'TRANSX':[TransX_scaled],'TRANSY':[TransY_scaled],'TRANSZ':[TransZ_scaled]} 
    dataX_4 = pd.DataFrame(data) # type 4 is coarse scale data with spatial info
    dataX.append(dataX_4)
    data = {'WI':[WI]} # didn't scale
    dataX_5 = pd.DataFrame(data) # type 5 is coarse scale data for each well
    dataX.append(dataX_5)
 
    features = svd_preprocess_feature(dataX, True, num_inj, num_prod, wellLoc_upscale["level_"+str(lvl_chosen+1)], 
                                    coarseLevels[0][lvl_chosen], coarseLevels[1][lvl_chosen], coarseLevels[2][lvl_chosen], grids,
                                      num_reals,scale_factor,saved_scaler,saved_pca)
    NPV_upscaled_test = NPV_upscaled[lvl_chosen]
    models = saved_model['lgbm']
    y_pred_lgb = predict(features, models, num_well)
    models = saved_model['rf']
    y_pred_rf = predict(features, models, num_well)

    y_pred_lgb_all = np.sum(y_pred_lgb,axis=0)
    y_pred_rf_all = np.sum(y_pred_rf,axis=0)            
    ################################################ update feval UoF #########################################################
    output = NPV_upscaled[0]+y_pred_lgb_all/2+y_pred_rf_all/2
    np.savetxt(outfile,output)

if __name__ == '__main__':
    main(sys.argv[1:])
