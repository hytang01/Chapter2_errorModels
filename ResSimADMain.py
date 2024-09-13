import julia
from julia import Main
from julia import ResSimAD

def ResSimADMain(dataPath,grids,gridSize,poro,wellLoc,num_inj,inj_bhp,num_prod,prod_bhp,
                 initial_pres,total_time,intial_wat_sat,well_raidus,perf_z1,perf_z2,dt0):
    # Unit transformation
    bar2psi = 14.50377
    cm2bbl = 6.289811
    cm2cft = 35.31466
    m2ft = 3.2808
    β = cm2bbl / bar2psi
    
    options = {}
    # Grid and Rock
    x=0;y=1;z=2;
    options["nx"] = grids[x]; options["ny"] = grids[y]; options["nz"] = grids[z];
    options["dx"] = gridSize[x]*m2ft; options["dy"] = gridSize[y]*m2ft; options["dz"] = gridSize[z]*m2ft;
#     options["v"] = 2858291.712 * cm2cft
    options["tops"] = 6080.*m2ft#
    options["perm"] = dataPath+'PERM.DAT';
    options["multpermz"] = 0.1;
    options["poro"] = poro;#dataPath+r'\poro.DAT';#

    # Fluid
    options["fluid"] = "OW"
    options["po"] = initial_pres*bar2psi; options["sw"] = intial_wat_sat;
    options["PVDO"] = dataPath+'PVDO.DAT';#ResSimAD.get_example_data("PVDO.DAT");
    options["PVTW"] = dataPath+'PVTW.DAT';#ResSimAD.get_example_data("PVTW.DAT");#
    options["SWOF"] = dataPath+'SWOF.DAT'#ResSimAD.get_example_data("SWOF.DAT");

    # Wells
    # injectors
    options["injectors"] = []
    
    for ind in range(num_inj):
        i = {};
        x=int(wellLoc[0][ind][0])
        y=int(wellLoc[0][ind][1])
        z1=int(perf_z1)
        z2=int(perf_z2)
        i["name"] = "I"+str(ind+1); 
#         for i in range(z2-z1):
        i["perforation"] = [(x,y,z) for z in range(z1,z2+1)];
        i["radius"] = well_raidus*m2ft;
        i["mode"] = "bhp"; i["target"] = inj_bhp*bar2psi;
        options["injectors"].append(i);
        
    #producers
    options["producers"] = []
    for ind in range(num_prod):
        p = {};
        x=int(wellLoc[1][ind][0])
        y=int(wellLoc[1][ind][1])
        z1=int(perf_z1)
        z2=int(perf_z2)
        p["name"] = "P"+str(ind+1); p["perforation"] = [(x,y,z) for z in range(z1,z2+1)]; 
        p["radius"] =well_raidus*m2ft;
        p["mode"] = "bhp"; p["target"] = prod_bhp*bar2psi;
        options["producers"].append(p);
    

    # Nonlinear solver options
    options["max_newton_iter"] = 40
    options["min_err"] = 1.0e-5

    # Schedule
    options["dt0"] = dt0;# 0.01; #2-5
    options["dt_max"] = 100.;#100.;#
    options["t_end"] = total_time;
#     options["min_err"] = 1.0e-9;

    # # solver
    # options["linear_solver"]="BICGSTAB_ILU_DUNE_ISTL"
    Main.sim = ResSimAD.Sim(options)
    
    return options, Main.sim