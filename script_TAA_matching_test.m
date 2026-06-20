% Single-sample test: cylinder -> sample_00001
addpath(genpath('/home/shiyi/fshapesTk/Bin'));

r = 1.0;
source = import_fshape_vtk('./vtk/cylinder.vtk');
source.x = source.x * r;
target = import_fshape_vtk('./vtk/sample_00001.vtk');
target.x = target.x * r;

comp_method = 'cuda';
defo.kernel_size_mom = [0.3, 0.2];
defo.nb_euler_steps  = 15;
defo.method          = comp_method;

objfun.distance                         = 'kernel';
objfun.kernel_distance.distance         = 'var';
objfun.kernel_distance.kernel_size_geom = 0.3;
objfun.kernel_distance.kernel_size_signal = 1.8;
objfun.kernel_distance.method           = comp_method;
objfun.pen_signal         = 'h2';
objfun.weight_coef_dist   = 3000;
objfun.weight_coef_pen_fr = 0.03;
objfun.weight_coef_pen_f  = 0;

optim.method     = 'bfgs';
optim.bfgs.maxit = 30;

[momentums, summary] = match_geom(source, target, defo, objfun, optim);
mkdir('./matchings');
saveDir = './matchings/sample_00001';
export_matching_tan(source, momentums, zeros(size(source.f)), target, summary, saveDir);
disp('Done: sample_00001 exported');
