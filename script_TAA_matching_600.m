% Register cylinder to first 600 samples (sample_00000 through sample_00599)
clear all
restoredefaultpath
addpath(genpath('/home/shiyi/fshapesTk/Bin'))

VTK_DIR   = './vtk';
MATCH_DIR = './matchings';

r = 1.0;
source = import_fshape_vtk(fullfile(VTK_DIR, 'cylinder.vtk'));
source.x = source.x * r;

comp_method = 'cuda';
defo.kernel_size_mom = [0.3, 0.2];
defo.nb_euler_steps  = 15;
defo.method          = comp_method;
objfun.distance                           = 'kernel';
objfun.kernel_distance.distance           = 'var';
objfun.kernel_distance.kernel_size_geom   = 0.3;
objfun.kernel_distance.kernel_size_signal = 1.8;
objfun.kernel_distance.method             = comp_method;
objfun.pen_signal         = 'h2';
objfun.weight_coef_dist   = 3000;
objfun.weight_coef_pen_fr = 0.03;
objfun.weight_coef_pen_f  = 0;
optim.method     = 'bfgs';
optim.bfgs.maxit = 30;

for i = 0:599
    sample_name = sprintf('sample_%05d', i);
    vtk_file    = fullfile(VTK_DIR, [sample_name, '.vtk']);
    saveDir     = fullfile(MATCH_DIR, sample_name);

    if ~exist(vtk_file, 'file')
        fprintf('[%d/600] SKIP %s (no vtk file)\n', i+1, sample_name);
        continue
    end

    fprintf('[%d/600] %s\n', i+1, sample_name);
    target   = import_fshape_vtk(vtk_file);
    target.x = target.x * r;

    [momentums, summary] = match_geom(source, target, defo, objfun, optim);
    export_matching_tan(source, momentums, zeros(size(source.f)), target, summary, saveDir);
end

fprintf('\nDone: 600 samples complete.\n');
