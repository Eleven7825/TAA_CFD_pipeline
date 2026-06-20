function lddmm_register_single(source_vtk, target_vtk, output_dir)
% Single LDDMM registration: register source (cylinder) to target (current geometry).
%
% Called from Python:
%   matlab -sd /home/shiyi/TAA_CFD_pipeline \
%          -batch "lddmm_register_single('source.vtk', 'target.vtk', 'out_dir')"
%
% Outputs 1-shoot-1.vtk ... 1-shoot-16.vtk to output_dir.
% The final frame (1-shoot-16.vtk) is the registered source geometry.

restoredefaultpath
addpath(genpath('/home/shiyi/fshapesTk/Bin'))

fprintf('source : %s\n', source_vtk);
fprintf('target : %s\n', target_vtk);
fprintf('output : %s\n', output_dir);

r = 1.0;

source = import_fshape_vtk(source_vtk);
source.x = source.x * r;

target = import_fshape_vtk(target_vtk);
target.x = target.x * r;

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

[momentums, summary] = match_geom(source, target, defo, objfun, optim);

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

export_matching_tan(source, momentums, zeros(size(source.f)), target, summary, output_dir);

fprintf('Registration complete. Shoot frames saved to: %s\n', output_dir);
end
