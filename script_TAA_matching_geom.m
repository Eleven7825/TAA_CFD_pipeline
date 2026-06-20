% Geometric matching: register base cylinder to each TAA sample (bulged interface)
% using LDDMM with varifold data attachment (fshapesTk toolkit).
%
% Prerequisites:
%   1. Run convert_interfaces_to_vtk.py to generate vtk/ directory
%   2. CUDA MEX binaries must be compiled (see fshapesTk/Bin/kernels/cuda/)

clear all
restoredefaultpath

addpath(genpath('/home/sc3544/fshapesTk/Bin'))

SCRIPT_DIR  = fileparts(mfilename('fullpath'));
VTK_DIR     = fullfile(SCRIPT_DIR, 'vtk');
MATCH_DIR   = fullfile(SCRIPT_DIR, 'matchings');

%------------%
%  template  %
%------------%

r = 1.0; % geometry already in cm; adjust if needed

source = import_fshape_vtk(fullfile(VTK_DIR, 'cylinder.vtk'));
source.x = source.x * r;

%------------%
% parameters %
%------------%

comp_method = 'cuda'; % 'cuda' or 'matlab'

% Deformation kernels — tune to geometry scale (~0.647 cm radius, 15 cm height)
defo.kernel_size_mom = [0.3, 0.2];
defo.nb_euler_steps  = 15;
defo.method          = comp_method;

% Varifold data attachment
objfun.distance                       = 'kernel';
objfun.kernel_distance.distance       = 'var';
objfun.kernel_distance.kernel_size_geom   = 0.3;
objfun.kernel_distance.kernel_size_signal = 1.8;
objfun.kernel_distance.method         = comp_method;

objfun.pen_signal          = 'h2';
objfun.weight_coef_dist    = 3000;
objfun.weight_coef_pen_fr  = 0.03;
objfun.weight_coef_pen_f   = 0;

% BFGS optimiser
optim.method     = 'bfgs';
optim.bfgs.maxit = 30;

%----------%
% matching %
%----------%

sample_dirs = dir(fullfile(VTK_DIR, 'sample_*.vtk'));
n_samples   = length(sample_dirs);

% job array slicing: each SLURM task processes one chunk of samples
CHUNK = 100;
task_id = str2double(getenv('SLURM_ARRAY_TASK_ID'));
if isnan(task_id)
    k_start = 1;
    k_end   = n_samples;
else
    k_start = task_id * CHUNK + 1;
    k_end   = min(k_start + CHUNK - 1, n_samples);
end
fprintf('Processing samples %d to %d\n', k_start, k_end);

parpool('local', 2);

parfor k = k_start:k_end

    vtk_file    = fullfile(VTK_DIR, sample_dirs(k).name);
    sample_name = sample_dirs(k).name(1:end-4);

    saveDir = fullfile(MATCH_DIR, sample_name);
    if exist(fullfile(saveDir, 'summary.txt'), 'file')
        fprintf('\n[%d/%d] Skipping %s (already done)\n', k, n_samples, sample_name);
        continue
    end

    fprintf('\n[%d/%d] Matching %s\n', k, n_samples, sample_name);

    target   = import_fshape_vtk(vtk_file);
    target.x = target.x * r;

    [momentums, summary] = match_geom(source, target, defo, objfun, optim);

    export_matching_tan(source, momentums, zeros(size(source.f)), target, summary, saveDir);

end

delete(gcp('nocreate'));
fprintf('\nAll matchings complete.\n');
