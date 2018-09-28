addpath F:\MATLAB\matconvnet\matlab
%enable GPU 
vl_compilenn('enableGpu', true,...
    'cudaRoot', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0',...
    'cudaMethod','nvcc','enableCudnn',true,...
    'cudnnRoot','F:\MATLAB\matconvnet\local\cudnn-7');
%enable matconvnet
fprintf('enable matconvnet...');
vl_setupnn;
fprintf('matconvnet setup success!');
