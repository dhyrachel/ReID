clear all
clc
%M = rand(5, 5);
% M = rand(50, 50);
%M = rand(500, 500);
M = rand(5000, 5000);
 
tt1 = 0;
for i = 1:1000
    tic
    N = M .* M;
    t1 = toc;
    tt1 = tt1 + t1;
end
tt1
 
M = gpuArray(single(M));
tt2 = 0;
for i = 1:1000
    tic
    N1 = M .* M;
    t2 = toc;
    tt2 = tt2 + t2;
end
tt2