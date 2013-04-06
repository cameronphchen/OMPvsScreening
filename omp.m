% This program implement the OMP 
% Input:
%     dictionary: B; signal: x; error threshold: e; max number of iteration: kmax
% Output:
%     weight: w
% using pseudoinverse to handle all kinds of singularity problem when projecting
% signal on the selected dictionary
% by Cameron P.H. Chen @ Princeton

function [w]=omp(B,x,e0,k0,iter)

assert(size(B,1)==size(x,1), 'dimension of B and x must agree')

% initialization
n=size(B,1);
m=size(B,2);
w=zeros(m,1);
r=x;
e=norm(r,2);
B1=zeros(size(B));
SS=[];
for k=1:1:iter
  [value idc] =max(abs(B'*r));
  SS=sort([SS,idc]);
  w=zeros(m,1);
  w(SS)= pinv(B(:,SS))*x;
  r = x-B(:,SS)*w(SS);
  e=norm(r,2);
end

