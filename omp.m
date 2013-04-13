% This program implement the OMP
%
% There are two mode for using this functino,
% the first mode is set e0=0, iter ~= 0, which iterates until error less then a 
% threshold
% The second mode is set e0~=0, iter ==0, which iterates iter rounds
%
% Input:
%     dictionary: B; signal: x; error threshold: e0; max number of iteration: k0
% Output:
%     weight: w
% using pseudoinverse to handle all kinds of singularity problem when projecting
% signal on the selected dictionary
% by Cameron P.H. Chen @ Princeton

function [w]=omp(B,x,e0,iter)

assert(size(B,1)==size(x,1), 'dimension of B and x must agree')

% initialization
n=size(B,1);
m=size(B,2);
w=zeros(m,1);
r=x;
e=norm(r,2);
B1=zeros(size(B));
SS=[];

if iter~=0 & e0==0
  for k=1:1:iter
    [value idc] =max(abs(B'*r));
    SS=sort([SS,idc]);
    w=zeros(m,1);
    w(SS)= pinv(B(:,SS))*x;
    r = x-B(:,SS)*w(SS);
    e=norm(r,2);
  end
elseif iter==0 & e0~=0
  while true
    [value idc] =max(abs(B'*r));
    SS=sort([SS,idc]);
    w=zeros(m,1);
    w(SS)= pinv(B(:,SS))*x;
    r = x-B(:,SS)*w(SS);
    e=norm(r,2);
    if e <= e0 break; end;
  end
end

