% This program implement the OMP 
% Input:
%     dictionary: B; signal: x; error threshold: e; max number of iteration: kmax
% Output:
%     weight: w
% using pseudoinverse to handle all kinds of singularity problem when projecting
% signal on the selected dictionary

function [w]=omp(B,x,e0,k0)

% initialization
n=size(B,1);
m=size(B,2);
w=zeros(m,n);
r=x;
e=norm(r,2);
B1=zeros(size(B));
SS=[];
for k=1:1:n 
  [value idc] =max(abs(B'*r));
  SS=sort([SS,idc]);
  w(:,k)=zeros(m,1);
  w(SS,k)= pinv(B(:,SS))*x;
  r = x-B(:,SS)*w(SS,k);
  e=norm(r,2);
%  k=k+1;
end

