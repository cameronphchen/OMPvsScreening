% This program implement the basis selection method based on iterative
% dome test.

% by Cameron P.H. Chen @ Princeton

function [w,k]=IDT(B,x,lambda)

n=size(B,1);
m=size(B,2);

w=zeros(m,1);

% normalized the dictionary
dim=size(B,1);
x = x./sqrt(sum(x.^2,1));
B = B./(ones(dim,1)*sqrt(sum(B.^2,1)));

B_s = [];
SS = [];
lambda_max = max(B'*x);

q_t0 = x/lambda; 
r_t0 = norm(x,2)*(1/lambda_max - 1/lambda);
[C,I] = max(B'*q_t0)
bh_t0 = B(:,I); 
SS=sort([SS,I]);
B_s = [ B_s bh_t0 ]
psi_t0 = (q_t0'*bh_t0-1)/r_t0
while true
  
  fprintf('iteraion');
  q_t1 = q_t0-psi_t0*r_t0*bh_t0; 
  
  
  r_t1 = r_t0*sqrt(1-psi_t0^2); 
  [C,I] = max(B'*q_t1);
  SS=sort([SS,I]);
  bh_t1 = B(:,I); 
  B_s = [ B_s bh_t1 ]
  psi_t1 = (q_t1'*bh_t1-1)/r_t1;

  q_t0 = q_t1; 
  r_t0 = r_t1;
  psi_t0 = psi_t1

  if (psi_t0 > 1)| (psi_t0<=0) break; end;
end

w(SS)= pinv(B(:,SS))*x;
k = length(SS);
