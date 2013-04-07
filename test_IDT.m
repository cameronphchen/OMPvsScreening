% unit test for IDT
% by Cameron P.H. Chen @ Princeton

%simple toy exmaple
B = [1 0 0 0;
     0 1 0 0.707;
     0 0 1 0.707;]

x = [1 2 3]';

lambda = 0.00001;
[w k] = IDT(B,x,lambda)
assert(norm(w-[0.2673 0 0 0.9449]',2)<0.0001);



%singular B matrix
B = [0 0 0 0;
     0 0 0 0.707;
     0 0 1 0.707;]

x = [0 2 3]';

lambda = 0.00001;
[w k] = IDT(B,x,lambda)
assert(norm(w-[0 0 0.2774 0.7845]',2)<0.0001);


fprintf('Pass IDT unit test\n')
