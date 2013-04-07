% unit test for omp
% by Cameron P.H. Chen @ Princeton


%simple toy exmaple
B = [1 0 0 0;
     0 1 0 0.707;
     0 0 1 0.707;]

x = [1 2 3]';

w=omp(B,x,0,0,size(B,1))
assert(norm(w-[1 0 1 2.8289]',2)<0.0001);



%singular B matrix
B = [0 0 0 0;
     0 0 0 0.707;
     0 0 1 0.707;]

x = [0 2 3]';

w=omp(B,x,0,0,size(B,1))

assert(norm(w-[0 0 1 1.4144]',2)<0.0001);




fprintf('Pass omp unit test\n')
