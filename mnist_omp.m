% run omp on mnist data
% by Cameron P.H. Chen @ Princeton


images = loadMNISTImages('/Users/ChimatChen/Dropbox/Research/OMPvsScreening/data/train-images-idx3-ubyte');
labels = loadMNISTLabels('/Users/ChimatChen/Dropbox/Research/OMPvsScreening/data/train-labels-idx1-ubyte');
images = loadMNISTImages('/Users/ChimatChen/Dropbox/Research/OMPvsScreening/data/t10k-images-idx1-ubyte');
labels = loadMNISTLabels('/Users/ChimatChen/Dropbox/Research/OMPvsScreening/data/t10k-labels-idx1-ubyte');

w_omp = omp(B,x,0,0)


