% run omp on mnist data
% get 5000 balanced sample from training data, sample without replacment
% get 60 balanced sample from testing data, sample without replacment
% by Cameron P.H. Chen @ Princeton

clear all

trainSampleNum = 5000;
testSampleNum = 60;

train_images = loadMNISTImages('/Users/ChimatChen/Dropbox/Research/OMPvsScreening/data/train-images-idx3-ubyte');
train_labels = loadMNISTLabels('/Users/ChimatChen/Dropbox/Research/OMPvsScreening/data/train-labels-idx1-ubyte');
test_images = loadMNISTImages('/Users/ChimatChen/Dropbox/Research/OMPvsScreening/data/t10k-images-idx3-ubyte');
test_labels = loadMNISTLabels('/Users/ChimatChen/Dropbox/Research/OMPvsScreening/data/t10k-labels-idx1-ubyte');

% balanced sampling from training and testing dataset

train_image_sample = nan(size(train_images,1),trainSampleNum);
test_image_sample = nan(size(test_images,1),testSampleNum);
train_labels_sample = nan(trainSampleNum,1);
test_labels_sample = nan(testSampleNum,1);

test_distri = nan(10,1);
train_distri = nan(10,1);


for i=1:10
  test_distri(i,:) = sum(test_labels==(i-1));
  train_distri(i,:) = sum(train_labels==(i-1));
end

for i=1:10
  tmp_test_idc=find(test_labels==i-1);
  tmp_train_idc=find(train_labels==i-1);

  test_image_sample(:,(i-1)*(testSampleNum/10)+1:i*(testSampleNum/10))=...
        test_images(:,tmp_test_idc(randsample(test_distri(i,:),testSampleNum/10)));

  train_image_sample(:,(i-1)*(trainSampleNum/10)+1:i*(trainSampleNum/10))=...
        train_images(:,tmp_train_idc(randsample(train_distri(i,:),trainSampleNum/10)));

  test_labels_sample((i-1)*(testSampleNum/10)+1:i*(testSampleNum/10))=...
        test_labels(tmp_test_idc(randsample(test_distri(i,:),testSampleNum/10)));

  train_labels_sample((i-1)*(trainSampleNum/10)+1:i*(trainSampleNum/10))=...
        train_labels(tmp_train_idc(randsample(train_distri(i,:),trainSampleNum/10)));
end 

L2error=nan(60,1);

for i=1:60
  fprintf('i:%d\n',i);
  %%check the stopping criteria of OMP
  w=omp(train_image_sample,test_image_sample(:,i),0,0,30);
  L2error(i,:)=norm(train_image_sample*w-test_image_sample(:,i),2)/norm(test_image_sample(:,i),2);
end

