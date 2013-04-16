% run omp on mnist data
% get 5000 balanced sample from training data, sample without replacment
% get 60 balanced sample from testing data, sample without replacment
% by Cameron P.H. Chen @ Princeton

clear all

trainSampleNum = 5000;
testSampleNum = 500;

train_images = loadMNISTImages('/Users/ChimatChen/Dropbox/Research/OMPvsScreening/data/train-images-idx3-ubyte');
train_labels = loadMNISTLabels('/Users/ChimatChen/Dropbox/Research/OMPvsScreening/data/train-labels-idx1-ubyte');
test_images = loadMNISTImages('/Users/ChimatChen/Dropbox/Research/OMPvsScreening/data/t10k-images-idx3-ubyte');
test_labels = loadMNISTLabels('/Users/ChimatChen/Dropbox/Research/OMPvsScreening/data/t10k-labels-idx1-ubyte');

test_images2 = test_images (:,5001:5100);
test_labels2 = test_labels (5001:5100,:);

test_images= test_images(:,1:5000);
test_labels = test_labels (1:5000,:);


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

L2error1=nan(testSampleNum,1);
L2error2=nan(testSampleNum,1);

% check lambda value
u=-1:0.1:1;
lambda_val = 10.^u;
%lambda_val = [0.001 0.005 ];
L2err_lambda = nan(size(lambda_val,2),1);



kmax=50;

w_idt = nan(trainSampleNum,testSampleNum,size(lambda_val,2));
w_omp = nan(trainSampleNum,testSampleNum);


%IDT
fprintf('----------IDT----------\n');
l=0;
for lambda = lambda_val 
  fprintf('lambda:%f\n',lambda);
  l=l+1;
  for i=1:testSampleNum
    
    fprintf('i:%d\n',i);

    [w1 k]=IDT(train_image_sample,test_image_sample(:,i),lambda);
    %0.05-> 5% of the error rate between reconstructed signal and the orignal signal
    w1=omp(train_image_sample,test_image_sample(:,i),0,kmax-k,find(w1~=0)'); 
    L2error1(i,:)=norm(train_image_sample*w1-test_image_sample(:,i),2)/norm(test_image_sample(:,i),2);
   % w_idt(l,ceil(i/10),mod(i,testSampleNum/10)+1,:) = w1; 
    w_idt(:,i,l) = w1; 
 

  end
  L2err_lambda(l) = mean(L2error1);
end

%OMP
fprintf('----------OMP----------\n');
for i=1:testSampleNum
    fprintf('i:%d\n',i);

    %%check the stopping criteria of OMP
    % choose the same amount of basis as IDT, to be modified
    w2=omp(train_image_sample,test_image_sample(:,i),0,kmax,[]);
    L2error2(i,:)=norm(train_image_sample*w2-test_image_sample(:,i),2)/norm(test_image_sample(:,i),2);
    %w_omp(ceil(i/10),mod(i,testSampleNum/10)+1,:) = w2;
    w_omp(:,i) = w2;
end

%calculate the mean of weights
w_omp_classif = nan(trainSampleNum,10);
w_idt_classif = nan(trainSampleNum,10,size(lambda_val,2));
for i=1:10
  w_omp_classif(:,i)=mean(w_omp(:,find(test_labels_sample==(i-1))),2);
end

l=0;
for lambda = lambda_val
  l=l+1;
  for i=1:10
    w_idt_classif(:,i,l)=mean(w_idt(:,find(test_labels_sample==(i-1)),l),2);
  end
end

%generate the predict label

omp_output_label = nan(size(test_images2,2),1);
for i=1:size(test_images2,2)
  distance=nan(10,1);
  for j=1:10
    distance(j,1) = norm(test_images2(:,i)-train_image_sample*w_omp_classif(:,j),2);
  end
  assert(sum(isnan(distance))==0,'distance nan');
  [C,I] = min(distance);
  omp_output_label(i,1)=I; 
end


idt_output_label = nan(size(test_images2,2),size(lambda_val,2));
l=0;
for lambda = lambda_val
  l=l+1;
  for i=1:size(test_images2,2)
    distance=nan(10,1);
    for j=1:10
      distance(j,1) = norm(test_images2(:,i)-train_image_sample*w_idt_classif(:,j,l),2);
    end
    assert(sum(isnan(distance))==0,'distance nan');
    [C,I] = min(distance);
    idt_output_label(i,l)=I; 
  end
end

% calculate the error rate
omp_class_err=sum((omp_output_label-1)~=test_labels2)/size(test_images2,2);
l=0;
idt_class_err = nan(length(lambda_val),1);
for lambda = lambda_val
  l=l+1;
   idt_class_err(l,:)=sum((idt_output_label(:,l)-1)~=test_labels2)/size(test_images2,2);
end

clf;

% standard plot
%{
figure
hold on;
plot(lambda_val, L2err_lambda, 'b-')
plot(lambda_val, mean(L2error2)*ones(size(lambda_val,2),1), 'r-')
xlabel('lambda')
ylabel('error (%)')
legend('IDT','OMP')
title('standard')
hold off
%}
%saveas(gcf, 'Plot_Standard', 'tiff') 

% standard plot with log scale x axis
%{
figure
hold on;
plot(log(lambda_val), L2err_lambda, 'b-')
plot(log(lambda_val), mean(L2error2)*ones(size(lambda_val,2),1), 'r-')
xlabel('lambda')
ylabel('error (%)')
legend('IDT','OMP')
title('recovery error logscale')
hold off
%}
%saveas(gcf, 'Plot_LogScale', 'tiff') 


% semilog plot on x scale
%{
figure
hold on
semilogx(lambda_val, L2err_lambda, 'b-')
semilogx(lambda_val, mean(L2error2)*ones(size(lambda_val,2),1), 'r-')
xlabel('lambda')
ylabel('error (%)')
legend('IDT','OMP')
title('semilogx')
hold off
%}
%saveas(gcf, 'Plot_SemilogPlot', 'tiff')


% classification error
figure
hold on;
plot(log(lambda_val), idt_class_err, 'b-')
plot(log(lambda_val), omp_class_err*ones(size(lambda_val,2),1), 'r-')
xlabel('lambda')
ylabel('error (%)')
legend('IDT','OMP')
title('classification error logscale')
hold off
%saveas(gcf, 'classification_Plot_LogScale', 'tiff')



