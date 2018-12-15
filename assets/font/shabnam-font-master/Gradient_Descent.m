close all
clear
clc

load regression_dataset.csv

X = regression_dataset(: , 1:2 ) ;
X = [ones( size(X,1) , 1 ) , X] ;
Y = regression_dataset(: , 3) ;
W = 1*rand( size(X , 2) , 1) ;

Theta = [3.9933; 7.2963; 3.0936];
IterMax = 100 ;
alpha = .1 ;

% W = Theta ;
W2 = W;

% First Method
for i = 1: IterMax
    Err(i) = ErrorFunc(W,X,Y) ;
    Grad = GradFunc(W,X,Y) ;
    W = W - alpha * Grad;
%     pause(1)
%     plot(Err)
end

% Second Method

for i = 1: IterMax
        Err2(i) = ErrorFunc(W2,X,Y) ;
    for j =1 : length(W2)
        Grad = sum( ( X(:,j) * W2(j) - Y )' * X(:,j) )/(size(X,1)) ;
        W2(j) = W2(j) - alpha * Grad;
    end
    
end

figure
ii = 1:IterMax;
plot(ii,Err,'b',ii,Err2,'r')

W3 = inv(X'*X)*X'*Y;

