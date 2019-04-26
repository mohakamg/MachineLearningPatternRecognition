close all;
clc, clear;
m=[zeros(5,1) ones(5,1)]; %% mean (:,1) class 1, (:,2) class 2,
S(:,:,1)=[ 0.8   0.2   0.1  0.05  0.01;
           0.2   0.7   0.1  0.03  0.02;
           0.1   0.1   0.8  0.02  0.01;
          0.05  0.03  0.02   0.9  0.01;
          0.01  0.02  0.01  0.01   0.8 ];

S(:,:,2)=[  0.9   0.1  0.05  0.02  0.01;
            0.1   0.8   0.1  0.02  0.02;
           0.05   0.1   0.7  0.02  0.01;
           0.02  0.02  0.02   0.6  0.02;
           0.01  0.02  0.01  0.02   0.7 ];

P=[1/2 1/2]';
%% generating samples
    rng('default');
    %train 1
    rng(0);
    N_1 = 100;
    half_N_1 = N_1/2;
    y_1 = zeros(N_1, 1);
    y_1( 1:half_N_1, 1 ) = 1; % class 1
    X_1 = genTwoClassNormal(m, S, y_1);

    %rng(0);
    N_12 = 1000;
    half_N_12 = N_12/2;
    y_12 = zeros(N_12, 1);
    y_12( 1:half_N_12, 1 ) = 1; % class 1
    X_12 = genTwoClassNormal(m, S, y_12);

    %test
    rng(100);
    N_2 = 10000;
    half_N_2 = N_2/2;
    y_2 = zeros(N_2, 1);
    y_2(1:half_N_2, 1) = 1; % class 1
    X_2 = genTwoClassNormal(m, S,y_2);

%% NAIVE BAYES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %labels
    y_test = [ones(half_N_2,1); %class 1 == 1
             zeros(half_N_2,1)];%class 2 == 0

    %TRAIN (X_1): mean & variance estimation
    %class 1
    [S_est(:,1), m_est(:,1)] =...
        muVarNaiveBayesGaussEstimates(X_1(1:half_N_1,:));
    [S_est2(:,1), m_est2(:,1)] =... % FOR 1000 train samples
        muVarNaiveBayesGaussEstimates(X_12(1:half_N_12,:));
    %class 2
    [S_est(:,2), m_est(:,2)] =...
        muVarNaiveBayesGaussEstimates(X_1((half_N_1+1):N_1,:));
    [S_est2(:,2), m_est2(:,2)] =... % FOR 1000 train samples
        muVarNaiveBayesGaussEstimates(X_12((1+half_N_12):N_12,:));

    %init prob.
    Naive_bayes_C1_test = ones(N_2,1);
    Naive_bayes_C2_test = ones(N_2,1);
    Naive_bayes_C1_test2 = ones(N_2,1);
    Naive_bayes_C2_test2 = ones(N_2,1);

    p_of_x_i_given_C1_test = zeros(N_2,5);
    p_of_x_i_given_C1_test2 = zeros(N_2,5);
    
    p_of_x_i_given_C2_test = zeros(N_2,5);
    p_of_x_i_given_C2_test2 =zeros(N_2,5);
    for i=1:5
      %% class 1
        % P(x_i | C1 ) 100 samples
        p_of_x_i_given_C1_test(:,i) = normpdf(X_2(:,i),...
            m_est(i,1), sqrt(S_est(i,1)) );
        Naive_bayes_C1_test =...
            Naive_bayes_C1_test.*p_of_x_i_given_C1_test(:,i);
        % P(x_i | C1 ) 1000 samples
        p_of_x_i_given_C1_test2(:,i) = normpdf(X_2(:,i),...
            m_est2(i,1), sqrt(S_est2(i,1)) );
        Naive_bayes_C1_test2 =...
            Naive_bayes_C1_test2.*p_of_x_i_given_C1_test2(:,i);

      %% class 2
        % P(x_i | C2 )
        p_of_x_i_given_C2_test(:,i) = normpdf(X_2(:,i),...
            m_est(i,2), sqrt(S_est(i,2)) );
        Naive_bayes_C2_test =...
            Naive_bayes_C2_test.*p_of_x_i_given_C2_test(:,i);
        % P(x_i | C2 ) 1000 samples
        p_of_x_i_given_C2_test2(:,i) = normpdf(X_2(:, i),...
            m_est2(i,2), sqrt(S_est2(i,2)) );
        Naive_bayes_C2_test2 =...
            Naive_bayes_C2_test2.*p_of_x_i_given_C2_test2(:,i);

    end
    naiv_classif = zeros(N_2,1);
    naiv_class_indexes = find(Naive_bayes_C1_test > Naive_bayes_C2_test);
    naiv_classif(naiv_class_indexes) = 1;
    err_naive = sum(naiv_classif ~= y_2)/N_2;
    
    %sum(y_test(naiv_class_indexes))/half_N_2
    naiv_class_indexes2 = find(Naive_bayes_C1_test2 > Naive_bayes_C2_test2);
    naiv_classif2 = zeros(N_2,1);    
    naiv_classif2(naiv_class_indexes2) = 1;    
    err_naive2 = sum(naiv_classif2 ~= y_2)/N_2;
    %err_naive2 = length(naiv_class_indexes2 )/N_2;

    disp('====== Error Naive Bayes ======');
    display(['100 train, 10,000 test: ', num2str(err_naive)]);
    display(['1000 train, 10,000 test: ', num2str(err_naive2 )]);
%% MLE BAYES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p_of_C1_given_x = 0;
p_of_C2_given_x = 0;
p_of_C1_given_x2 = 0;
p_of_C2_given_x2 = 0;

    [S_est_MLE(:,:,1), m_est_MLE(:,1)] =...
        multivGaussParamEstimate(X_1(1:half_N_1,:));
    [S_est_MLE(:,:,2), m_est_MLE(:,2)] =...
        multivGaussParamEstimate( X_1(half_N_1+1:N_1,:)  );

    [S_est_MLE(:,:,3), m_est_MLE(:,3)] =...
        multivGaussParamEstimate(X_12(1:half_N_12,:));
    [S_est_MLE(:,:,4), m_est_MLE(:,4)] =...
        multivGaussParamEstimate( X_12(half_N_12+1:N_12,:)  );

   for i = 1:N_2
%% 100 samples
        p_of_C1_given_x(i,:) = calcMultivNormal( X_2(i, :)',...
            m_est_MLE(:,1), S_est_MLE(:,:,1), 1/2);
        p_of_C2_given_x(i,:)  =...
            calcMultivNormal( X_2(i, :)', m_est_MLE(:,2),...
            S_est_MLE(:,:,2), 1/2);
%% 1000 samples
        p_of_C1_given_x2(i,:) = calcMultivNormal( X_2(i, :)',...
            m_est_MLE(:,3), S_est_MLE(:,:,3), 1/2);
        p_of_C2_given_x2(i,:)  =...
            calcMultivNormal( X_2(i, :)', m_est_MLE(:,4),...
            S_est_MLE(:,:,4), 1/2);

   end
    indexes_MLE = find(p_of_C1_given_x  > p_of_C2_given_x);
    MLE_classif = zeros(N_2,1);    
    MLE_classif(indexes_MLE) = 1;    
    err_MLE = sum(MLE_classif ~= y_2)/N_2; 
      
    indexes_MLE2 = find(p_of_C1_given_x2 > p_of_C2_given_x2);
    MLE_classif2 = zeros(N_2,1);    
    MLE_classif2(indexes_MLE2) = 1;    
    err_MLE2 = sum(MLE_classif2 ~= y_2)/N_2;

    disp('===== MLE Bayes ======');
    display(['100 train, 10,000 test: ', num2str(err_MLE)]);
    display(['1000 train, 10,000 test: ', num2str(err_MLE2)]);

%% TRUE VALUE BAYES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p_of_C1_given_x = 0;
p_of_C2_given_x = 0;
p_of_C1_given_x2 = 0;
p_of_C2_given_x2 = 0;

   for i = 1:N_2
%% 100 samples
        p_of_C1_given_x(i,:) = calcMultivNormal( X_2(i, :)',...
            m(:,1), S(:,:,1), 1/2);
        p_of_C2_given_x(i,:)  =...
            calcMultivNormal( X_2(i, :)', m(:,2),...
            S(:,:,2), 1/2);
   end

   for i=1:N_12
       %% 1000 samples
        p_of_C1_given_x2(i,:) = calcMultivNormal( X_12(i, :)',...
            m(:,1), S(:,:,1), 1/2);
        p_of_C2_given_x2(i,:)  =...
            calcMultivNormal( X_12(i, :)', m(:,2),...
            S(:,:,2), 1/2);
   end
   
    indexes_tru = find(p_of_C1_given_x  > p_of_C2_given_x);
    tru_classif = zeros(N_2,1);    
    tru_classif(indexes_tru) = 1;    
    err_TRUE = sum(tru_classif ~= y_2)/N_2;

    indexes_tru2 = find(p_of_C1_given_x2 > p_of_C2_given_x2);
    tru_classif2 = zeros(N_12,1);    
    tru_classif2(indexes_tru2) = 1;    
    err_TRUE2 = sum(tru_classif2 ~= y_12)/N_12; 
    
    disp('===== true parameters Bayes ======');
    display(['10,000 test: ', num2str(err_TRUE)]);
    %display(['1000 test: ', num2str(err_TRUE2)]);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% functions
function X = genTwoClassNormal(MU, COV, y)
    %multv_gauss(COV,mu, = exp(-1/2*COV\(x-mu)'*(x-mu))/(sqrt((2*pi)^N * det(COV))
    N = length(y);
    C_1_index = find(y == 1);%indexes for C1
    C_1_length = length(C_1_index);
    C_2_index = find(y ~= 1);%indexes for C2
    C_2_length = length(C_2_index);

    X = zeros(N,length(COV));
    X(C_1_index, :) = mvnrnd(MU(:,1), COV(:,:,1),C_1_length );
    X(C_2_index, :) = mvnrnd(MU(:,2), COV(:,:,2),C_2_length );
end

function [S_est, m_est] = muVarNaiveBayesGaussEstimates(X)
    %X is a matrix with :
        %-L features (columns) and
        %-N samples(rows)
    m_est = mean(X);% MEAN of each feature
    S_est = var(X);% MEAN of each feature
end

%%from S. Theodoridis, A. Pikrakis, K. Koutroumbas, D. Cavouras
function [S_est, m_est] = multivGaussParamEstimate(X)
    %X is a matrix with :
        %-L features (columns) and
        %-N samples(rows)
    [N, L] =  size(X);
    m_est = mean(X)';
    X = X'; %now columns are samples and rows are features
     % (as in the original function)
    %%as implemented by S. Theodoridis, A. Pikrakis, K. Koutroumbas, D. Cavouras
    S_est = 0;
    for i=1:N
        S_est=S_est+( X(:,i) - m_est )*( X(:,i) - m_est )';
    end
    S_est = 1/(N-1)*S_est;
end

%from  S. Theodoridis, A. Pikrakis, K. Koutroumbas, D. Cavouras
function f_x =calcMultivNormal(X,MU,SIGMA, P_X_given_C_i)
    %X is a matrix with :
        %-L features (columns) and
        %-N samples(rows)
    MU = MU;
    %X = X';
    n=length(MU);
    f_x = P_X_given_C_i*(1/( (2*pi)^(n/2)*det(SIGMA)^0.5) )*...
        exp(-0.5*(X - MU)'*inv(SIGMA)*(X - MU));

end