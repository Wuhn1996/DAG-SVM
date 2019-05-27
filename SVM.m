clear; clc;
load('MNIST_data.mat')
polynomial_deg = 4;


round = 0;
votes = zeros(size(test_samples_labels,1),10); 

% Train classifier_m_n
for m = 0 : 8
    for n = m + 1 : 9
        round = round + 1;
        [x_mat, y_vec] = strip_m_n(train_samples,train_samples_labels,m,n);
        alpha_vec = findAlpha(x_mat, y_vec, polynomial_deg);
        pred_vec = predict_class(alpha_vec,x_mat,y_vec,test_samples, polynomial_deg);
        m_class = pred_vec > 0;
        pred_vec(m_class) = m;
        pred_vec(~m_class) = n; 
        for i = 1:size(pred_vec,1)
            votes(i, pred_vec(i) + 1) = votes(i, pred_vec(i) + 1) + 1; 
        end
    end
end

% Confusion matrix
conf_mat_1_1 = computeConf(votes, test_samples_labels);

%% Section 3 Train Classifier with 1-rest scheme
round = 0;
pred_mat = zeros(size(test_samples,1),10); % prediction of 10 classifiers

% Train classifier_m 
for m = 0 : 9
    round = round + 1;
    x_mat = train_samples;
    y_vec = train_samples_labels;
    m_class = y_vec == m;
    y_vec(m_class) = 1;
    y_vec(~m_class) = -1/9; % let negative class has target -1/(K-1)
    alpha_vec = findAlpha(x_mat, y_vec, polynomial_deg);
    pred_mat(:, m + 1) = predict_class(alpha_vec,x_mat,y_vec,test_samples, polynomial_deg);
end

% Confusion matrix
conf_mat_1_rest = computeConf(pred_mat ,test_samples_labels);

%% Section 4 Train DAGSVM
votes = ones(size(test_samples_labels,1),10); 

for depth = 1 : 9 % tree depth, depth i has i nodes
    for m = 0 : depth - 1  
        n = m + (10 - depth); 
        [x_mat, y_vec] = strip_m_n(train_samples,train_samples_labels,m,n);
        alpha_vec = findAlpha(x_mat, y_vec, polynomial_deg);
        pred_vec = predict_class(alpha_vec,x_mat,y_vec,test_samples,polynomial_deg);
        m_class = pred_vec > 0; 
        votes(m_class, n + 1) = 0;
        votes(~m_class, m + 1) = 0;
    end
end

% Confusion Matrix
conf_mat_DAG = computeConf(votes, test_samples_labels);

%% Compute confusion matrix and post-processing

function conf_mat = computeConf(votes, test_samples_labels) 


conf_mat = zeros(10,10); 
[max_counts, max_index] = max(votes,[],2);
for i = 1:size(max_index, 1)
    conf_mat(test_samples_labels(i) + 1, max_index(i)) ...
        = conf_mat(test_samples_labels(i) + 1, max_index(i)) + 1;
end

disp(conf_mat);
accuracy = trace(conf_mat) / size(test_samples_labels,1);
fprintf('Accuracy is: %.3f\n\n',accuracy);

end

%% function to solve quadratic programming 

function alpha_vec = findAlpha(x_mat, y_vec, poly_deg)
N = size(y_vec,1); % N data points
H = ((x_mat * x_mat').^poly_deg) .* (y_vec * y_vec');
f = -ones(N,1); 
A = -eye(N);
b = zeros(N,1);
Aeq = [y_vec'; zeros(N-1,N)]; 
beq = zeros(N,1); 
options = optimoptions('quadprog',...
    'Algorithm','interior-point-convex','Display','off');
alpha_vec = quadprog(H, f, A, b, Aeq, beq, [],[],[], options);

end

%% predict testing data class with alpha vector 

function prediction_vec = predict_class(alpha_vec, x_mat, y_vec, test_data, poly_deg)
        
        support_index = alpha_vec > 0.0001;
        support_mat_x = x_mat(support_index,:);
        support_vec_y = y_vec(support_index);
        support_alpha = alpha_vec(support_index);
        
        M = size(support_vec_y,1); 
        b = 1/M * sum(support_vec_y - ((support_mat_x * support_mat_x').^poly_deg * ...
            (support_vec_y .* support_alpha)));
        prediction_vec = (test_data * support_mat_x').^poly_deg * (support_vec_y .* support_alpha) + b;
        
end

function [x_mat, y_vec] = strip_m_n(data, label, m, n)

x_mat = [];
y_vec = [];
for i = 1:size(data,1)
    if label(i) == m
        y_vec = [y_vec; 1];
    elseif label(i) == n
        y_vec = [y_vec; -1];
    else
        continue
    end
    x_mat = [x_mat; data(i,:)];
end

end