%% Initialization
clear ; close all; clc

%% Load Data
struct_data = load('mos_data.mat');
table = struct_data.mos5_data;
data = zeros(5,54);
data(1,:) = table(5,:);
data(2,:) = table(6,:);
data(3,:) = table(10,:);
data(4,:) = table(13,:);
data(5,:) = table(15,:);

%% Initialise result array
result_predict = zeros(size(data,1), size(data,2)-2);
result_mos = zeros(1, size(data,2)-2);

result_predict_all_unknownUser = zeros(size(table,1), size(table,2)-2);
result_mos_all_unknownUser = zeros(1, size(table,2)-2);


%% For statement to calculate each user
for i=3:54

  %% Select X for User i
  X = data(:, [1, 2]); y = data(:, i);
  
  % Add Polynomial Features
  X = mapFeature(X(:,1), X(:,2));
  
  % Initialize fitting parameters
  initial_theta = zeros(size(X, 2), 1);

  % Set regularization parameter lambda to 1
  lambda = 1;
  
  % Compute initial cost and gradient for regularized logistic
  % regression
  [cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
  
  % Set Options
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  
  % Optimize
  [theta, J, exit_flag] = ...
	  fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
  
  % Compute accuracy on our training set
  p = predict(theta, X);
  result_predict(:, i-2) = p;
  result_mos(1, i-2) = mean(double(p == y)) * 100;  
 
  fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
  
  data_all = struct_data.mos1_data;
  
  %% Select X for User i
  X_all = data_all(:, [1, 2]); y_all = data_all(:, i);
  
  % Add Polynomial Features
  X_all = mapFeature(X_all(:,1), X_all(:,2));
  p_all = predict(theta, X_all);
  result_predict_all_unknownUser(:, i-2) = p_all;
  result_mos_all_unknownUser(1, i-2) = mean(double(p_all == y_all)) * 100;
  
end

result_predict5_unknownUser = result_predict;
result_mos5_unknownUser = result_mos;
save('These/RegLog/Moyenne/mos5_predict_unknownUser.mat','result_predict5_unknownUser');
save('These/RegLog/Moyenne/mos5_result_unknownUser.mat','result_mos5_unknownUser');

result_predict5_unknownUser_all = result_predict_all_unknownUser;
result_mos5_unknownUser_all = result_mos_all_unknownUser;
save('These/RegLog/Moyenne/mos5_predict_unknownUser_all.mat','result_predict5_unknownUser_all');
save('These/RegLog/Moyenne/mos5_result_unknownUser_all.mat','result_mos5_unknownUser_all');
