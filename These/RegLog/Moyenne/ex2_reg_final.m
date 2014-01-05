%% Initialization
clear ; close all; clc

%% Load Data
struct_data = load('mos_data.mat');
data = struct_data.mos5_data;

%% Initialise result array
result_predict = zeros(size(data,1), size(data,2)-2);
result_mos = zeros(1, size(data,2)-2);

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
end

result_predict5 = result_predict;
result_mos5 = result_mos;
save('These/RegLog/Moyenne/mos5_predict.mat','result_predict5');
save('These/RegLog/Moyenne/mos5_result.mat','result_mos5');
