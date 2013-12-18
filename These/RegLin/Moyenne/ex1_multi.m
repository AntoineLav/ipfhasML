%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('dataUser30.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0d \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0d %.0d], y = %.0d \n', [X(1:10, 2:3) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.3;
num_iters = 10;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-g', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %d \n', theta);
fprintf('\n');

% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
%predict = [1 500 2];
%predict = [1 2 500];
predict = [1 720 500];

for i = 1:size(sigma, 2)
    predict(1, i+1) = (predict(1, i+1) - mu(1, i)) / sigma(1, i);
end

price = predict * theta; % You should change this


% ============================================================

fprintf(['Predicted MOS grade of a 720p, 500Kbits/s movie ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;


% =============================================================
% Compare all results

X = data(:, 1:2);
y = data(:, 3);
X = [ones(m, 1) X];
Xnorm = ones(size(X));

fprintf('X charged again\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

for j = 1:m
    for i = 1:size(sigma, 2)
        Xnorm(j, i+1) = (X(j, i+1) - mu(1, i)) / sigma(1, i);
    end
end

yCalc = Xnorm * theta;

pause;

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' y = %f, yCalc = %f \n', [y(:,1) yCalc(:,1)]');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%




%% Load Data
data = csvread('dataUser30.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================

%predict = [1 500 2];
%predict = [1 2 500];
predict = [1 720 500];

price = predict * theta; % You should change this


% ============================================================

fprintf(['Predicted MOS grade of a 720p, 500Kbits/s movie ' ...
         '(using normal equations):\n $%f\n'], price);


% =============================================================
% Compare all results

X = data(:, 1:2);
y = data(:, 3);
X = [ones(m, 1) X];

yCalc = X * theta;

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' y = %f, yCalc = %f \n', [y(:,1) yCalc(:,1)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

     
     