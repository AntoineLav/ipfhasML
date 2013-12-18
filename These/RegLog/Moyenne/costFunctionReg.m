function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%Prediction Calcul
%X(:, 1) = [];
%theta(1) = [];
g = X * theta;
g = exp(-g);
g = ones(size(g)) + g;
predictions = ones(size(g)) ./ g;

%sqrErrors Calcul
sqrErrors1 = -y .* log(predictions);
sqrErrors2 = (ones(size(y)) - y);
sqrErrors3 = log(ones(size(predictions)) - predictions);
sqrErrors4 = (sqrErrors2 .* sqrErrors3);
% We do not regularize the parameter theta(1)
theta1 = theta;
theta1(1) = [];
sqrErrors5 = lambda / (2 * m) * sum(theta1.^2);

%Search the NaN values and replace them by a 0
%k = find(isnan(sqrErrors1));
%sqrErrors1(k) = 0;
%k = find(isnan(sqrErrors4));
%sqrErrors4(k) = 0;

sqrErrors = sqrErrors1 - sqrErrors4;

J = sum(sqrErrors) / m + sqrErrors5; 

grad(1) = sum(X(:, 1)' * (predictions - y)) / m;

for i = 2:size(theta)
    grad(i) = (sum(X(:, i)' * (predictions - y)) / m) + lambda * theta(i) / m;
end

% =============================================================

end
