function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

%Prediction Calcul
g = X * theta;
g = exp(-g);
g = ones(size(g)) + g;
predictions = ones(size(g)) ./ g;

%sqrErrors Calcul
sqrErrors1 = -y .* log(predictions);
sqrErrors2 = (ones(size(y)) - y);
sqrErrors3 = log(ones(size(predictions)) - predictions);
sqrErrors4 = (sqrErrors2 .* sqrErrors3);

%Search the NaN values and replace them by a 0
k = find(isnan(sqrErrors1));
sqrErrors1(k) = 0;
k = find(isnan(sqrErrors4));
sqrErrors4(k) = 0;

sqrErrors = sqrErrors1 - sqrErrors4;

% Cost Function
J = sum(sqrErrors) / m;

% Partial derivative

for i = 1:size(theta)
    grad(i) = sum((predictions - y) .* X(:, i)) / m;
end
    
% =============================================================

end
