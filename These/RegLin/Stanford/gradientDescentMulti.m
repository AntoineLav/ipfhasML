function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    predictions = X * theta;
   
    sqrErrorsTheta = zeros(size(X, 1), size(X, 2));
    temp = zeros(1, size(X, 2));
    
    for i = 1:size(X, 2)
        sqrErrorsTheta(:, i) = (predictions - y) .* X(:, i);
    end
       
    %sqrErrorsTheta0 = predictions - y;
    %sqrErrorsTheta1 = (predictions - y) .* X(:,2);
    
    for j = 1:size(X, 2)
        temp(1, j) = alpha * sum(sqrErrorsTheta(:, j)) / m;
    end
        
    %temp0 = alpha * sum(sqrErrorsTheta0) / m;
    %temp1 = alpha * sum(sqrErrorsTheta1) / m;
    
    for k = 1:size(X, 2)
        theta(k) = theta(k) - temp(1, k);
    end  
        
    %theta(1) = theta(1) - temp0;
    %theta(2) = theta(2) - temp1;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
