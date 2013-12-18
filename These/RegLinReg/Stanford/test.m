X = [ones(10,1) sin(1:1.5:15)' cos(1:1.5:15)'];
y = sin(1:3:30)';
Xval = [ones(10,1) sin(0:1.5:14)' cos(0:1.5:14)'];
yval = sin(1:10)';

[lambda_vec, error_train, error_val] = ...
        validationCurve(X, y, Xval, yval);
out = sprintf('%0.5f ', ...
        [lambda_vec(:); error_train(:); error_val(:)]);