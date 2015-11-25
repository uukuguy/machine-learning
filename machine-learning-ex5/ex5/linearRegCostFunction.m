function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


J = sum((X * theta - y) .^ 2)/2/m + sum(theta(2:end) .^ 2) * lambda / 2 / m;

%grad = X' * (X * theta - y) / m;
hypothesis = X * theta;
for j = 1:1:size(grad)
    if j == 1
        grad(j)=sum((hypothesis-y) .* X(:,j))/m; 
    else
         grad(j)=sum((hypothesis-y) .* X(:,j))/m + theta(j) * lambda / m;        
    end
end




% =========================================================================

grad = grad(:);

end
