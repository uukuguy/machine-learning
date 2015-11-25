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

hypothesis = sigmoid(X * theta)
J = -sum(log(hypothesis) .* y + (1-y) .* log(1 - hypothesis))/m + (lambda / (2 * m)) * (theta(2) ^2 + theta(3) ^2)

for j = 1:1:size(grad)
    if j == 1
        grad(1)=sum(hypothesis-y)/m;    
    else
        grad(j)=sum((hypothesis-y) .* X(:,j))/m + (lambda / m) * theta(j); 
    end
end





% =============================================================

end
