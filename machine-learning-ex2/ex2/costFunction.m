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

%z = theta(1) + theta(2) * X(:,2) + theta(3) * X(:,3);
%hypothesis = sigmoid(z);

hypothesis = sigmoid(X * theta)

%hypothesis = zeros(m,1)
%for i = 1:1:m
%    z = theta(1) + theta(2) * X(i,2) + theta(3) * X(i,3);
%    hypothesis(i) = sigmoid(z);    
%end

J = -sum(log(hypothesis) .* y + (1-y) .* log(1 - hypothesis))/m; 
grad = X' * (hypothesis - y) / m;

%for j = 1:1:size(grad)
%    if j == 1
%        grad(1)=sum(hypothesis-y)/m;    
%    else
%        grad(j)=sum((hypothesis-y) .* X(:,j))/m; 
%    end
%end



% =============================================================

end
