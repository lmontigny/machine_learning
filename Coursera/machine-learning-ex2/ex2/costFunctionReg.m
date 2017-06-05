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


h = sigmoid(X*theta);
J = 1/m * sum(-y.*log(h)-(1-y).*log(1-h)) + lambda/(2*m)*sum(theta(2:end).^2);


% h1=sigmoid(X(1,:)*theta(1,:));
% grad1 = 1/m * X(1,:)'*(h1-y(1,:)) + lambda/m*theta(1);

% h2=sigmoid(X(2:end,:)*theta(2:end));
% grad2 = 1/m * X(2:end,:)'*(h2-y(2:end)) + lambda/m*theta(2:end);


grad = 1/m * X'*(h-y) + lambda/m*theta;

%h1=sigmoid(X(1,:)*theta(1,:));
a = 1/m * X'*(h-y);
grad(1)=a(1);


%grad =
%grad(1)=1/m * X'*(h-y);



% =============================================================

end
