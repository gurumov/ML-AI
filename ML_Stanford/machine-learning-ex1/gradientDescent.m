function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    J_history(iter) = computeCost(X, y, theta);
    disp(J_history(iter));
    cost_vector = (X*theta - y);
    theta(1) = theta(1) - ((alpha * sum(cost_vector.*X(:,1))) / m );
    theta(2) = theta(2) - ((alpha * sum(cost_vector.*X(:,2))) / m );
end

end
