function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
								   hidden_layer_2_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
[Theta1 Theta2 Theta3] = reshapeTheta(nn_params, ...
							input_layer_size, ...
							hidden_layer_size, ...
							hidden_layer_2_size, ...
							num_labels);

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Compute y_vec
y_vec = zeros(m, num_labels);
for i = 1:m,
	y_vec(i, y(i)) = 1;
end;

% Calculate h
a_1 = [ones(size(X, 1), 1) X];
z_2 = a_1 * Theta1';
a_2 = sigmoid(z_2);

a_2 = [ones(size(a_2, 1), 1) a_2];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

a_3 = [ones(size(a_3, 1), 1) a_3];
z_4 = a_3 * Theta3';
a_4 = sigmoid(z_4);

h = a_4;

% Calculate Regularization term

theta1_reg = Theta1 .^ 2;
theta1_reg(:, 1) = 0;
theta2_reg = Theta2 .^ 2;
theta2_reg(:, 1) = 0;
theta3_reg = Theta3 .^ 2;
theta3_reg(:, 1) = 0;
reg = lambda/(2*m) * sum([theta1_reg(:); theta2_reg(:); theta3_reg(:)]);

% Calculate J
J = -1/m * sum(sum((y_vec .* log(h) .+ (1 .- y_vec) .* log(1 .- h)), 2)) + reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

DELTA_3 = zeros(size(Theta3));
DELTA_2 = zeros(size(Theta2));
DELTA_1 = zeros(size(Theta1));

for i = 1:m,
	% 1 Forward prop
	a_1 = [1; X(i, :)']; % gets ith row and transposes it

	z_2 = Theta1 * a_1;      % z_2
	a_2 = [1; sigmoid(z_2)]; % a_2

	z_3 = Theta2 * a_2;      % z_3
	a_3 = [1; sigmoid(z_3)]; % a_3

	z_4 = Theta3 * a_3; % z_4
	a_4 = sigmoid(z_4); % a_4
	
	h_i = a_4;

	% 2 Initial delta
	delta_4 = h_i .- y_vec(i, :)';
	
	% 3 Calculate delta
	delta_3 = Theta3(:, 2:end)' * delta_4 .* sigmoidGradient(z_3);
	delta_2 = Theta2(:, 2:end)' * delta_3 .* sigmoidGradient(z_2);

	% 4 Increment DELTA
	DELTA_3 = DELTA_3 + delta_4 * a_3';
	DELTA_2 = DELTA_2 + delta_3 * a_2';
	DELTA_1 = DELTA_1 + delta_2 * a_1';
end;

theta1_reg = Theta1;
theta1_reg(:, 1) = 0;
theta2_reg = Theta2;
theta2_reg(:, 1) = 0;
theta3_reg = Theta3;
theta3_reg(:, 1) = 0;
Theta1_grad = 1/m .* DELTA_1 + lambda/m .* theta1_reg;
Theta2_grad = 1/m .* DELTA_2 + lambda/m .* theta2_reg;
Theta3_grad = 1/m .* DELTA_3 + lambda/m .* theta3_reg;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];

end
