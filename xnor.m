## Author: liruxuan <liruxuan@liruxuans-MacBook-Pro.local>
## Created: 2017-03-25

% XNOR Neural Network in Octave
% a b | a XNOR b
%_______________
% 0 0 |    1
% 0 1 |    0
% 1 0 |    0
% 1 1 |    1

clear; close all; clc
addpath("./lib");

fprintf('XNOR Neural Network');

% Declare NN architecture
input_layer_size = 2 % a, b
hidden_layer_size = 4
hidden_layer_2_size = 4
output_layer_size = 2 % a XNOR b

% Data Import
fprintf("Loading training data\n");
A = load("data.txt");
X = featureScale(A(:,[1 2]));
y = A(:, 3);
m = size(X, 1);

fprintf('Size of X: %d x %d\n', size(X));
fprintf('Size of y: %d x %d\n', size(y));

% Plot Data
fprintf('Graphing Data\n');
figure;
hold on;

pos = find(y == 1);
neg = find(y == 0);
plot(X(pos, 1), X(pos, 2), "+");
plot(X(neg, 1), X(neg, 2), "o");
xlabel("Input a");
ylabel("Input b");
legend("True", "False");

hold off;

pause;

% Rand init
fprintf("Initializing Neural Network Parameters\n");

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_2_size)
initial_Theta3 = randInitializeWeights(hidden_layer_2_size, output_layer_size)

% Unroll parameters
initial_nn_params = [initial_Theta1(:); initial_Theta2(:); initial_Theta3(:)];

pause;

% Neural Network training
fprintf("Training Neural Network\n");

options = optimset("GradObj", "on", "MaxIter", 200);
lambda = 0;
costFunction = @(t) ...
	nnCostFunction(...
		t, ...
		input_layer_size, ...
		hidden_layer_size, ...
		hidden_layer_2_size, ...
		output_layer_size, ...
		X, ...
		y + 1, ... % y + 1 to index by 1
		lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and 2 from nn_params
[Theta1 Theta2 Theta3] = reshapeTheta(nn_params, ...
							input_layer_size, ...
							hidden_layer_size, ...
							hidden_layer_2_size, ...
							output_layer_size);

% Plot cost
figure; hold on;
plot(cost);
hold off;

pause;

% Check predictions against training data
% Training compare
pred = predict(Theta1, Theta2, Theta3, X);
pred = pred - 1; % -1 to undo index by 1
fprintf('Training Set Accuracy: %f\n', mean(pred == y) * 100);
% Cross Validation Compare
C = load("crossValidation.txt");
C_X = featureScale(C(:,[1 2]));
C_y = C(:,3);
pred = predict(Theta1, Theta2, Theta3, C_X);
pred = pred - 1;
fprintf('Cross Validation set Accuracy: %f\n', mean(pred == C_y) * 100);

pause;

% Print thetas
Theta1
Theta2
Theta3

pause;

% Re plot with hidden layer 1 lines
fprintf('Graphing Data and hidden layer 1 theta results\n');
figure; hold on;

pos = find(y == 1);
neg = find(y == 0);

plot(X(pos, 1), X(pos, 2), "+");
plot(X(neg, 1), X(neg, 2), "o");
xlabel("Input a");
ylabel("Input b");
legend("True", "False");
axis([-2 2 -2 2]);

for i=1:hidden_layer_size,
	line_a = (-Theta1(i, 1) - Theta1(i, 2)*3)/Theta1(i, 3);
	line_b = (-Theta1(i, 1) - Theta1(i, 2)*-3)/Theta1(i, 3);
	plot([-3; 3], [line_a; line_b]);
end;

hold off;
