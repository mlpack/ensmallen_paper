% This is actually a GNU Octave file because I don't have access to the Global
% Optimization Toolkit.
pkg load optim;

global trainData;
global trainLabels;
global testData;
global testLabels;

function [obj, grad] = lr(theta)
  global trainData;
  global trainLabels;
  objReg = 0.5 / 2.0 * (theta(2:end)' * theta(2:end));
  sigmoids = 1.0 ./ (1.0 .+ exp(-(theta(1) .+ trainData * theta(2:end))));
  innerSecondTerm = 1.0 .- trainLabels .+ sigmoids .* (2.0 .* trainLabels .- 1.0);

  obj = objReg - sum(log(innerSecondTerm));
  grad = [-sum(trainLabels .- sigmoids); (((sigmoids - trainLabels)' * trainData)' .+ 0.5 .* theta(2:end))];
endfunction

function [acc] = compute_accuracy(theta, data, labels)
  predictions = floor(1.0 ./ (1.0 .+ exp(-(theta(1) .+ data * theta(2:end)))) .+ 0.5);

  acc = sum(predictions == labels) / size(labels)(1);
end

args = argv();
trainFile = args{1, 1};
trainLabelsFile = args{2, 1};
testFile = args{3, 1};
testLabelsFile = args{4, 1};

trainData = csvread(trainFile);
trainLabels = csvread(trainLabelsFile);
testData = csvread(testFile);
testLabels = csvread(testLabelsFile);

% I guess maybe there is a JIT or something that might accelerate this, so let's
% let it "burn in" once.
theta0 = zeros(size(trainData)(2) + 1, 1);
control = {10, 0}; % maxiters, verbosity
bfgsmin('lr', {theta0}, control);

tic;
[thetaOut, objVal, conv, iters] = bfgsmin('lr', {theta0}, control);
toc

compute_accuracy(thetaOut, trainData, trainLabels)
compute_accuracy(thetaOut, testData, testLabels)
