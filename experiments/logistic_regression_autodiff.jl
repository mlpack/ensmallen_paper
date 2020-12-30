using Optim
using LineSearches
using LinearAlgebra: dot
using DataFrames
using CSV

if length(ARGS) < 4
  print("need six arguments: trainFile, trainLabelsFile, testFile, testLabelsFile")
  exit(1)
end

trainFile = ARGS[1]
trainLabelsFile = ARGS[2]
testFile = ARGS[3]
testLabelsFile = ARGS[4]

# Logistic regression Evaluate().
function f(theta, X, y)
    # Lambda is hardcoded to 0.5.
    objReg = 0.5 / 2.0 * dot(theta[2:end], theta[2:end])

    sigmoids = 1.0 ./ (1.0 .+ exp.(-(theta[1] .+ theta[2:end]' * X)))'

    return objReg - sum(log.(1.0 .- y .+ sigmoids .* (2.0 .* y .- 1.0)))
end

function compute_accuracy(params, data, labels)::Float64
    predictions = floor.((1.0 ./ (1.0 .+ exp.(-params[1] .- params[2:end]' *
        data))) .+ 0.5)

    return sum(predictions' .== labels) / length(labels)
end

# Load MNIST data.
trainData = convert(Matrix{Float64}, CSV.read(trainFile, DataFrame, header=false))'
trainLabels = convert(Matrix{Int64}, CSV.read(trainLabelsFile, DataFrame, header=false))
testData = convert(Matrix{Float64}, CSV.read(testFile, DataFrame, header=false))'
testLabels = convert(Matrix{Int64}, CSV.read(testLabelsFile, DataFrame, header=false))
dim = size(trainData, 1)
print("dim: $(dim)\n")

point = zeros(dim + 1)
result = optimize(t -> f(t, trainData, trainLabels), point,
LBFGS(linesearch=LineSearches.BackTracking()), Optim.Options(x_tol = 0, f_tol = 0, g_tol = 0, iterations = 10))
print(result)
print("\n")

point = zeros(dim + 1)
result = @time optimize(t -> f(t, trainData, trainLabels), point,
    LBFGS(linesearch=LineSearches.BackTracking()), Optim.Options(x_tol = 0, f_tol = 0, g_tol = 0, iterations = 10))

print("Training set accuracy: $(compute_accuracy(Optim.minimizer(result), trainData, trainLabels))\n")
print("Test set accuracy: $(compute_accuracy(Optim.minimizer(result), testData, testLabels))\n")
