/**
 * @file linear_regression.cpp
 * @author Ryan Curtin
 *
 * A simple implementation of the LinearRegressionFunction used as an example in
 * the paper, using EvaluateWithGradient().
 */
#define ENS_PRINT_INFO
#define ENS_PRINT_WARN
#include <ensmallen.hpp>

using namespace ens;

class LogisticRegressionFunction
{
 public:
  // Labels should be 0.0 or 1.0.
  LogisticRegressionFunction(arma::mat& predictors, arma::rowvec& responses) :
      predictors(predictors), responses(responses) { }

  double Evaluate(const arma::mat& parameters)
  {
    // Lambda is hardcoded to 0.5 for this example.
    const double objectiveRegularization = 0.5 / 2.0 *
        arma::dot(parameters.tail_cols(parameters.n_elem - 1),
                  parameters.tail_cols(parameters.n_elem - 1));

    // Calculate the sigmoid function values.
    const arma::rowvec sigmoids = 1.0 / (1.0 +
        arma::exp(-(parameters(0, 0) +
                    parameters.tail_cols(parameters.n_elem - 1) * predictors)));

    // Now compute the objective function using the sigmoids.
    double result = arma::accu(arma::log(1.0 - responses +
        sigmoids % (2 * responses - 1.0)));

    // Invert the result, because it's a minimization.
    return objectiveRegularization - result;
  }

  void Gradient(const arma::mat& parameters, arma::mat& gradient)
  {
    // Calculate the sigmoid function values.
    const arma::rowvec sigmoids = 1.0 / (1.0 +
        arma::exp(-(parameters(0, 0) +
                    parameters.tail_cols(parameters.n_elem - 1) * predictors)));

    gradient.set_size(arma::size(parameters));
    gradient[0] = -arma::accu(responses - sigmoids);
    gradient.tail_cols(parameters.n_elem - 1) = (sigmoids - responses) *
        predictors.t() + 0.5 * parameters.tail_cols(parameters.n_elem - 1);
  }

 private:
  arma::mat& predictors;
  arma::rowvec& responses;
};

double ComputeAccuracy(const arma::mat& X,
                       const arma::rowvec& y,
                       const arma::mat& theta)
{
  arma::rowvec labels = arma::conv_to<arma::rowvec>::from(
      arma::conv_to<arma::Row<size_t>>::from((1.0 /
      (1.0 + arma::exp(-theta(0) - theta.tail_cols(theta.n_elem - 1) * X))) +
      (0.5)));

  // Count the number of responses that were correct.
  size_t count = 0;
  for (size_t i = 0; i < y.n_elem; i++)
  {
    if (labels(i) == y(i))
      count++;
  }

  return (double) (count * 100) / y.n_elem;
}

int main(int argc, char** argv)
{
  if (argc < 5)
  {
    throw std::invalid_argument("usage: program <trainFile> <trainLabelsFile> "
        "<testFile> <testLabelsFile>");
  }

  std::string trainFile = argv[1];
  std::string trainLabelsFile = argv[2];
  std::string testFile = argv[3];
  std::string testLabelsFile = argv[4];

  // This is just noise... the model will be worthless.  But that doesn't
  // actually matter.
  arma::mat trainData, testData, trainLabelsIn, testLabelsIn;
  trainData.load(arma::csv_name(trainFile, arma::csv_opts::trans + arma::csv_opts::no_header));
  testData.load(arma::csv_name(testFile, arma::csv_opts::trans + arma::csv_opts::no_header));
  trainLabelsIn.load(arma::csv_name(trainLabelsFile, arma::csv_opts::trans + arma::csv_opts::no_header));
  testLabelsIn.load(arma::csv_name(testLabelsFile, arma::csv_opts::trans + arma::csv_opts::no_header));

  arma::rowvec trainLabels = trainLabelsIn.row(0);
  arma::rowvec testLabels = testLabelsIn.row(0);

  LogisticRegressionFunction lrf(trainData, trainLabels);
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10;

  arma::mat theta(1, trainData.n_rows + 1, arma::fill::zeros);

  arma::wall_clock clock;

  clock.tic();
  const double result = lbfgs.Optimize(lrf, theta);
  std::cout << clock.toc() << std::endl;
  std::cout << "obj: " << result << "\n";

  // Now compute the accuracy.
  std::cout << "Training set accuracy: "
      << ComputeAccuracy(trainData, trainLabels, theta)
      << ".\n";
  std::cout << "Test set accuracy: "
      << ComputeAccuracy(testData, testLabels, theta)
      << ".\n";
}
