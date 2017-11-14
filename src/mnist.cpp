#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include "libs/fmt/format.h"
#include "libs/fmt/ostream.h"
#include <utility>
#include <cstdint>
#include <fenv.h>
#include "Network.h"
#include "mnist_utils.h"

using namespace std;
using namespace Eigen;
using fmt::print;

// Classifies `img`, compares with target `num`, and returns the weight deltas
Network classify_err(const Network& net, const VectorXd& img, const uint8_t num) {
	const auto result = net.run_all(img);
	VectorXd target = VectorXd::Zero(10);
	target(num) = 1.;

	return net.backprop(result, target);
}

// Classifies `img`
uint8_t classify(const Network& net, const VectorXd& img) {
	const auto result = net.run(img);

	uint8_t num;
	result.maxCoeff(&num);

	return num;
}

// Takes a vector of images and corresponding vector of labels and returns a closure that takes an index `i` and returns the `i`-th input/target vector pair
auto func(const vector<VectorXd>& imgs, const vector<uint8_t>& nums) {
	// Preprocess labels into target vectors
	vector<VectorXd> targs(nums.size());
	for (size_t i=0; i<nums.size(); i++) {
		targs[i] = VectorXd::Zero(10);
		targs[i](nums[i]) = 1.;		
	}
	// Return closure
	return [&imgs,targs](const size_t i) -> pair<VectorXd,VectorXd> {
		return {imgs[i], targs[i]};
	};
}

// Returns number of correct guesses for `N` random images in test set (imgs,nums)
int benchmark(const Network& net, vector<VectorXd> imgs, vector<uint8_t> nums, const int N) {
	int ret = 0;
	for (int i=0; i<N; i++) {
		const auto index = rand() % imgs.size();
		const auto result = classify(net, imgs[index]);
		if (result == nums[index]) {
			ret++;
		}
	}
	return ret;
}

// Returns number of correct guesses all the images in the set
int benchmark(const Network& net, vector<VectorXd> imgs, vector<uint8_t> nums) {
	int ret = 0;
	for (size_t i=0; i<imgs.size(); i++) {
		const auto result = classify(net, imgs[i]);
		if (result == nums[i]) {
			ret++;
		}
	}
	return ret;
}

int main(int argc, char const *argv[])
{
	feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT & ~FE_UNDERFLOW);
	srand(time(NULL));

	// Read MNIST data from files
	const auto train_images = read_images("train-images.idx3-ubyte");
	const auto train_labels = read_labels("train-labels.idx1-ubyte");
	const auto test_images = read_images("t10k-images.idx3-ubyte");
	const auto test_labels = read_labels("t10k-labels.idx1-ubyte");
	// Check if sizes are ok
	assert(train_images.size() == train_labels.size());
	assert(test_images.size()  == test_labels.size());
	const auto train_size = train_images.size();
	const auto test_size  = test_images.size();
	assert(train_size == 60000);
	assert(test_size  == 10000);

	print("Dataset read.\n");

	// Create a network with 28*28 input neurons, one 300-neuron hidden layer, and 10 output neurons (and random-initialize weights)
	Network net({28*28, 300, 10});

	// Uncomment to print 10 sample images
	print("id   lbl guessed\n");
	for (int i=0; i<10; i++) {
		const auto index = rand() % test_images.size();
		print("{:4} {}   {} ======\n", index, test_labels[index], classify(net, test_images[index]));
		print_img(test_images[index]);
	}
	print("Before training: {}/{}\n", benchmark(net, test_images, test_labels), test_size);
	
	// Train network using SGD with mini-batches of size 16, for 10 epochs. Enable L2 regularization
	for (int i=0; i<10; i++) {
		net.grad_descent(
			func(train_images, train_labels), train_size, 
			train_size/8, 16, 
			Network::Reg::L2);
		print("Partial: {}/{}\n", benchmark(net, test_images, test_labels), test_size);
	}

	// Uncomment to print 10 sample images
	print("id   lbl guessed\n");
	for (int i=0; i<10; i++) {
		const auto index = rand() % test_images.size();
		print("{:4} {}   {} ======\n", index, test_labels[index], classify(net, test_images[index]));
		print_img(test_images[index]);
	}
	print("After training: {}/{}\n", benchmark(net, test_images, test_labels), test_size);

	// Save weights to file
	net.save("mnist-300.dat");

	return 0;
}
