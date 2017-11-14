#include "Network.h"

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cstdio>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <utility>
#include <random>

using namespace std;
using namespace Eigen;

using fmt::print;

static const Eigen::IOFormat noaligncols(Eigen::FullPrecision, Eigen::DontAlignCols, " ", "\n", "", "", "", "");

// Sigmoid function
static double sigm(const double x) {
	if (x<-64.)
		return 0.;
	if (x> 64.)
		return 1.;
	return 1./(1.+exp(-x));
}

// Derivative of sigmoid (d/dx(sigm(x)) = dsigm(sigm(x)))
static double dsigm(const double x) {
	return x*(1-x);
}

Network::Network(size_t n) : size(n) {
	weights = vector<MatrixXd>(n);
	biases  = vector<VectorXd>(n);
}

Network::Network(initializer_list<int> sizes) : Network(distance(sizes.begin(), sizes.end()) - 1) {
	mt19937 rand((random_device())());	
	// normal_distribution<double> dist(0., 1.);
	normal_distribution<double> dist(0., 1./static_cast<double>(*sizes.begin()));
	auto gauss = [&rand,&dist](double x) {return dist(rand);};

	for (size_t i=0; i<size; i++) {
		const auto& s1 = *(sizes.begin()+i);
		const auto& s2 = *(sizes.begin()+i+1);
		// weights[i] = MatrixXd::Random(s2, s1);
		// biases[i]  = VectorXd::Random(s2);
		weights[i] = MatrixXd::NullaryExpr(s2, s1, gauss);
		biases[i]  = VectorXd::NullaryExpr(s2, gauss);
	}
}

Network operator+(Network a, const Network& b) {
	// assert(a.size == b.size);

	for (size_t i=0; i<a.size; i++) {
		a.weights[i] += b.weights[i];
		a.biases[i]  += b.biases[i];
	}
	return a;
}

Network operator*(double a, Network b) {
	for (size_t i=0; i<b.size; i++) {
		b.weights[i] *= a;
		b.biases[i]  *= a;
	}
	return b;
}

Network operator/(Network a, const double b) {
	return (1./b) * a;
}

Network& Network::operator+=(const Network& x) {
	// assert(size == x.size);

	for (size_t i=0; i<size; i++) {
		weights[i].noalias() += x.weights[i];
		biases[i].noalias()  += x.biases[i];
	}
	return *this;
}



// Run neural network and return output
VectorXd Network::run(VectorXd in) const {
	for (size_t i=0; i<size; i++) {
		in = (weights[i]*in + biases[i]).unaryExpr(ref(sigm));
	}
	return in;
}

// Run neural network and return all layers
Result_t Network::run_all(VectorXd in) const {
	vector<VectorXd> partials(size);
	auto p = partials.begin();
	for (size_t i=0; i<size; i++) {
		*p++ = in;
		in = (weights[i]*in + biases[i]).unaryExpr(ref(sigm));
	}
	return make_pair(partials, in);
}

// Run neural network for a mini-batch and return output
MatrixXd Network::run_batch(MatrixXd in) const {
	const VectorXd ones = VectorXd::Ones(in.cols());

	for (size_t i=0; i<size; i++) {
		in = (weights[i]*in + biases[i]*ones.transpose()).unaryExpr(ref(sigm));
	}
	return in;
}

// Run neural network for a mini-batch and return all layers
Resultb_t Network::run_all_batch(MatrixXd in) const {
	const VectorXd ones = VectorXd::Ones(in.cols());

	vector<MatrixXd> partials(size);
	auto p = partials.begin();
	for (size_t i=0; i<size; i++) {
		*p++ = in;
		in = (weights[i]*in + biases[i]*ones.transpose()).unaryExpr(ref(sigm));
	}
	return make_pair(partials, in);
}

// Save weights to file
void Network::save(const char* fname) const {
	FILE* f = fopen(fname, "w");

	print(f, "{}\n", size);
	for (size_t i=0; i<size; i++) {
		print(f, "{} {}\n", weights[i].rows(), weights[i].cols());
		print(f, "{}\n", weights[i].format(noaligncols));
		print(f, "{}\n", biases[i].format(noaligncols));
	}

	fclose(f);
}

// Load weights from file (saved with `save`)
Network::Network(const char* fname) {
	FILE* f = fopen(fname, "r");
	if (f != nullptr) {
		fscanf(f, "%zu", &size);
		weights = vector<MatrixXd>(size);
		biases  = vector<VectorXd>(size);
		for (size_t n=0; n<size; n++) {
			size_t rows, cols;
			fscanf(f, "%zu %zu", &rows, &cols);
			weights[n].resize(rows, cols);
			for (size_t i=0; i<rows; i++) {
				for (size_t j=0; j<cols; j++) {
					fscanf(f, "%lf", &(weights[n](i,j)));
				}
			}
			biases[n].resize(rows);
			for (size_t i=0; i<rows; i++) {
				fscanf(f, "%lf", &(biases[n](i)));
			}
		}
		fclose(f);
	} else {
		perror("Error opening weights file");
		size = 0;
	}
}

// Calculate gradient of E = t⋅ln(y) - (1-t)⋅ln(1-y) with respect to weights
Network Network::backprop(const Result_t& result, const VectorXd& target) const {
	// const auto& [partials, out] = result;  // C++17
	const auto& partials = result.first;
	const auto& out      = result.second;

	vector<VectorXd> errs(size);

	// Output layer
	errs.back() = (out-target);
	// Hidden layers
	for (size_t i=size-2; i!=static_cast<size_t>(-1); i--) {
		errs[i] = (weights[i+1].transpose()*errs[i+1]) .cwiseProduct( partials[i+1].unaryExpr(ref(dsigm)) );
	}

	// Weight deltas
	Network deltas(size);
	for (size_t i=0; i<size; i++) {
		deltas.weights[i] = errs[i] * partials[i].transpose();
		deltas.biases[i] = errs[i];
	}
	// deltas.biases = errs;

	return deltas;
}

// Calculate gradient of E = t⋅ln(y) - (1-t)⋅ln(1-y) with respect to weights, on a batch
Network Network::backprop_batch(const Resultb_t& result, const MatrixXd& target) const {
	// const auto& [partials, out] = result;  // C++17
	const auto& partials = result.first;
	const auto& out      	= result.second;

	vector<MatrixXd> errs(size);

	// Output layer
	errs.back() = (out-target);
	// Hidden layers
	for (size_t i=size-2; i!=static_cast<size_t>(-1); i--) {
		errs[i] = (weights[i+1].transpose()*errs[i+1]) .cwiseProduct( partials[i+1].unaryExpr(ref(dsigm)) );
	}

	// Weight deltas
	const auto bsize = target.cols();
	Network deltas(size);
	{
		for (size_t j=0; j<size; j++) {
			deltas.weights[j] = errs[j].col(0) * partials[j].col(0).transpose();	
			deltas.biases[j] = errs[j].col(0);
		}
	}
	for (int i=1; i<bsize; i++) {
		for (size_t j=0; j<size; j++) {
			deltas.weights[j] += errs[j].col(i) * partials[j].col(i).transpose();	
			deltas.biases[j] += errs[j].col(i);
		}
	}
	// deltas.biases = errs;

	return deltas/bsize;
}

Network::Reg operator|(const Network::Reg a, const Network::Reg b) {
	return static_cast<Network::Reg>(static_cast<unsigned>(a) | static_cast<unsigned>(b));
}

Network::Reg operator&(const Network::Reg a, const Network::Reg b) {
	return static_cast<Network::Reg>(static_cast<unsigned>(a) & static_cast<unsigned>(b));
}
