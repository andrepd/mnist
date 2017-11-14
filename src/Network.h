#ifndef NETWORK_H
#define NETWORK_H

#include <Eigen/Dense>
#include <vector>
#include <utility>

using Result_t = std::pair<std::vector<Eigen::VectorXd>,Eigen::VectorXd>;
using Resultb_t = std::pair<std::vector<Eigen::MatrixXd>,Eigen::MatrixXd>;

struct Network {
	enum class Reg: unsigned {
		none    = 0,
		Dropout = 1 << 0,
		L2      = 1 << 1
	};


	std::vector<Eigen::MatrixXd> weights;
	std::vector<Eigen::VectorXd> biases;

	// auto size() {return weights.size();};
	size_t size;  // TODO const
	// const size_t size() const;

	// Constructs a network with `n` layers
	Network(size_t n);
	// Constructs a network with layer sizes given by `sizes`, and initializes the weights randomly with a normal distribution around 0
	Network(std::initializer_list<int> sizes);
	// Constructs a network with weights previously saved to `fname` by .save()
	Network(const char* fname);

	// Save weights to `fname`
	void save(const char* fname) const;

	// Feed-forward, return final output
	Eigen::VectorXd run(Eigen::VectorXd in) const;
	// Feed-forward, return all layers' activations
	Result_t run_all(Eigen::VectorXd in) const;
	// Matrix-based feed-forward on batch, return final output
	Eigen::MatrixXd run_batch(Eigen::MatrixXd in) const;
	// Matrix-based feed-forward on batch, return all layers' activations
	Resultb_t run_all_batch(Eigen::MatrixXd in) const;

	// Calculate gradients. `result` is the result of feed-forward from run_all, `target` is desired output. Uses cross-entropy loss function
	Network backprop(const Result_t& result, const Eigen::VectorXd& target) const;
	// Ditto on a batch
	Network backprop_batch(const Resultb_t& result, const Eigen::MatrixXd& target) const;

	// Perform stochastic gradient descent on `max_steps` data points, in mini-batches of size `batch_size`. `calc_grad` should be a function that takes a size_t `i` and returns the `i`-th input/target pair. Pass members of reg or'd together (|) to enable regularization
	// Signature: std::pair<Eigen::VectorXd,Eigen::VectorXd> get_pair(size_t i);
	template<typename F>
	void grad_descent(F calc_grad, const size_t dataset_size, 
		const int max_steps, const int batch_size, 
		// const double eta, const double alpha, const double lambda, 
		const Reg ropts=Network::Reg::none);

	Network& operator+=(const Network& x);
};

// Lets members of Network::Reg be combined like type-safe bitflags
Network::Reg operator|(const Network::Reg a, const Network::Reg b);
Network::Reg operator&(const Network::Reg a, const Network::Reg b);

// Addition and scalar multiplication of networks is equal to adding/multiplying all the layers' weights and biases
Network operator+(Network a, const Network& b);
Network operator*(double a, Network b);
Network operator/(Network a, const double b);

#include "Network.tpp"

#endif /* NETWORK_H */
