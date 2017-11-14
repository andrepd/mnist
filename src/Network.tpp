#include <Eigen/Dense>
#include <fmt/format.h>
#include <algorithm>
#include <numeric>
#include <random>
// #include <tuple>

template<typename F> 
void Network::grad_descent(F get_pair, const size_t dataset_size, 
	const int max_steps, const int batch_size, 
	const Network::Reg ropts)
{
	constexpr double eta    = -0.5;  // Learning rate
	constexpr double alpha  = 0.2;  // Momentum factor
	constexpr double lambda = 0.1;  // L2 regularization coefficient

	// Helper struct, shuffles dataset and yields elements, reshuffling when it reaches the end
	struct getter {
		unsigned operator()() {
			if (head == indexes.end()) {
				std::shuffle(indexes.begin(), indexes.end(), rand);
				head = indexes.begin();
			}
			return *head++;
		}
		getter(size_t N) : rand((std::random_device())()) {
			indexes.resize(N);
			std::iota(indexes.begin(), indexes.end(), 0);

			std::shuffle(indexes.begin(), indexes.end(), rand);
			head = indexes.begin();
		}
	private:
		std::mt19937 rand;
		std::vector<unsigned> indexes;
		decltype(indexes.begin()) head;
	};

	// Parse regularization options
	const auto opt_dropout = static_cast<unsigned>(ropts & Reg::Dropout);
	const auto opt_l2      = static_cast<unsigned>(ropts & Reg::L2);
	if (opt_dropout) fmt::print("Dropout unimplemented.\n");
	if (opt_l2)      fmt::print("L2 regularization enabled.\n");

	fmt::print("Start training\n");

	Network step(size);
	getter get(dataset_size);
	double err;
	
	fmt::print("0/{}", max_steps);
	{
		Eigen::MatrixXd inp(weights.front().cols(), batch_size);
		Eigen::MatrixXd targ(weights.back().rows(), batch_size);
		for (int j=0; j<batch_size; j++) {
			auto res = get_pair(get());
			inp.col(j) = res.first;
			targ.col(j) = res.second;
			// std::tie(inp.col(j), targ.col(j)) = get_pair(get());
		}
		Network deltas = backprop_batch(run_all_batch(inp), targ);
		step = (1-alpha)*eta*deltas;
		// L2 regularization
		if (opt_l2) for (int j=0; j<size; j++) step.weights[j] += eta*lambda/dataset_size * weights[j];
		*this += step;
	}
	for (int i=1; i<max_steps; i++) {
		if (i % (max_steps>100 ? max_steps/100 : 1) == 0) {fmt::print("\r{}/{}", i, max_steps); fflush(stdout);}
		Eigen::MatrixXd inp(weights.front().cols(), batch_size);
		Eigen::MatrixXd targ(weights.back().rows(), batch_size);
		for (int j=0; j<batch_size; j++) {
			auto res = get_pair(get());
			inp.col(j) = res.first;
			targ.col(j) = res.second;
			// std::tie(inp.col(j), targ.col(j)) = get_pair(get());
		}
		Network deltas = backprop_batch(run_all_batch(inp), targ);
		step = (1-alpha)*eta*deltas + alpha*step;
		// L2 regularization
		if (opt_l2) for (int j=0; j<size; j++) step.weights[j] += eta*lambda/dataset_size * weights[j];
		*this += step;
	}
	
	fmt::print("\rFinish training\n");
}
