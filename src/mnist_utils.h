#ifndef MNIST_UTILS_H
#define MNIST_UTILS_H

#include <Eigen/Dense>
#include <vector>
#include <cstdint>

// Vector of images and labels such that imgs[i] has label nums[i]
/*struct MNIST_set {
	std::vector<Eigen::VectorXd> imgs;
	std::vector<std::uint8_t> nums;	
};*/

// Reads a big-endian 32 bit unsigned integer from file `f` (Careful! Doesn't perform any checks: assumes little-endian platform)
uint32_t read_bigendian_int32(FILE* f);

// Reads images in the idx3 format from `fname`. Returns vector of fp pixel values in row-major order
std::vector<Eigen::VectorXd> read_images(const char* fname);

// Reads labels in the idx1 format from `fname`. Returns vector of unsigned 8-bit integers
std::vector<std::uint8_t> read_labels(const char* fname);

// Prints a size `rows`x`cols` image in ASCII
void print_img(const Eigen::VectorXd& img, const int rows=28, const int cols=28);

#endif /* MNIST_UTILS_H */
