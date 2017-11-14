#include "mnist_utils.h"
#include <Eigen/Dense>
#include <vector>
#include <fmt/format.h>
#include <utility>
#include <cstdint>
#include <cstdio>

using namespace std;
using namespace Eigen;
using fmt::print;

// Reads a big-endian 32 bit unsigned integer from file `f`
// Care! Doesn't perform any checks: assumes little-endian platform
uint32_t read_bigendian_int32(FILE* f) {
	unsigned char buf[4];
	// fread(buf, 4, 1, f);
	for (auto p = &buf[3]; p>=&buf[0]; p--) {
		fread(p, 1, 1, f);
	}
	return *(reinterpret_cast<uint32_t*>(buf));
}

// Reads images in the idx3 format from `fname`. Returns vector of fp pixel values in row-major order
vector<VectorXd> read_images(const char* fname) {
	FILE* f = fopen(fname, "rb");
	if (f != nullptr) {
		// Header
		const auto header = read_bigendian_int32(f);
		if (header != 0x0803) {
			print(stderr, "Error: File {} isn't a idx3-ubyte file.\n");
			return vector<VectorXd>(0);
		}
		// Number of items, rows, and columns
		const auto N = read_bigendian_int32(f);
		const auto rows = read_bigendian_int32(f);
		const auto cols = read_bigendian_int32(f);
		// Read images
		vector<VectorXd> ret(N);
		unsigned char buf[rows*cols];
		for (auto i = decltype(N){0}; i<N; i++) {
			fread(buf, sizeof(buf), 1, f);
			ret[i] = VectorXd(rows*cols);
			for (auto j = decltype(rows*cols){0}; j<rows*cols; j++) {
				ret[i](j) = static_cast<double>(buf[j]) / 255.;
			}
		}
		fclose(f);
		return ret;
	} else {
		perror("Error opening images file");
		return vector<VectorXd>(0);
	}
}

// Reads labels in the idx1 format from `fname`. Returns vector of unsigned 8-bit integers
vector<uint8_t> read_labels(const char* fname) {
	FILE* f = fopen(fname, "rb");
	if (f != nullptr) {
		// Header
		const auto header = read_bigendian_int32(f);
		if (header != 0x0801) {
			print(stderr, "Error: File {} isn't a idx1-ubyte file.\n");
			return vector<uint8_t>(0);
		}
		// Number of items, rows, and columns
		const auto N = read_bigendian_int32(f);
		// Read labels
		vector<uint8_t> ret(N);
		for (auto i = decltype(N){0}; i<N; i++) {
			fread(&ret[0], ret.size(), 1, f);
		}
		fclose(f);
		return ret;
	} else {
		perror("Error opening labels file");
		return vector<uint8_t>(0);
	}
}

// Prints a size `rows`x`cols` image in ASCII
void print_img(const VectorXd& img, const int rows, const int cols) {
	static constexpr char shades[] = ".:-=+*#%@";
	for (int i=0; i<rows; i++) {
		for (int j=0; j<cols; j++) {
			if (img(i*cols + j) < 1./4) {
				putchar(' ');
			} else if (img(i*cols + j) < 2./4) {
				putchar('-');
			} else if (img(i*cols + j) < 3./4) {
				putchar('%');
			} else {
				putchar('#');
			}
			// putchar(shades[static_cast<int>(img(i*cols+j)*10)]);
		}
		putchar('\n');
	}
}
