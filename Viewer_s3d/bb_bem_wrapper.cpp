#include "bb_bem_wrapper.h"

namespace
{
	double compute_relative_error(const bb_result_t& a, const bb_result_t& b) {
		if (a.dim != b.dim) {
			return -1.0;
		}

		double a_b_2{};
		double a_2{};
		for (int i = 0; i < a.dim; ++i) {
			for (int n = 0; n < a.input.para_batch_unaligned; ++n) {
				a_b_2 += (a.sol[i][n] - b.sol[i][n]) * (a.sol[i][n] - b.sol[i][n]);
				a_2 += (a.sol[i][n]) * (a.sol[i][n]);
			}
		}

		return std::sqrt(a_b_2 / a_2);
	}
}

bb_result_t& bb_bem_wrapper::ResultHolder::get(bb_compute_t compute) {
	switch (compute) {
	case BB_COMPUTE_NAIVE:
		return bb_naive;
	case BB_COMPUTE_CUDA:
		return bb_cuda;
	case BB_COMPUTE_CUDA_WMMA:
		return bb_cuda_wmma;
	default:
		assert(false);
		return bb_naive; // Should never reach here
	}
}

int bb_bem_wrapper::ResultHolder::para_batch_unaligned() const {
	return bb_naive.input.para_batch_unaligned;
}

void bb_bem_wrapper::ResultHolder::varifyResult() {
	std::cout << "----------------------------------------------- Result verification\n";

	std::cout << std::format("Relative error between Naive and Cuda: {}\n",
	                         compute_relative_error(bb_naive, bb_cuda));
	std::cout << std::format("Relative error between Naive and Cuda-WMMA: {}\n",
	                         compute_relative_error(bb_naive, bb_cuda_wmma));
	std::cout << std::format("Relative error between Cuda and Cuda-WMMA: {}\n",
	                         compute_relative_error(bb_cuda, bb_cuda_wmma));

	std::cout << std::format("Compute time (Naive): {} sec\n", bb_naive.compute_time);
	std::cout << std::format("Compute time (Cuda): {} sec\n", bb_cuda.compute_time);
	std::cout << std::format("Compute time (Cuda-WMMA): {} sec\n", bb_cuda_wmma.compute_time);
}

void bb_bem_wrapper::ResultHolder::release() {
	release_bb_result(&bb_naive);
	release_bb_result(&bb_cuda);
	release_bb_result(&bb_cuda_wmma);
}

std::string bb_bem_wrapper::GetNameU8(bb_compute_t compute) {
	switch (compute) {
	case BB_COMPUTE_NAIVE:
		return "Naive";
	case BB_COMPUTE_CUDA:
		return "CUDA";
	case BB_COMPUTE_CUDA_WMMA:
		return "CUDA WMMA";
	default:
		assert(false);
		return {};
	}
}
