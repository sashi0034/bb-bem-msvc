#pragma once

#include "../bb-bem-msvc/src/bb_bem.h"

namespace bb_bem_wrapper
{
	struct ResultHolder {
		bb_result_t bb_naive{};
		bb_result_t bb_cuda{};
		bb_result_t bb_cuda_wmma{};
		bb_result_t bb_cuda_cublas{};

		bb_result_t& get(bb_compute_t compute);

		bb_result_t& get(int compute);

		int para_batch_unaligned() const;

		void varifyResult();

		void release();
	};

	bb_compute_t NextIndex(bb_compute_t compute);

	std::string GetNameU8(bb_compute_t compute);

#ifdef SIV3D_PLATFORM
	inline String GetName(bb_compute_t compute) {
		return Unicode::Widen(GetNameU8(compute));
	}
#endif
}
