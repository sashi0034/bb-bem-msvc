#ifndef BB_BEM_H
#define BB_BEM_H

// ユーザー側で定義される関数のアトリビュート
#define BB_USER_FUNC extern

#define BB_API

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vector3_t {
    double x, y, z;
} vector3_t;

typedef struct bb_props_t {
    int nond;
    int nofc;
    int nond_on_face;
    int para_batch;
    vector3_t* np; /* [nond] */
    int* face2node; /* [nofc * nond_on_face] */
} bb_props_t;

/// @brief ユーザー側で定義される、計算対象要素における i, j 節点間の物理量 (例: 積分値) を返す関数
///
/// この関数は、境界要素法による 3 次元問題の離散化モデルに基づき、
/// 指定された要素内の2節点 (i, j) に関連する物理量を計算します。
///
/// @return 要素内の i, j 節点間の計算結果 (例: 積分値)
///
/// @note 各要素は多角形 (三角形など) であり、その頂点は節点として表される。
/// 節点座標 `np` および構成節点番号 `face2node` を用いて各要素の幾何情報を参照する。
BB_USER_FUNC double element_ij_(
    const int* p_i,
    const int* p_j,
    const bb_props_t* props
);

/**
 * @brief ユーザー側で定義される右辺ベクトル [i] を返す関数
 * 
 * @return 右辺ベクトルの i 番目の要素
 *
 * @code for fortran 'real(8) function rhs_vector_i(i, nint_para_fc, ndble_para_fc, int_para_fc, dble_para_fc)'
 */
BB_USER_FUNC double rhs_vector_i_(
    const int* p_i,
    const int* p_n,
    const int* nint_para_fc,
    const int* int_para_fc, /* [nofc * nint_para_fc] */
    const int* ndble_para_fc,
    const double* dble_para_fc, /* [nofc * ndble_para_fc] */
    const bb_props_t* props
);

// -----------------------------------------------

/// @brief ファイルから読み込まれる入力データ構造体
typedef struct bb_input_t {
    /// @brief 節点数 (Number of nodes)
    int nond;

    /// @brief 要素数 (Number of faces/elements)
    int nofc_unaligned;

    /// @brief 要素数 (Number of faces/elements, 8-aligned)
    int nofc;

    /// @brief 各要素を構成する節点数 (Nodes per face) 
    int nond_on_face;

    /// @brief 各要素上で定義される int 型パラメータ数 
    int nint_para_fc;

    /// @brief 各要素上で定義される double 型パラメータ数
    int ndble_para_fc;

    /// @brief 各要素におけるパラメータのバッチ数
    int para_batch_unaligned;

    /// @brief 各要素におけるパラメータのバッチ数 (8-aligned)
    int para_batch;

    /// @brief 節点座標 (サイズ: nond) 
    vector3_t* np;

    /// @brief 各要素を構成する節点番号 (サイズ: nofc * nond_on_face)
    int** face2node;

    /// @brief 各要素の int パラメータ (サイズ: para_batch * nofc * nint_para_fc)
    int*** int_para_fc;

    /// @brief 各要素の double パラメータ (サイズ: para_batch * nofc * ndble_para_fc)
    double*** dble_para_fc;
} bb_input_t;

/// @brief BEM の計算結果を格納する構造体
typedef struct bb_result_t {
    /// @brief 入力データ
    bb_input_t input;

    /// @brief 解ベクトルのサイズ (サイズ: nofc * 1)
    int dim;

    /// @brief 解ベクトル (サイズ: dim * batch)
    double** sol;

    /// @brief 演算部分の計算時間
    double compute_time;
} bb_result_t;

typedef enum {
    BB_OK = 0, // 成功
    BB_ERR_FILE_OPEN, // ファイルが開けなかった
    BB_ERR_FILE_FORMAT, // ファイルフォーマット不正
    BB_ERR_MEMORY_ALLOC, // メモリ確保失敗
    BB_ERR_UNKNOWN // その他不明なエラー
} bb_status_t;

typedef enum {
    BB_COMPUTE_NAIVE,
    BB_COMPUTE_CUDA,
    BB_COMPUTE_CUDA_WMMA,
    BB_COMPUTE_CUDA_CUBLAS,
} bb_compute_t;

/// @brief BEM の計算を実行する関数
BB_API bb_status_t bb_bem(const char* filename /* in */, bb_compute_t /* in */ compute, bb_result_t* result /* out */);

BB_API void release_bb_result(bb_result_t* result /* in */);

#ifdef __cplusplus
}
#endif

#endif // BB_BEM_H
