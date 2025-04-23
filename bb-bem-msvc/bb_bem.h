#ifndef BB_BEM_H
#define BB_BEM_H

// ユーザー側で定義される関数
#define BB_USER_FUNC

#define BB_API

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vector3_t {
    double x, y, z;
} vector3_t;

/// @brief ユーザー側で定義される、計算対象要素における i, j 節点間の物理量（例: 積分値）を返す関数
///
/// この関数は、境界要素法による 3 次元問題の離散化モデルに基づき、
/// 指定された要素内の2節点（i, j）に関連する物理量を計算します。
///
/// @param[in] i 節点 i のインデックス（0-origin）
/// @param[in] j 節点 j のインデックス（0-origin）
/// @param[in] nond 全体の節点数
/// @param[in] nofc 全体の要素数
/// @param[in] np 節点の3次元座標値配列（サイズ: [3][*nond]）
/// @param[in] face2node 各要素を構成する節点番号の配列（サイズ: [*nond_on_face][*nofc]）
///
/// @return 要素内の i, j 節点間の計算結果（例: 積分値）
///
/// @note 各要素は多角形（三角形など）であり、その頂点は節点として表される。
/// 節点座標 `np` および構成節点番号 `face2node` を用いて各要素の幾何情報を参照する。
/// 
/// @remark Fortran の場合は 'real(8) function element_ij(i, j, nond, nofc, np, face2node)' として定義してください。
BB_USER_FUNC double element_ij_(int* i, int* j, int* nond, int* nofc, vector3_t* np, int* face2node);

// -----------------------------------------------

/// @brief ファイルから読み込まれる入力データ構造体
typedef struct bb_input_t {
    /// @brief 節点数 (Number of nodes)
    int nond;

    /// @brief 要素数 (Number of faces/elements) 
    int nofc;

    /// @brief 各要素を構成する節点数 (Nodes per face) 
    int nond_on_face;

    /// @brief 各要素上で定義される int 型パラメータ数 
    int nint_para_fc;

    /// @brief 各要素上で定義される double 型パラメータ数 
    int ndble_para_fc;

    /// @brief 節点座標 (サイズ: nond) 
    vector3_t* np;

    /// @brief 各要素を構成する節点番号 (サイズ: nofc * nond_on_face)
    int** face2node;

    /// @brief 各要素の int パラメータ (サイズ: nofc * nint_para_fc)
    int** int_para_fc;

    /// @brief 各要素の double パラメータ (サイズ: nofc * ndble_para_fc)
    double** dble_para_fc;
} bb_input_t;

/// @brief BEM の計算結果を格納する構造体
typedef struct bb_result_t {
    /// @brief 入力データ
    bb_input_t input;

    /// @brief 解ベクトルのサイズ (サイズ: nofc * 1)
    int dim;

    /// @brief 解ベクトル (サイズ: dim)
    double* sol;
} bb_result_t;

typedef enum {
    BB_SUCCESS = 0, // 成功
    BB_ERR_FILE_OPEN, // ファイルが開けなかった
    BB_ERR_FILE_FORMAT, // ファイルフォーマット不正
    BB_ERR_MEMORY_ALLOC, // メモリ確保失敗
    BB_ERR_UNKNOWN // その他不明なエラー
} bb_status_t;

/// @brief BEM の計算を実行する関数
BB_API bb_status_t bb_bem(const char* filename, bb_result_t* result);

BB_API void release_bb_result(bb_result_t* result);

#ifdef __cplusplus
}
#endif

#endif
