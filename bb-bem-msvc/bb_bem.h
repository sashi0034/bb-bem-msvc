#ifndef BB_BEM_H
#define BB_BEM_H

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
double element_ij_(int* i, int* j, int* nond, int* nofc, vector3_t* np, int* face2node);

#endif
