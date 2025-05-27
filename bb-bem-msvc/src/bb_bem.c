#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#include "stl_loader.h"

#include "bb_bem.h"

#include "bicgstab_naive.h"
#include "bicgstab_cuda.h"
#include "bicgstab_cuda_wmma.h"

#if !defined(BB_NO_MAIN)
int main() {
    bb_result_t result;
    bb_bem("input.txt", BB_COMPUTE_CUDA, &result); // CUDA で実行

    // ----------------------------------------------- fp
    FILE* fp = fopen("out2.data", "w");
    for (int n = 0; n < result.input.para_batch; n++) {
        for (int i = 0; i < result.dim; i++) {
            fprintf(fp, "%20.14e \n", result.sol[i][n]);
        }
    }

    fclose(fp);
    // -----------------------------------------------

    release_bb_result(&result);

    return 0;
}
#endif

// -----------------------------------------------

static void** allocate_matrix(size_t rows, size_t cols, size_t elem_size) {
    void** array = (void**)malloc(sizeof(void*) * rows);
    if (!array) return NULL;

    array[0] = malloc(rows * cols * elem_size); // CHECK: Should we use aligned_allocate instead?
    if (!array[0]) {
        free(array);
        return NULL;
    }

    for (size_t i = 1; i < rows; i++) {
        array[i] = (uint8_t*)array[0] + i * cols * elem_size;
    }

    if (((uintptr_t)&array[0]) % 16 != 0) {
        printf("ERROR: matrix is not 16-byte aligned! Address: %p\n", (void*)&array[0]);
    }

    return array;
}

static void release_matrix(void** matrix) {
    if (matrix && matrix[0]) free(matrix[0]);
    if (matrix) free(matrix);
}

static void transpose_double_matrix(size_t rows, size_t cols, double** mat, double** mat_T) {
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            mat_T[col][row] = mat[row][col];
        }
    }
}

// -----------------------------------------------

#define NUMBER_ELEMENT_DOF  1

#define TOR 1e-8 // Tolerance for convergence

#define MAX_STEPS 1000 // Maximum number of iterations

static bb_status_t read_input_from_file(const char* filename, bb_input_t* input) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return BB_ERR_FILE_OPEN;
    }

    int success = 0;

    // Read the number of nodes from an input data file: nond  
    if (fscanf(fp, "%d", &input->nond) != 1) goto fail;

    // Allocation for the array for the coordinates of the nodes 
    input->np = (vector3_t*)malloc(sizeof(vector3_t) * input->nond);
    if (!input->np) goto bad_alloc;

    // Read the coordinates of the nodes from an input data file: np  
    for (int i = 0; i < input->nond; i++) {
        if (fscanf(fp, "%lf %lf %lf", &(input->np[i].x), &(input->np[i].y), &(input->np[i].z)) != 3) {
            goto fail;
        }
    }

    // Read the number of faces from an input data file: nofc  
    if (fscanf(fp, "%d", &input->nofc) != 1) goto fail;

    // Read the number of nodes on each face from an input data file: nond_on_face  
    if (fscanf(fp, "%d", &input->nond_on_face) != 1) goto fail;

    // Read the number of integer parameters set on each face from an input data file: nint_para_fc  
    if (fscanf(fp, "%d", &input->nint_para_fc) != 1) goto fail;

    // Read the number of real(double precision) parameters set on each face from an input data file: ndble_para_fc  
    if (fscanf(fp, "%d", &input->ndble_para_fc) != 1) goto fail;

    // Read the number of parameters set on each face from an input data file: para_batch
    if (fscanf(fp, "%d", &input->para_batch) != 1) goto fail;

    printf("Number of nodes=%d Number of faces=%d\n", input->nond, input->nofc);

    // -----------------------------------------------

    // face2node
    input->face2node = (int**)allocate_matrix(input->nofc, input->nond_on_face, sizeof(int));
    if (!input->face2node) goto bad_alloc;

    for (int i = 0; i < input->nofc; i++) {
        for (int j = 0; j < input->nond_on_face; j++) {
            if (fscanf(fp, "%d", &input->face2node[i][j]) != 1) goto fail;
        }
    }

    // int_para_fc
    if (input->nint_para_fc > 0) {
        input->int_para_fc = (int***)malloc(sizeof(int**) * input->para_batch);
        for (int n = 0; n < input->para_batch; n++) {
            input->int_para_fc[n] = (int**)allocate_matrix(input->nofc, input->nint_para_fc, sizeof(int));
            if (!input->int_para_fc[n]) goto bad_alloc;

            for (int i = 0; i < input->nofc; i++) {
                for (int j = 0; j < input->nint_para_fc; j++) {
                    if (fscanf(fp, "%d", &input->int_para_fc[n][i][j]) != 1) goto fail;
                }
            }
        }
    }

    // dble_para_fc
    if (input->ndble_para_fc > 0) {
        input->dble_para_fc = (double***)malloc(sizeof(double**) * input->para_batch);
        for (int n = 0; n < input->para_batch; n++) {
            input->dble_para_fc[n] = (double**)allocate_matrix(input->nofc, input->ndble_para_fc, sizeof(double));
            if (!input->dble_para_fc[n]) goto bad_alloc;

            for (int i = 0; i < input->nofc; i++) {
                for (int j = 0; j < input->ndble_para_fc; j++) {
                    if (fscanf(fp, "%lf", &input->dble_para_fc[n][i][j]) != 1) goto fail;
                }
            }
        }
    }

    success = 1;

fail:
    if (!success) {
        fclose(fp);
        printf("Error: Invalid file format %s\n", filename);
        return BB_ERR_FILE_FORMAT;
    }

bad_alloc:
    if (!success) {
        fclose(fp);
        printf("Error: Out of memory while reading %s\n", filename);
        return BB_ERR_FILE_OPEN;
    }

    fclose(fp);
    return BB_OK;
}

// -----------------------------------------------

static bool has_stl_extension(const char* filename) {
    const char* dot = strrchr(filename, '.');
    return dot != NULL && strcmp(dot, ".stl") == 0;
}

static bb_status_t read_input_from_stl(const char* filename, bb_input_t* input) {
    stl_model_t model;
    if (!load_stl_ascii(filename, &model)) {
        printf("Error: Cannot open file %s\n", filename);
        return BB_ERR_FILE_OPEN;
    }

    int success = 0;

    // Read the number of nodes from an input data file: nond

    input->nond = model.num_facets * 3;

    // Allocation for the array for the coordinates of the nodes 
    input->np = (vector3_t*)malloc(sizeof(vector3_t) * input->nond);
    if (!input->np) goto bad_alloc;

    // Read the coordinates of the nodes from an input data file: np  
    for (int i = 0; i < model.num_facets; i++) {
        for (int j = 0; j < 3; j++) {
            input->np[i * 3 + j].x = model.facets[i].v[j].x;
            input->np[i * 3 + j].y = model.facets[i].v[j].y;
            input->np[i * 3 + j].z = model.facets[i].v[j].z;
        }
    }

    // Set value: nofc
    input->nofc = model.num_facets;

    // Set value: nond_on_face  
    input->nond_on_face = 3;

    // Set value: nint_para_fc
    input->nint_para_fc = 0;

    // Set value: ndble_para_fc
    input->ndble_para_fc = 0;

    // Set value: para_batch
    input->para_batch = 16;

    printf("Number of nodes=%d Number of faces=%d\n", input->nond, input->nofc);

    // -----------------------------------------------

    // face2node
    input->face2node = (int**)allocate_matrix(input->nofc, input->nond_on_face, sizeof(int));
    if (!input->face2node) goto bad_alloc;

    for (int i = 0; i < input->nofc; i++) {
        for (int j = 0; j < input->nond_on_face; j++) {
            input->face2node[i][j] = i * 3 + j;
        }
    }

    // int_para_fc
    // no

    // dble_para_fc
    // no

    success = 1;

    // fail:
    //     if (!success) {
    //         free_stl_model(&model);
    //         printf("Error: Invalid file format %s\n", filename);
    //         return BB_ERR_FILE_FORMAT;
    //     }

bad_alloc:
    if (!success) {
        free_stl_model(&model);
        printf("Error: Out of memory while reading %s\n", filename);
        return BB_ERR_FILE_OPEN;
    }

    free_stl_model(&model);
    return BB_OK;
}

// -----------------------------------------------

bb_status_t bb_bem(const char* filename, bb_compute_t /* in */ compute, bb_result_t* result) {
    bb_input_t* input = &result->input;
    *input = (bb_input_t){0}; // Initialize input structure

    bb_status_t input_status;
    if (has_stl_extension(filename)) {
        input_status = read_input_from_stl(filename, input);
    } else {
        input_status = read_input_from_file(filename, input);
    }

    if (input_status != BB_OK) {
        return input_status;
    }

    // -----------------------------------------------

    result->dim = input->nofc * NUMBER_ELEMENT_DOF;

    printf("Matrix size: %dx%d\n", result->dim, result->dim);

    double** A = (double**)allocate_matrix(result->dim, result->dim, sizeof(double));
    if (!A) return BB_ERR_MEMORY_ALLOC;

    printf("RHS vector size: %dx%d\n", result->dim, input->para_batch);

    double** rhs = (double**)allocate_matrix(result->dim, input->para_batch, sizeof(double));
    if (!rhs) return BB_ERR_MEMORY_ALLOC;

    result->sol = (double**)allocate_matrix(result->dim, input->para_batch, sizeof(double));
    if (!result->sol) return BB_ERR_MEMORY_ALLOC;

    for (int i = 0; i < result->dim; i++) {
        for (int n = 0; n < input->para_batch; n++) {
            result->sol[i][n] = 0.0;
        }
    }

    // User Specified Function 
    // element_integral(coordinate np, double **a, );

    bb_props_t props;
    props.nofc = input->nofc;
    props.nond = input->nond;
    props.nond_on_face = input->nond_on_face;
    props.para_batch = input->para_batch;
    props.np = &input->np[0];
    props.face2node = &input->face2node[0][0];

    for (int i = 0; i < result->dim; i++) {
        for (int j = 0; j < result->dim; j++) {
            A[i][j] = element_ij_(&i, &j, &props);
        }
    }

    for (int i = 0; i < result->dim; i++) {
        for (int n = 0; n < input->para_batch; n++) {
            const int* int_para_fc = input->int_para_fc ? &input->int_para_fc[n][0][0] : NULL;
            const double* dble_para_fc = input->dble_para_fc ? &input->dble_para_fc[n][0][0] : NULL;
            rhs[i][n] =
                rhs_vector_i_(&i, &n, &input->nint_para_fc, int_para_fc, &input->ndble_para_fc, dble_para_fc, &props);
        }
    }

    printf("Linear system was generated.\n");

    const clock_t compute_start = clock(); // <-- Start time measurement

#if 0
    // pass the rhs vector through the solution (for debugging)
    {
        for (int i = 0; i < result->dim; i++) {
            for (int n = 0; n < input->para_batch; n++) {
                result->sol[i][n] = rhs[i][n];
            }
        }
    }
#else
    if (compute == BB_COMPUTE_NAIVE) {
        bicgstab_naive(input->para_batch, result->dim, A, rhs, result->sol, TOR, MAX_STEPS);
    } else if (compute == BB_COMPUTE_CUDA) {
        bicgstab_cuda(input->para_batch, result->dim, A, rhs, result->sol, TOR, MAX_STEPS);
    } else if (compute == BB_COMPUTE_CUDA_WMMA) {
        bicgstab_cuda_wmma(input->para_batch, result->dim, A, rhs, result->sol, TOR, MAX_STEPS);
    } else {
        printf("Error: Unknown compute type\n");
    }
#endif

    result->compute_time = (double)(clock() - compute_start) / CLOCKS_PER_SEC; // <-- End time measurement

    printf("Completed\n");

    release_matrix(A);

    release_matrix(rhs);

    return BB_OK;
}

void release_bb_result(bb_result_t* result) {
    if (!result) return;
    // -----------------------------------------------

    bb_input_t* input = &result->input;
    if (!input) return;
    // -----------------------------------------------

    if (input->np) free(input->np);

    if (input->face2node) {
        release_matrix(input->face2node);
    }

    if (input->int_para_fc) {
        for (int n = 0; n < input->para_batch; n++) { release_matrix(input->int_para_fc[n]); }

        free(input->int_para_fc);
    }

    if (input->dble_para_fc) {
        for (int n = 0; n < input->para_batch; n++) { release_matrix(input->dble_para_fc[n]); }

        free(input->dble_para_fc);
    }

    if (result->sol) release_matrix(result->sol);

    input->np = NULL;
    input->face2node = NULL;
    input->int_para_fc = NULL;
    input->dble_para_fc = NULL;
    result->sol = NULL;
}
