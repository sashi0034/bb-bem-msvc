#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "bb_bem.h"

static void pbicgstab(
    int batch,
    int dim,
    double** A /* [dim][dim] */,
    double** /* [dim][batch] */ rhs,
    double** /* [dim][batch] */ sol,
    double tor,
    int max_steps
);

#if !defined(BB_NO_MAIN)
int main() {
    bb_result_t result;
    bb_bem("input.txt", &result);

    // ----------------------------------------------- fp

    FILE* fp = fopen("out2.data", "w");
    for (int i = 0; i < result.dim; i++) {
        fprintf(fp, "%20.14e \n", result.sol[i]);
    }

    fclose(fp);

    // -----------------------------------------------

    release_bb_result(&result);

    return 0;
}
#endif

// -----------------------------------------------

#define NUMBER_ELEMENT_DOF  1;

#define TOR 1e-8 // Tolerance for convergence

#define MAX_STEPS 1000 // Maximum number of iterations

void** allocate_matrix(size_t rows, size_t cols, size_t elem_size) {
    void** array = (void**)malloc(sizeof(void*) * rows);
    if (!array) return NULL;

    array[0] = malloc(rows * cols * elem_size);
    if (!array[0]) {
        free(array);
        return NULL;
    }

    for (size_t i = 1; i < rows; i++) {
        array[i] = (uint8_t*)array[0] + i * cols * elem_size;
    }

    return array;
}

static void release_matrix(void** matrix) {
    if (matrix && matrix[0]) free(matrix[0]);
    if (matrix) free(matrix);
}

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
    if (input->np) {
        fclose(fp);
        printf("Error: Out of memory while reading %s\n", filename);
        return BB_ERR_FILE_OPEN;
    }

    fclose(fp);
    return BB_SUCCESS;
}

bb_status_t bb_bem(const char* filename, bb_result_t* result) {
    bb_input_t* input = &result->input;
    *input = (bb_input_t){0}; // Initialize input structure

    read_input_from_file(filename, input);

    // -----------------------------------------------

    result->dim = input->nofc * NUMBER_ELEMENT_DOF;

    double** A = (double**)allocate_matrix(result->dim, result->dim, sizeof(double));
    if (!A) return BB_ERR_MEMORY_ALLOC;

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

    for (int i = 0; i < result->dim; i++) {
        for (int j = 0; j < result->dim; j++) {
            A[i][j] = element_ij_(&i, &j, &input->nond, &input->nofc, &input->np[0], &input->face2node[0][0]);
        }
    }

    for (int i = 0; i < result->dim; i++) {
        for (int n = 0; n < input->para_batch; n++) {
            rhs[i][n] = input->dble_para_fc[n][i][0]; // TODO
        }
    }

    printf("Linear system was generated.\n");

    pbicgstab(input->para_batch, result->dim, A, rhs, result->sol, TOR, MAX_STEPS);

    // printf("%d,%d\n",nint_para_fc,ndble_para_fc); 

    printf("OK\n");

    release_matrix(A);

    release_matrix(rhs);

    return BB_SUCCESS;
}

void release_bb_result(bb_result_t* result) {
    if (!result) return;

    bb_input_t* input = &result->input;

    free(input->np);

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

    release_matrix(result->sol);

    input->np = NULL;
    input->face2node = NULL;
    input->int_para_fc = NULL;
    input->dble_para_fc = NULL;
    result->sol = NULL;
}

// -----------------------------------------------

// Matrix vector multiplication with a dense matrix: q = A p 
static void matvec(int dim, double** A, const double* p, double* q /** @out */) {
    for (int row = 0; row < dim; row++) {
        q[row] = 0.0;
    }

    for (int row = 0; row < dim; row++) {
        for (int col = 0; col < dim; col++) {
            q[row] = q[row] + A[row][col] * p[col];
        }
    }
}

static void batch_matvec(
    int batch,
    int dim,
    double** mat /* [dim][dim] */,
    double** P /* [dim][batch] */,
    double** Q /** @out [dim][batch] */
) {
    for (int row = 0; row < dim; ++row) {
        for (int n = 0; n < batch; ++n) {
            Q[row][n] = 0.0;
        }
    }

    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            for (int n = 0; n < batch; ++n) {
                Q[row][n] += mat[row][col] * P[col][n];
            }
        }
    }
}

// Calculation of residual matrix with a dense matrix: r = b - A x 
static void residual(int dim, double** A, const double* x, const double* b, double* r /** @out */) {
    for (int row = 0; row < dim; row++) {
        r[row] = b[row];
    }

    for (int row = 0; row < dim; row++) {
        for (int col = 0; col < dim; col++) {
            r[row] = r[row] - A[row][col] * x[col];
        }
    }
}

static void batch_residual(
    int batch,
    int dim,
    double** A /* [dim][dim] */,
    double** X /* [dim][batch] */,
    double** B /* [dim][batch] */,
    double** R /** @out [dim][batch] */
) {
    for (int row = 0; row < dim; ++row) {
        for (int n = 0; n < batch; ++n) {
            R[row][n] = B[row][n];
        }
    }

    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            for (int n = 0; n < batch; ++n) {
                R[row][n] -= A[row][col] * X[col][n];
            }
        }
    }
}

static double dot_product(int dim, const double* x, const double* y) {
    double sum = 0;
    for (int i = 0; i < dim; i++) {
        sum = sum + x[i] * y[i];
    }

    return sum;
}

static void batch_dot_product(
    int batch,
    int dim,
    double** X /* [dim][batch] */ ,
    double** Y /* [dim][batch] */ ,
    double* out /** @out */
) {
    for (int n = 0; n < batch; ++n) {
        out[n] = 0.0;
    }

    for (int i = 0; i < dim; ++i) {
        for (int n = 0; n < batch; ++n) {
            out[n] += X[i][n] * Y[i][n];
        }
    }
}

static void batch_sqrt(int batch, const double* x, double* out /** @out */) {
    for (int n = 0; n < batch; ++n) {
        out[n] = sqrt(x[n]);
    }
}

// x * y
static void batch_mul(int batch, const double* x, const double* y, double* out /** @out */) {
    for (int n = 0; n < batch; ++n) {
        out[n] = x[n] * y[n];
    }
}

// x / y
static void batch_div(int batch, const double* x, const double* y, double* out /** @out */) {
    for (int n = 0; n < batch; ++n) {
        out[n] = x[n] / y[n];
    }
}

// x < y
static int batch_lt(int batch, const double* x, double y) {
    for (int n = 0; n < batch; ++n) {
        if (x[n] >= y) return 0;
    }

    return 1;
}

// -----------------------------------------------
static void pbicgstab(
    int batch,
    int dim,
    double** A /* [dim][dim] */,
    double** rhs /* [dim][batch] */,
    double** sol /* [dim][batch] */,
    double tor,
    int max_steps
) {
    // Initialization

    double** P = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: P
    for (int i = 0; i < dim; i++) {
        for (int n = 0; n < batch; n++) {
            P[i][n] = 0.0;
        }
    }

    double* bnorm = (double*)malloc(sizeof(double) * batch); // <-- Allocation: bnorm
    // bnorm = sqrt(dot_product(dim, rhs, rhs));
    batch_dot_product(batch, dim, rhs, rhs, bnorm); // dot_product(dim, rhs, rhs)
    batch_sqrt(batch, bnorm, bnorm); // sqrt(dot_product(dim, rhs, rhs))

    // Initial residual
    double** R = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: R
    // residual(dim, A, sol, rhs, r);
    batch_residual(batch, dim, A, sol, rhs, R);

    // Set shadow vector
    double** shdw = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: shdw
    // for (int i = 0; i < dim; i++) { shdw[i] = r[i]; }
    for (int i = 0; i < dim; i++) {
        for (int n = 0; n < batch; n++) {
            shdw[i][n] = R[i][n];
        }
    }

    double* rnorm = malloc(sizeof(double) * batch); // <-- Allocation: rnorm
    // rnorm = sqrt(dot_product(dim, r, r));
    batch_dot_product(batch, dim, R, R, rnorm);
    batch_sqrt(batch, rnorm, rnorm);

    printf("Original relative residual norm [0] = %20.14e\n", rnorm[0] / bnorm[0]);

    // Allocation of arrays
    double** t = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: t 
    double** ap = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: ap 
    double** akp = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: akp  
    double** kt = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: kt 
    double** akt = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: akt 
    double** kp = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: kp

    double* nom = malloc(sizeof(double) * batch); // <-- Allocation: nom
    double* nomold = malloc(sizeof(double) * batch); // <-- Allocation: nomold
    double* den = malloc(sizeof(double) * batch); // <-- Allocation: den
    double* alpha = malloc(sizeof(double) * batch); // <-- Allocation: alpha
    double* beta = malloc(sizeof(double) * batch); // <-- Allocation: beta
    double* zeta = malloc(sizeof(double) * batch); // <-- Allocation: zeta

    double* tmp = malloc(sizeof(double) * batch); // <-- Allocation: tmp

    // if (rnorm / bnorm < tor) {  goto release; }
    batch_div(batch, rnorm, bnorm, tmp);
    if (batch_lt(batch, tmp, tor)) { goto release; }

    for (int n = 0; n < batch; ++n) {
        alpha[n] = 0.0;
        beta[n] = 0.0;
        zeta[n] = 0.0;
    }

    // BiCGSTAB iteration 
    for (int step = 1; step <= max_steps; step++) {
        // matvec(dim, A, p, ap);
        batch_matvec(batch, dim, A, P, ap);

        // for (int i = 0; i < dim; i++) { p[i] = r[i] + beta * (p[i] - zeta * ap[i]); }
        for (int i = 0; i < dim; i++) {
            for (int n = 0; n < batch; n++) {
                P[i][n] = R[i][n] + beta[n] * (P[i][n] - zeta[n] * ap[i][n]);
            }
        }

        // No preconditioning 

        // for (int i = 0; i < dim; i++) { kp[i] = p[i]; }
        for (int i = 0; i < dim; i++) {
            for (int n = 0; n < batch; n++) {
                kp[i][n] = P[i][n];
            }
        }

        // matvec(dim, A, kp, akp);
        batch_matvec(batch, dim, A, kp, akp);

        // nom = dot_product(dim, shdw, r);
        batch_dot_product(batch, dim, shdw, R, nom);

        // den = dot_product(dim, shdw, akp);
        batch_dot_product(batch, dim, shdw, akp, den);

        // alpha = nom / den;
        batch_div(batch, nom, den, alpha);

        // nomold = nom;
        for (int i = 0; i < batch; i++) { nomold[i] = nom[i]; }

        // for (int i = 0; i < dim; i++) { t[i] = r[i] - alpha * akp[i]; }
        for (int i = 0; i < dim; i++) {
            for (int n = 0; n < batch; n++) {
                t[i][n] = R[i][n] - alpha[n] * akp[i][n];
            }
        }

        // No preconditioning 
        // for (int i = 0; i < dim; i++) { kt[i] = t[i]; }
        for (int i = 0; i < dim; i++) {
            for (int n = 0; n < batch; n++) {
                kt[i][n] = t[i][n];
            }
        }

        // matvec(dim, A, kt, akt);
        batch_matvec(batch, dim, A, kt, akt);

        // nom = dot_product(dim, akt, t);
        batch_dot_product(batch, dim, akt, t, nom);

        // den = dot_product(dim, akt, akt);
        batch_dot_product(batch, dim, akt, akt, den);

        // zeta = nom / den;
        batch_div(batch, nom, den, zeta);

        // for (int i = 0; i < dim; i++) { sol[i] = sol[i] + alpha * kp[i] + zeta * kt[i]; }
        for (int i = 0; i < dim; i++) {
            for (int n = 0; n < batch; n++) {
                sol[i][n] += alpha[n] * kp[i][n] + zeta[n] * kt[i][n];
            }
        }

        // for (int i = 0; i < dim; i++) { r[i] = t[i] - zeta * akt[i]; }
        for (int i = 0; i < dim; i++) {
            for (int n = 0; n < batch; n++) {
                R[i][n] = t[i][n] - zeta[n] * akt[i][n];
            }
        }

        // beta = alpha / zeta * dot_product(dim, shdw, r) / nomold;
        batch_dot_product(batch, dim, shdw, R, beta); // dot_product(dim, shdw, r)
        batch_mul(batch, beta, alpha, beta); // alpha * dot_product(dim, shdw, r)
        batch_div(batch, beta, zeta, beta); // alpha / zeta * dot_product(dim, shdw, r)
        batch_div(batch, beta, nomold, beta); // alpha / zeta * dot_product(dim, shdw, r) / nomold

        // rnorm = sqrt(dot_product(dim, r, r));
        batch_dot_product(batch, dim, R, R, rnorm); // dot_product(dim, r, r)
        batch_sqrt(batch, rnorm, rnorm); // sqrt(dot_product(dim, r, r))

        printf("  Step %d relative residual norm [0] = %20.14e \n", step, rnorm[0] / bnorm[0]);

        // if (rnorm / bnorm < tor) { break; }
        batch_div(batch, rnorm, bnorm, tmp);
        if (batch_lt(batch, tmp, tor)) { break; }
    }
    // -----------------------------------------------

    // Confirmation of residual 

    // residual(dim, A, sol, rhs, r);
    batch_residual(batch, dim, A, sol, rhs, R);

    // rnorm = sqrt(dot_product(dim, r, r));
    batch_dot_product(batch, dim, R, R, rnorm); // dot_product(dim, r, r)
    batch_sqrt(batch, rnorm, rnorm); // sqrt(dot_product(dim, r, r))

    printf("Relative residual norm [0] = %20.14e \n", rnorm[0] / bnorm[0]);

    // -----------------------------------------------

release:
    if (tmp) free(tmp);

    if (zeta) free(zeta);
    if (beta) free(beta);
    if (alpha) free(alpha);
    if (den) free(den);
    if (nomold) free(nomold);
    if (nom) free(nom);

    if (kp) release_matrix(kp);
    if (akt) release_matrix(akt);
    if (kt) release_matrix(kt);
    if (akp) release_matrix(akp);
    if (ap) release_matrix(ap);
    if (t) release_matrix(t);

    if (rnorm) free(rnorm);

    if (shdw) release_matrix(shdw);

    if (R) release_matrix(R);

    if (bnorm) free(bnorm);

    if (P) release_matrix(P);
}
