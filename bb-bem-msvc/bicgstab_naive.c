#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "bicgstab_naive.h"

static void** allocate_matrix(size_t rows, size_t cols, size_t elem_size) {
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

// -----------------------------------------------

// Matrix vector multiplication with a dense matrix: q = A p 
static void matvec(int dim, double** A, const double* p, double* q /* out */) {
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
    double** Q /* out [dim][batch] */
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
static void residual(int dim, double** A, const double* x, const double* b, double* r /* out */) {
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
    double** R /* out [dim][batch] */
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
    double* out /* out */
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

static void batch_sqrt(int batch, const double* x, double* out /* out */) {
    for (int n = 0; n < batch; ++n) {
        out[n] = sqrt(x[n]);
    }
}

// x * y
static void batch_mul(int batch, const double* x, const double* y, double* out /* out */) {
    for (int n = 0; n < batch; ++n) {
        out[n] = x[n] * y[n];
    }
}

// x / y
static void batch_div(int batch, const double* x, const double* y, double* out /* out */) {
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
void bicgstab_naive(
    int batch,
    int dim,
    double** A /* in [dim][dim] */,
    double** b /* in [dim][batch] */,
    double** x /* out [dim][batch] */,
    double tor,
    int max_steps
) {
    // Initialization

    double** p = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: P
    for (int i = 0; i < dim; i++) {
        for (int n = 0; n < batch; n++) {
            p[i][n] = 0.0;
        }
    }

    double* bnorm = (double*)malloc(sizeof(double) * batch); // <-- Allocation: bnorm
    // bnorm = sqrt(dot_product(dim, b, b));
    batch_dot_product(batch, dim, b, b, bnorm); // dot_product(dim, b, b)
    batch_sqrt(batch, bnorm, bnorm); // sqrt(dot_product(dim, b, b))

    // Initial residual
    double** r = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: r
    // residual(dim, A, x, b, r);
    batch_residual(batch, dim, A, x, b, r);

    // Set shadow vector
    double** r0 = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: r0
    // for (int i = 0; i < dim; i++) { r0[i] = r[i]; }
    for (int i = 0; i < dim; i++) {
        for (int n = 0; n < batch; n++) {
            r0[i][n] = r[i][n];
        }
    }

    double* rnorm = malloc(sizeof(double) * batch); // <-- Allocation: rnorm
    // rnorm = sqrt(dot_product(dim, r, r));
    batch_dot_product(batch, dim, r, r, rnorm);
    batch_sqrt(batch, rnorm, rnorm);

    printf("Original relative residual norm [0] = %20.14e\n", rnorm[0] / bnorm[0]);

    // Allocation of arrays
    double** t = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: t 
    double** Ap = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: Ap 
    double** Akp = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: Akp  
    double** kt = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: kt 
    double** Akt = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: Akt 
    double** kp = (double**)allocate_matrix(dim, batch, sizeof(double)); // <-- Allocation: kp

    double* nom = malloc(sizeof(double) * batch); // <-- Allocation: nom
    double* nom_old = malloc(sizeof(double) * batch); // <-- Allocation: nom_old
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
        // matvec(dim, A, p, Ap);
        batch_matvec(batch, dim, A, p, Ap);

        // for (int i = 0; i < dim; i++) { p[i] = r[i] + beta * (p[i] - zeta * Ap[i]); }
        for (int i = 0; i < dim; i++) {
            for (int n = 0; n < batch; n++) {
                p[i][n] = r[i][n] + beta[n] * (p[i][n] - zeta[n] * Ap[i][n]);
            }
        }

        // No preconditioning 

        // for (int i = 0; i < dim; i++) { kp[i] = p[i]; }
        for (int i = 0; i < dim; i++) {
            for (int n = 0; n < batch; n++) {
                kp[i][n] = p[i][n];
            }
        }

        // matvec(dim, A, kp, Akp);
        batch_matvec(batch, dim, A, kp, Akp);

        // nom = dot_product(dim, r0, r);
        batch_dot_product(batch, dim, r0, r, nom);

        // den = dot_product(dim, r0, Akp);
        batch_dot_product(batch, dim, r0, Akp, den);

        // alpha = nom / den;
        batch_div(batch, nom, den, alpha);

        // nom_old = nom;
        for (int i = 0; i < batch; i++) { nom_old[i] = nom[i]; }

        // for (int i = 0; i < dim; i++) { t[i] = r[i] - alpha * Akp[i]; }
        for (int i = 0; i < dim; i++) {
            for (int n = 0; n < batch; n++) {
                t[i][n] = r[i][n] - alpha[n] * Akp[i][n];
            }
        }

        // No preconditioning 
        // for (int i = 0; i < dim; i++) { kt[i] = t[i]; }
        for (int i = 0; i < dim; i++) {
            for (int n = 0; n < batch; n++) {
                kt[i][n] = t[i][n];
            }
        }

        // matvec(dim, A, kt, Akt);
        batch_matvec(batch, dim, A, kt, Akt);

        // nom = dot_product(dim, Akt, t);
        batch_dot_product(batch, dim, Akt, t, nom);

        // den = dot_product(dim, Akt, Akt);
        batch_dot_product(batch, dim, Akt, Akt, den);

        // zeta = nom / den;
        batch_div(batch, nom, den, zeta);

        // for (int i = 0; i < dim; i++) { x[i] = x[i] + alpha * kp[i] + zeta * kt[i]; }
        for (int i = 0; i < dim; i++) {
            for (int n = 0; n < batch; n++) {
                x[i][n] += alpha[n] * kp[i][n] + zeta[n] * kt[i][n];
            }
        }

        // for (int i = 0; i < dim; i++) { r[i] = t[i] - zeta * Akt[i]; }
        for (int i = 0; i < dim; i++) {
            for (int n = 0; n < batch; n++) {
                r[i][n] = t[i][n] - zeta[n] * Akt[i][n];
            }
        }

        // beta = alpha / zeta * dot_product(dim, r0, r) / nom_old;
        batch_dot_product(batch, dim, r0, r, beta); // dot_product(dim, r0, r)
        batch_mul(batch, beta, alpha, beta); // alpha * dot_product(dim, r0, r)
        batch_div(batch, beta, zeta, beta); // alpha / zeta * dot_product(dim, r0, r)
        batch_div(batch, beta, nom_old, beta); // alpha / zeta * dot_product(dim, r0, r) / nom_old

        // rnorm = sqrt(dot_product(dim, r, r));
        batch_dot_product(batch, dim, r, r, rnorm); // dot_product(dim, r, r)
        batch_sqrt(batch, rnorm, rnorm); // sqrt(dot_product(dim, r, r))

        printf("  Step %d relative residual norm [0] = %20.14e \n", step, rnorm[0] / bnorm[0]);

        // if (rnorm / bnorm < tor) { break; }
        batch_div(batch, rnorm, bnorm, tmp);
        if (batch_lt(batch, tmp, tor)) { break; }
    }
    // -----------------------------------------------

    // Confirmation of residual 

    // residual(dim, A, x, b, r);
    batch_residual(batch, dim, A, x, b, r);

    // rnorm = sqrt(dot_product(dim, r, r));
    batch_dot_product(batch, dim, r, r, rnorm); // dot_product(dim, r, r)
    batch_sqrt(batch, rnorm, rnorm); // sqrt(dot_product(dim, r, r))

    printf("Relative residual norm [0] = %20.14e \n", rnorm[0] / bnorm[0]);

    // -----------------------------------------------

release:
    if (tmp) free(tmp);

    if (zeta) free(zeta);
    if (beta) free(beta);
    if (alpha) free(alpha);
    if (den) free(den);
    if (nom_old) free(nom_old);
    if (nom) free(nom);

    if (kp) release_matrix(kp);
    if (Akt) release_matrix(Akt);
    if (kt) release_matrix(kt);
    if (Akp) release_matrix(Akp);
    if (Ap) release_matrix(Ap);
    if (t) release_matrix(t);

    if (rnorm) free(rnorm);

    if (r0) release_matrix(r0);

    if (r) release_matrix(r);

    if (bnorm) free(bnorm);

    if (p) release_matrix(p);
}
