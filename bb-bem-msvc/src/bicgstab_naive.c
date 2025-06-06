#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "bicgstab_naive.h"

// TODO: Should we attach 'restrict' keyword?

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
    int dim,
    double** A /* in [dim][dim] */,
    double* b /* in [dim] */,
    double* x /* out [dim] */,
    double tor,
    int max_steps
) {
    // Initialization

    double* p = malloc(sizeof(double) * dim); // <-- Allocation: p

    for (int i = 0; i < dim; i++) {
        p[i] = 0.0;
    }

    double bnorm = sqrt(dot_product(dim, b, b));

    // Initial residual
    double* r = (double*)malloc(sizeof(double) * dim); // <-- Allocation: r
    residual(dim, A, x, b, r);

    // Set shadow vector
    double* r0 = (double*)malloc(sizeof(double) * dim); // <-- Allocation: r0
    for (int i = 0; i < dim; i++) { r0[i] = r[i]; }

    double rnorm = sqrt(dot_product(dim, r, r));

    // Allocation of arrays
    double* t = malloc(dim * sizeof(double)); // <-- Allocation: t 
    double* Ap = malloc(dim * sizeof(double)); // <-- Allocation: Ap 
    double* Akp = malloc(dim * sizeof(double)); // <-- Allocation: Akp  
    double* kt = malloc(dim * sizeof(double)); // <-- Allocation: kt 
    double* Akt = malloc(dim * sizeof(double)); // <-- Allocation: Akt 
    double* kp = malloc(dim * sizeof(double)); // <-- Allocation: kp

    double nom = 0;
    double nom_old = 0;
    double den = 0;
    double alpha = 0;
    double beta = 0;
    double zeta = 0;

    double tmp;

    tmp = rnorm / bnorm;
    printf("Original relative residual norm = %20.14e\n", tmp);
    if (tmp < tor) { goto finalize; }

    // BiCGSTAB iteration 
    for (int step = 1; step <= max_steps; step++) {
        matvec(dim, A, p, Ap);

        for (int i = 0; i < dim; i++) { p[i] = r[i] + beta * (p[i] - zeta * Ap[i]); }

        // No preconditioning 

        for (int i = 0; i < dim; i++) { kp[i] = p[i]; }

        matvec(dim, A, kp, Akp);

        nom = dot_product(dim, r0, r);

        den = dot_product(dim, r0, Akp);

        alpha = nom / den;

        nom_old = nom;

        for (int i = 0; i < dim; i++) { t[i] = r[i] - alpha * Akp[i]; }

        // No preconditioning 
        for (int i = 0; i < dim; i++) { kt[i] = t[i]; }

        matvec(dim, A, kt, Akt);

        nom = dot_product(dim, Akt, t);

        den = dot_product(dim, Akt, Akt);

        zeta = nom / den;

        for (int i = 0; i < dim; i++) { x[i] = x[i] + alpha * kp[i] + zeta * kt[i]; }

        for (int i = 0; i < dim; i++) { r[i] = t[i] - zeta * Akt[i]; }

        beta = alpha / zeta * dot_product(dim, r0, r) / nom_old;

        rnorm = sqrt(dot_product(dim, r, r));

        tmp = rnorm / bnorm;
        printf("  Step %d relative residual norm = %20.14e \n", step, tmp);
        if (tmp < tor) { break; }
    }
    // -----------------------------------------------

    // Confirmation of residual 

    residual(dim, A, x, b, r);

    rnorm = sqrt(dot_product(dim, r, r));

    printf("Relative residual norm = %20.14e \n", rnorm / bnorm);

    // -----------------------------------------------

finalize:
    if (kp) free(kp);
    if (Akt) free(Akt);
    if (kt) free(kt);
    if (Akp) free(Akp);
    if (Ap) free(Ap);
    if (t) free(t);

    if (r0) free(r0);

    if (r) free(r);

    if (p) free(p);
}
