#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bb_bem.h"

static void pbicgstab(int dim, double** mat, double* rhs, double* sol, double tor, int max_steps);

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

#define NUMBER_ELEMENT_DOF  1;

bb_status_t bb_bem(char* filename, bb_result_t* result) {
    bb_input_t* input = &result->input;
    *input = (bb_input_t){0}; // Initialize input structure

    int i, j;

    FILE* fp;

    double** a;
    double *rhs, tor;
    int max_steps;

    fp = fopen(filename, "r");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return BB_ERR_FILE_OPEN;
    }

    // Read number of nodes from input data file : nond  
    fscanf(fp, "%d", &input->nond);

    // Allocation for the array for the coordinates of the nodes 
    input->np = (vector3_t*)malloc(sizeof(vector3_t) * input->nond);

    // Read the coordinates of the nodes from input data file : np  
    for (i = 0; i < input->nond; i++) {
        fscanf(fp, "%lf %lf %lf", &(input->np[i].x), &(input->np[i].y), &(input->np[i].z));
    }

    // Read number of faces from input data file : nofc  
    fscanf(fp, "%d", &input->nofc);

    // Read number of nodes on each face from input data file : nond_on_face  
    fscanf(fp, "%d", &input->nond_on_face);

    // Read number of integer parameters set on each face from input data file : nint_para_fc  
    fscanf(fp, "%d", &input->nint_para_fc);

    // Read number of real(double precision) parameters set on each face from input data file : ndble_para_fc  
    fscanf(fp, "%d", &input->ndble_para_fc);

    printf("Number of nodes=%d Number of faces=%d\n", input->nond, input->nofc);

    // -----------------------------------------------

    input->face2node = (int**)malloc(sizeof(int*) * input->nofc);
    input->face2node[0] = (int*)malloc(sizeof(int) * input->nofc * input->nond_on_face);
    for (i = 1; i < input->nofc; i++) {
        input->face2node[i] = input->face2node[i - 1] + input->nond_on_face;
    }
    for (i = 0; i < input->nofc; i++) {
        for (j = 0; j < input->nond_on_face; j++) {
            fscanf(fp, "%d", &(input->face2node[i][j]));
        }
    }

    if (input->nint_para_fc > 0) {
        input->int_para_fc = (int**)malloc(sizeof(int*) * input->nofc);
        input->int_para_fc[0] = (int*)malloc(sizeof(int) * input->nofc * input->nint_para_fc);
        for (i = 1; i < input->nofc; i++) {
            input->int_para_fc[i] = input->int_para_fc[i - 1] + input->nint_para_fc;
        }

        for (i = 0; i < input->nofc; i++) {
            for (j = 0; j < input->nint_para_fc; j++) {
                fscanf(fp, "%d", &input->int_para_fc[i][j]);
            }
        }
    }

    if (input->ndble_para_fc > 0) {
        input->dble_para_fc = (double**)malloc(sizeof(double*) * input->nofc);
        input->dble_para_fc[0] = (double*)malloc(sizeof(double) * input->nofc * input->ndble_para_fc);
        for (i = 1; i < input->nofc; i++) {
            input->dble_para_fc[i] = input->dble_para_fc[i - 1] + input->ndble_para_fc;
        }

        for (i = 0; i < input->nofc; i++) {
            for (j = 0; j < input->ndble_para_fc; j++) {
                fscanf(fp, "%lf", &(input->dble_para_fc[i][j]));
            }
        }
    }

    result->dim = input->nofc * NUMBER_ELEMENT_DOF;

    a = (double**)malloc(sizeof(double*) * result->dim);
    a[0] = (double*)malloc(sizeof(double) * result->dim * result->dim);
    for (i = 1; i < result->dim; i++) {
        a[i] = a[i - 1] + result->dim;
    }

    for (i = 0; i < result->dim; i++) {
        for (j = 0; j < result->dim; j++) {
            a[i][j] = 0.0;
        }
    }

    tor = 1E-8;
    max_steps = 1000;

    rhs = (double*)malloc(sizeof(double) * result->dim);
    result->sol = (double*)malloc(sizeof(double) * result->dim);

    for (i = 0; i < result->dim; i++) {
        result->sol[i] = 0.0;
    }

    // User Specified Function 
    // element_integral(coordinate np, double **a, ); 

    for (i = 0; i < result->dim; i++) {
        for (j = 0; j < result->dim; j++) {
            a[i][j] = element_ij_(&i, &j, &input->nond, &input->nofc, &input->np[0], &input->face2node[0][0]);
        }
    }

    for (i = 0; i < result->dim; i++) {
        rhs[i] = input->dble_para_fc[i][0];
    }

    fclose(fp);

    printf("Linear system was generated.\n");

    pbicgstab(result->dim, a, rhs, result->sol, tor, max_steps);

    // printf("%d,%d\n",nint_para_fc,ndble_para_fc); 

    printf("OK\n");

    free(a[0]);
    free(a);

    free(rhs);

    return BB_SUCCESS;
}

void release_bb_result(bb_result_t* result) {
    if (!result) return;

    bb_input_t* input = &result->input;

    free(input->np);

    if (input->face2node) {
        free(input->face2node[0]); // This also frees memory in [1]-[^1]
        free(input->face2node);
    }

    if (input->int_para_fc) {
        free(input->int_para_fc[0]); // This also frees memory in [1]-[^1]
        free(input->int_para_fc);
    }

    if (input->dble_para_fc) {
        free(input->dble_para_fc[0]); // This also frees memory in [1]-[^1]
        free(input->dble_para_fc);
    }

    free(result->sol);

    input->np = NULL;
    input->face2node = NULL;
    input->int_para_fc = NULL;
    input->dble_para_fc = NULL;
    result->sol = NULL;
}

// Matrix vector multiplication with a dense matrix: q=Ap 
static void matvec_direct(int dim, double** mat, double* p, double* q) {
    int row, col;

    for (row = 0; row < dim; row++) {
        q[row] = 0.0;
    }

    for (row = 0; row < dim; row++) {
        for (col = 0; col < dim; col++) {
            q[row] = q[row] + mat[row][col] * p[col];
        }
    }
}

// -----------------------------------------------
// Calculation of residual matrix with a dense matrix: r=b-Ax 
static void residual_direct(int dim, double** mat, double* x, double* b, double* r) {
    int row, col;

    for (row = 0; row < dim; row++) {
        r[row] = b[row];
    }

    for (row = 0; row < dim; row++) {
        for (col = 0; col < dim; col++) {
            r[row] = r[row] - mat[row][col] * x[col];
        }
    }
}

// -----------------------------------------------
// Calculation of dot product 
static double dot_product(int dim, double* x, double* y) {
    double sum = 0;
    int i;

    for (i = 0; i < dim; i++) {
        sum = sum + x[i] * y[i];
    }

    return sum;
}

// -----------------------------------------------
static void pbicgstab(int dim, double** mat, double* rhs, double* sol, double tor, int max_steps) {
    int step, i;
    double *r, *shdw, *p, *t, *ap, *kp, *akp, *kt, *akt;
    double alpha, beta, zeta, nom, den, nomold, rnorm, bnorm;

    // Allocation of arraies 
    r = (double*)malloc(sizeof(double) * dim);
    shdw = (double*)malloc(sizeof(double) * dim);
    p = (double*)malloc(sizeof(double) * dim);
    t = (double*)malloc(sizeof(double) * dim);
    ap = (double*)malloc(sizeof(double) * dim);
    kp = (double*)malloc(sizeof(double) * dim);
    akp = (double*)malloc(sizeof(double) * dim);
    kt = (double*)malloc(sizeof(double) * dim);
    akt = (double*)malloc(sizeof(double) * dim);

    // Initialization
    for (i = 0; i < dim; i++) { p[i] = 0.0; }
    alpha = 0.0;
    beta = 0.0;
    zeta = 0.0;

    bnorm = sqrt(dot_product(dim, rhs, rhs));

    // Initial residual   
    residual_direct(dim, mat, sol, rhs, r);

    // Set shadow vector 
    for (i = 0; i < dim; i++) { shdw[i] = r[i]; }

    rnorm = sqrt(dot_product(dim, r, r));
    printf("Original relative residual norm = %20.14e\n", rnorm / bnorm);

    if (rnorm / bnorm < tor) { return; }

    // BiCGSTAB iteration 
    for (step = 1; step <= max_steps; step++) {
        matvec_direct(dim, mat, p, ap);

        for (i = 0; i < dim; i++) { p[i] = r[i] + beta * (p[i] - zeta * ap[i]); }

        // No preconditioning 
        for (i = 0; i < dim; i++) { kp[i] = p[i]; }

        matvec_direct(dim, mat, kp, akp);

        nom = dot_product(dim, shdw, r);
        den = dot_product(dim, shdw, akp);
        alpha = nom / den;
        nomold = nom;

        // printf("alpha= %lf",alpha); 

        for (i = 0; i < dim; i++) { t[i] = r[i] - alpha * akp[i]; }

        // No preconditioning 
        for (i = 0; i < dim; i++) { kt[i] = t[i]; }

        matvec_direct(dim, mat, kt, akt);

        nom = dot_product(dim, akt, t);
        den = dot_product(dim, akt, akt);
        zeta = nom / den;

        for (i = 0; i < dim; i++) { sol[i] = sol[i] + alpha * kp[i] + zeta * kt[i]; }
        for (i = 0; i < dim; i++) { r[i] = t[i] - zeta * akt[i]; }
        beta = alpha / zeta * dot_product(dim, shdw, r) / nomold;

        rnorm = sqrt(dot_product(dim, r, r));
        printf("  Step %d relative residual norm = %20.14e \n", step, rnorm / bnorm);

        if (rnorm / bnorm < tor) { break; }
    }
    // -----------------------------------------------

    // Confirmation of residual 

    residual_direct(dim, mat, sol, rhs, r);
    rnorm = sqrt(dot_product(dim, r, r));

    printf("Relative residual norm = %20.14e \n", rnorm / bnorm);

    free(r);
    free(shdw);
    free(p);
    free(t);
    free(ap);
    free(kp);
    free(akp);
    free(kt);
    free(akt);

    return;
}
