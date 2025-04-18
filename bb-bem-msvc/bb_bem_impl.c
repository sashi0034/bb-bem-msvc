#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bb_bem.h"

void pbicgstab(int dim, double** mat, double* rhs, double* sol, double tor, int max_steps);

int main() {
    int nofc;
    int nond;
    int number_element_dof = 1;
    int nond_on_face;
    int nint_para_fc;
    int** int_para_fc = NULL;
    int ndble_para_fc;
    double** dble_para_fc = NULL;
    int** face2node = NULL;
    int i, j;

    FILE *fp, *test, *testrhs;

    double** a;
    double *rhs, *sol, tor;
    int dim, max_steps;

    vector3_t* np;

    fp = fopen("input.txt", "r");

    // Read number of nodes from input data file : nond  
    fscanf(fp, "%d", &nond);

    // Allocation for the array for the coordinates of the nodes 
    np = (vector3_t*)malloc(sizeof(vector3_t) * nond);

    // Read the coordinates of the nodes from input data file : np  
    for (i = 0; i < nond; i++) {
        fscanf(fp, "%lf %lf %lf", &(np[i].x), &(np[i].y), &(np[i].z));
    }

    // printf("%lf %lf %lf\n",np[1].x,np[1].y,np[1].z); 

    // Read number of faces from input data file : nofc  
    fscanf(fp, "%d", &nofc);

    // Read number of nodes on each face from input data file : nond_on_face  
    fscanf(fp, "%d", &nond_on_face);

    // Read number of integer parameters set on each face from input data file : nint_para_fc  
    fscanf(fp, "%d", &nint_para_fc);

    // Read number of real(double precision) parameters set on each face from input data file : ndble_para_fc  
    fscanf(fp, "%d", &ndble_para_fc);

    printf("Number of nodes=%d Number of faces=%d\n", nond, nofc);

    // -----------------------------------------------

    face2node = (int**)malloc(sizeof(int*) * nofc);
    face2node[0] = (int*)malloc(sizeof(int) * nofc * nond_on_face);
    for (i = 1; i < nofc; i++) {
        face2node[i] = face2node[i - 1] + nond_on_face;
    }
    for (i = 0; i < nofc; i++) {
        for (j = 0; j < nond_on_face; j++) {
            fscanf(fp, "%d", &(face2node[i][j]));
        }
    }

    if (nint_para_fc > 0) {
        int_para_fc = (int**)malloc(sizeof(int*) * nofc);
        int_para_fc[0] = (int*)malloc(sizeof(int) * nofc * nint_para_fc);
        for (i = 1; i < nofc; i++) {
            int_para_fc[i] = int_para_fc[i - 1] + nint_para_fc;
        }

        for (i = 0; i < nofc; i++) {
            for (j = 0; j < nint_para_fc; j++) {
                fscanf(fp, "%d", &(int_para_fc[i][j]));
            }
        }
    }

    if (ndble_para_fc > 0) {
        dble_para_fc = (double**)malloc(sizeof(double*) * nofc);
        dble_para_fc[0] = (double*)malloc(sizeof(double) * nofc * ndble_para_fc);
        for (i = 1; i < nofc; i++) {
            dble_para_fc[i] = dble_para_fc[i - 1] + ndble_para_fc;
        }

        for (i = 0; i < nofc; i++) {
            for (j = 0; j < ndble_para_fc; j++) {
                fscanf(fp, "%lf", &(dble_para_fc[i][j]));
            }
        }
    }

    /* for(i=0;i<nofc;i++){
       for(j=0;j<nond_on_face;j++){
          printf("%d ",face2node[i][j]);
       }
       printf("\n");
    } */

    dim = nofc * number_element_dof;

    a = (double**)malloc(sizeof(double*) * dim);
    a[0] = (double*)malloc(sizeof(double) * dim * dim);
    for (i = 1; i < dim; i++) {
        a[i] = a[i - 1] + dim;
    }

    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            a[i][j] = 0.0;
        }
    }

    tor = 1E-8;
    max_steps = 1000;

    rhs = (double*)malloc(sizeof(double) * dim);
    sol = (double*)malloc(sizeof(double) * dim);

    for (i = 0; i < dim; i++) {
        //   rhs[i]=1.0; 
        sol[i] = 0.0;
        /*   a[i][i]=2.0;
        if (i-1>=0) {a[i][i-1]=-1.0;}
        if (i+1<dim) {a[i][i+1]=-1.0;} */
    }

    /*
   test = fopen("matrix.txt","r");
   for(i=0;i<dim;i++){
     for(j=0;j<dim;j++){
        fscanf(test, "%lf", &(a[i][j]));
      }
    }
   
   testrhs = fopen("rhs.txt","r");
   for(i=0;i<dim;i++){
        fscanf(testrhs, "%lf", &(rhs[i]));
    }
   
    fclose(test);
    fclose(testrhs);
    */

    // User Specified Function 
    // element_integral(coordinate np, double **a, ); 

    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            //     fscanf(test, "%lf", &(a[i][j]));
            a[i][j] = element_ij_(&i, &j, &nond, &nofc, &np[0], &face2node[0][0]);
        }
    }

    //testrhs = fopen("rhs.txt","r");
    for (i = 0; i < dim; i++) {
        //     fscanf(fp, "%lf", &(rhs[i])); 
        rhs[i] = dble_para_fc[i][0];
    }

    fclose(fp);

    printf("Linear system was generated.\n");

    pbicgstab(dim, a, rhs, sol, tor, max_steps);

    fp = fopen("out2.data", "w");
    for (i = 0; i < dim; i++)
        fprintf(fp, "%20.14e \n", sol[i]);

    // printf("%d,%d\n",nint_para_fc,ndble_para_fc); 
    free(np);
    free(face2node[0]);
    free(face2node);
    if (nint_para_fc > 0) {
        free(int_para_fc[0]);
        free(int_para_fc);
    }
    if (ndble_para_fc > 0) {
        free(dble_para_fc[0]);
        free(dble_para_fc);
    }

    printf("OK\n");
    free(a[0]);
    free(a);
    free(rhs);
    free(sol);
}

// Matrix vector multiplication with a dense matrix: q=Ap 
void matvec_direct(int dim, double** mat, double* p, double* q) {
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

// -----------------------------------------------
// Calculation of residual matrix with a dense matrix: r=b-Ax 
void residual_direct(int dim, double** mat, double* x, double* b, double* r) {
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

// -----------------------------------------------
// Calculation of dot product 
double dot_product(int dim, double* x, double* y) {
    double sum = 0;
    int i;

    for (i = 0; i < dim; i++) {
        sum = sum + x[i] * y[i];
    }

    return sum;
}

// -----------------------------------------------
void pbicgstab(int dim, double** mat, double* rhs, double* sol, double tor, int max_steps) {
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
