/* cython_fastshermor.c
 *
 * Rutger van Haasteren, April 19 2023, Hannover
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
                          double *alpha, double *a, int *lda, double *b,
                          int *ldb, double *beta, double *c, int *ldc);
extern void dgemv_(char *trans, int *m, int *n, double *alpha, double *a,
                          int *lda, double *x, int *incx, double *beta,
                          double *y, int *incy);
extern void dger_(int *m, int *n, double *alpha, double *x, int *incx,
                          double *y, int *incy, double *a, int *lda);

static void blas_block_shermor_2D_asym(
                int n_Z1_rows,
                int n_Z1_cols,
                int n_Z1_row_major,
                double *pd_Z1,
                int n_Z2_cols,
                int n_Z2_row_major,
                double *pd_Z2,
                double *pd_Nvec,
                int n_J_rows,
                double *pd_Jvec,
                int *pn_Uinds,
                double *pd_ZNZ,
                double *pd_Jldet
        ) {
    /* C implementation of python_block_shermor_2D_asym, because the python
     * overhead is large
     *
     * parameters
     * ----------
     *
     * :param n_Z1_rows:        Number of rows of Z1
     * :param n_Z1_cols:        Number of columns of Z1
     * :param n_Z1_row_major:   1 if Z1 is Row-Major, 0 if Column-Major
     * :param pd_Z1:            The Z1 matrix
     * :param n_Z2_cols:        Number of columns of Z2
     * :param n_Z2_row_major:   1 if Z2 is Row-Major, 0 if Column-Major
     * :param pd_Z2:            The Z2 matrix
     * :param pd_Nvec:          The Nvec vector
     * :param n_J_rows:         The number of Jvec elements
     * :param pd_Jvec:          The Jvec vector
     * :param pn_Uinds:         The matrix of quantization indices (Row-Major)
     * :param pd_ZNZ:           The return value of ZNZ (Column-Major)
     * :param pd_Jldet:         The return value of log(det(J))
     */

    double d_galpha=1.0, d_gbeta=0.0, d_nisum=0.0, d_beta;
    double *pd_Z1ni, *pd_ZNZ_add, *pd_ni, *pd_zn1, *pd_zn2;
    int cc, i, j, m, n, k, lda, ldb, ldc, n_jblock, n_jblock_i, n_index;
    char *transa, *transb;

    pd_Z1ni = malloc(n_Z1_rows*n_Z1_cols * sizeof(double));
    pd_ZNZ_add = calloc(n_Z1_rows*n_Z1_cols, sizeof(double));
    pd_ni = malloc(n_Z1_rows * sizeof(double));
    pd_zn1 = calloc(n_Z1_cols, sizeof(double));
    pd_zn2 = calloc(n_Z2_cols, sizeof(double));

    /* openmp this? */
    for(i=0; i<n_Z1_rows; ++i) {
        pd_ni[i] = 1.0 / pd_Nvec[i];
        for(j=0; j<n_Z1_cols; ++j) {
            if(n_Z1_row_major)
                n_index = j+i*n_Z1_cols;
            else
                n_index = i+j*n_Z1_rows;
            pd_Z1ni[n_index] = pd_Z1[n_index] * pd_ni[i];
        } /* for j */
    } /* for i */

    m = n_Z1_cols;
    n = n_Z2_cols;
    k = n_Z1_rows;
    ldc = n_Z1_cols;
    if(n_Z1_row_major) {
        transa = "N";
        lda = n_Z1_cols;
    } else {
        transa = "T";
        lda = n_Z1_rows;
    }
    if(n_Z2_row_major) {
        transb = "T";
        ldb = n_Z2_cols; 
    } else {
        transb = "N";
        ldb = n_Z1_rows;
    }

    dgemm_(transa, transb, &m, &n, &k, &d_galpha, pd_Z1ni, &lda, pd_Z2, &ldb,
           &d_gbeta, pd_ZNZ, &ldc);

    *pd_Jldet = 0.0;
    for(cc=0; cc<n_J_rows; ++cc) {
        if(pd_Jvec[cc] > 0.0) {

            /* Note: pn_Uinds is row-major */
            d_nisum = 0.0;
            n_jblock_i = pn_Uinds[2*cc];
            n_jblock = pn_Uinds[2*cc+1] - pn_Uinds[2*cc];
            for(i=pn_Uinds[2*cc]; i<pn_Uinds[2*cc+1]; ++i) {
                d_nisum += pd_ni[i];
            } /* for i */

            d_beta = -1.0 / (d_nisum + 1.0/pd_Jvec[cc]);

            /* Calculate zn1 = np.dot(niblock, Zblock1) */
            /* Use dgemm */
            /*
            m = 1;
            n = n_Z1_cols;
            k = n_jblock;
            lda = 1;
            ldc = 1;
            transa = "N";
            if(n_Z1_row_major) {
                transb = "T";
                ldb = n_Z1_cols;
                n_index = n_jblock_i*n_Z1_cols;
            } else {
                transb = "N";
                ldb = n_Z1_rows;
                n_index = n_jblock_i;
            }

            dgemm_(transa, transb, &m, &n, &k, &d_galpha, &pd_ni[n_jblock_i],
                    &lda, &pd_Z1[n_index], &ldb, &d_gbeta, pd_zn1, &ldc);
            */

            /* Calculate zn1 = np.dot(niblock, Zblock1) */
            /* Use dgemv */
            lda = 1;
            ldc = 1;
            if(n_Z1_row_major) {
                transb = "N";
                m = n_Z1_cols;
                n = n_jblock;
                ldb = n_Z1_cols;
                n_index = n_jblock_i*n_Z1_cols;
            } else {
                transb = "T";
                m = n_jblock;
                n = n_Z1_cols;
                ldb = n_Z1_rows;
                n_index = n_jblock_i;
            }
            dgemv_(transb, &m, &n, &d_galpha, &pd_Z1[n_index], &ldb,
                    &pd_ni[n_jblock_i], &lda, &d_gbeta, pd_zn1, &ldc);


            /* Calculate zn2 = np.dot(niblock, Zblock2) */
            /* Use dgemm */
            /*
            m = 1;
            n = n_Z2_cols;
            k = n_jblock;
            lda = 1;
            ldc = 1;
            transa = "N";
            if(n_Z2_row_major) {
                transb = "T";
                ldb = n_Z2_cols;
                n_index = n_jblock_i*n_Z2_cols;
            } else {
                transb = "N";
                ldb = n_Z1_rows;
                n_index = n_jblock_i;
            }

            dgemm_(transa, transb, &m, &n, &k, &d_galpha, &pd_ni[n_jblock_i],
                    &lda, &pd_Z2[n_index], &ldb, &d_gbeta, pd_zn2, &ldc);
            */

            /* Calculate zn2 = np.dot(niblock, Zblock2) */
            /* Use dgemv */
            lda = 1;
            ldc = 1;
            if(n_Z2_row_major) {
                transb = "N";
                m = n_Z2_cols;
                n = n_jblock;
                ldb = n_Z2_cols;
                n_index = n_jblock_i*n_Z2_cols;
            } else {
                transb = "T";
                m = n_jblock;
                n = n_Z2_cols;
                ldb = n_Z1_rows;
                n_index = n_jblock_i;
            }
            dgemv_(transb, &m, &n, &d_galpha, &pd_Z2[n_index], &ldb,
                    &pd_ni[n_jblock_i], &lda, &d_gbeta, pd_zn2, &ldc);

            /* Calculate zNz -= beta * np.outer(zn1.T, zn2) */
            /* beta is negative, because this is an update to pd_ZNZ */
            m = n_Z1_cols;
            n = n_Z2_cols;
            k = 1;
            lda = 1;
            ldb = 1;
            ldc = m;
            dger_(&m, &n, &d_beta, pd_zn1, &lda, pd_zn2, &ldb, pd_ZNZ, &ldc);

            *pd_Jldet += log(pd_Jvec[cc]) - log(-d_beta);

        } /* if pd_Jvec[cc] */
    } /* for cc */



    free(pd_ni);
    free(pd_ZNZ_add);
    free(pd_Z1ni);
    free(pd_zn1);
    free(pd_zn2);

    return;

}





