#include <mkl.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "csr.hpp"

int main(int argc, char *argv[]) {
    // 1. 读取矩阵
    std::string path = (argc > 1) ? argv[1] : "../data/bcsstk01/bcsstk01.mtx";
    CsrMatrix csr = read_mtx_to_csr(path);
    if(csr.nrows != csr.ncols) {
        std::cerr << "CG requires a square matrix, got " << csr.nrows << "x" << csr.ncols << '\n';
        return 1;
    }
    int n = csr.nrows;

    // 2. MKL 句柄创建与优化 (Setup 阶段)
    sparse_matrix_t A = nullptr;
    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, 
                            n, n, 
                            csr.ia.data(), csr.ia.data() + 1, 
                            csr.ja.data(), csr.a.data());

    matrix_descr descr{};
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_optimize(A);

    // 3. 内存分配 (Setup 阶段，不计入求解时间)
    double *x  = (double *)mkl_malloc(n * sizeof(double), 64);
    double *b  = (double *)mkl_malloc(n * sizeof(double), 64);
    double *r  = (double *)mkl_malloc(n * sizeof(double), 64);
    double *p  = (double *)mkl_malloc(n * sizeof(double), 64);
    double *Ap = (double *)mkl_malloc(n * sizeof(double), 64); 

    for (int i = 0; i < n; ++i) {
        x[i] = 0.0; // 初始解 x0 = 0
        b[i] = 1.0; // 右端项 b = 1
    }

    // ==========================================================
    // [计时开始] 记录求解阶段的起始时间
    // ==========================================================
    auto start_time = std::chrono::high_resolution_clock::now();

    /* Initialization */
    cblas_dcopy(n, b, 1, r, 1); // r0 = b
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A, descr, x, 1.0, r); 
    // r0 = - Ax + r0, which is equivalent to r0 = - Ax + b, the initial residue
    cblas_dcopy(n, r, 1, p, 1); // set initial search direction p0 = r0

    double rho, rho_new;
    rho = cblas_ddot(n, r, 1, r, 1); // compute initial scalar rho = p0 ^ 2
    double beta; // Gram-Schmidt coefficient

    /* Iterative Process */
    int max_iter = 1000000;
    int iter = 0;
    
    // 设定容差为 1e-10 (即相对误差的平方)
    while (rho > 1e-20 && iter < max_iter) {
        /* Calculate Step Length */
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, p, 0.0, Ap);
        double alpha = rho / cblas_ddot(n, p, 1, Ap, 1);

        /* Update Intermediate Results */
        cblas_daxpy(n, alpha, p, 1, x, 1); 
        cblas_daxpy(n, -alpha, Ap, 1, r, 1); 

        /* Calculate Gram-Schmidt coefficient */
        rho_new = cblas_ddot(n, r, 1, r, 1);
        beta = rho_new / rho;

        /* Update Search Direction */
        cblas_dscal(n, beta, p, 1); 
        cblas_daxpy(n, 1.0, r, 1, p, 1); 

        rho = rho_new;
        iter++;
    }

    // ==========================================================
    // [计时结束] 记录求解阶段的结束时间并计算耗时
    // ==========================================================
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    // 输出求解统计信息
    std::cout << "==================================================\n";
    std::cout << "CG Solver Summary\n";
    std::cout << "==================================================\n";
    std::cout << "Matrix Size         : " << n << '\n';
    std::cout << "A Nonzeros          : " << csr.ia[n] << '\n';
    std::cout << "Iterations          : " << iter << '\n';
    std::cout << "Solve Time          : " << elapsed_seconds.count() << " seconds\n";

    // ==========================================================
    // 真实残差校验 (Verification 阶段，通常不计入纯求解性能评估)
    // ==========================================================
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, 0.0, Ap);

    double *r_true = (double *)mkl_malloc(n * sizeof(double), 64);
    cblas_dcopy(n, b, 1, r_true, 1);            
    cblas_daxpy(n, -1.0, Ap, 1, r_true, 1);     

    double true_res_norm = cblas_dnrm2(n, r_true, 1);
    double b_norm = cblas_dnrm2(n, b, 1);
    double rel_err = (b_norm == 0.0) ? true_res_norm : (true_res_norm / b_norm);

    std::cout << "Recursive Res Norm  : " << std::sqrt(rho) << '\n';
    std::cout << "True Res Norm       : " << true_res_norm << '\n';
    std::cout << "Relative Error      : " << rel_err << '\n';
    
    if (rel_err < 1e-6) {
        std::cout << "[PASS] Solution is numerically accurate.\n";
    } else {
        std::cout << "[WARN] Floating-point drift detected!\n";
    }
    std::cout << "==================================================\n";

    // 4. 清理内存
    mkl_free(r_true);
    mkl_sparse_destroy(A);
    mkl_free(r); mkl_free(p); mkl_free(Ap);
    mkl_free(x); mkl_free(b); 

    return 0;
}
