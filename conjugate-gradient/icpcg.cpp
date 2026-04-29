#include <mkl.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include "csr.hpp"

static void check_sparse_status(sparse_status_t status, const char *context) {
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(context) + " failed");
    }
}

static CsrMatrix incomplete_cholesky0(const CsrMatrix& A) {
    if (A.nrows != A.ncols) {
        throw std::runtime_error("IC(0) requires a square matrix");
    }

    const int n = A.nrows;
    std::vector<std::unordered_map<int, double>> lower_values(n);

    for (int i = 0; i < n; ++i) {
        for (int p = A.ia[i]; p < A.ia[i + 1]; ++p) {
            const int j = A.ja[p];
            if (j < 0 || j >= n) {
                throw std::runtime_error("CSR column index out of range");
            }
            if (j <= i) {
                lower_values[i][j] += A.a[p];
            }
        }
    }

    std::vector<std::vector<int>> cols(n);
    std::vector<std::vector<double>> vals(n);
    std::vector<std::unordered_map<int, int>> pos(n);
    std::vector<int> diag_pos(n, -1);
    double diag_scale = 0.0;

    for (int i = 0; i < n; ++i) {
        cols[i].reserve(lower_values[i].size());
        for (const auto& entry : lower_values[i]) {
            cols[i].push_back(entry.first);
        }
        std::sort(cols[i].begin(), cols[i].end());

        vals[i].assign(cols[i].size(), 0.0);
        pos[i].reserve(cols[i].size());
        for (int p = 0; p < static_cast<int>(cols[i].size()); ++p) {
            pos[i][cols[i][p]] = p;
        }

        const auto diag_it = pos[i].find(i);
        if (diag_it == pos[i].end()) {
            throw std::runtime_error("IC(0) requires every diagonal entry to be present");
        }
        diag_pos[i] = diag_it->second;
        diag_scale = std::max(diag_scale, std::abs(lower_values[i][i]));
    }
    if (diag_scale == 0.0) {
        diag_scale = 1.0;
    }

    const double pivot_floor = std::max(1e-300, 1e-12 * diag_scale);
    constexpr double row_pivot_floor_ratio = 1e-3;
    auto clear_vals = [&]() {
        for (auto& row_vals : vals) {
            std::fill(row_vals.begin(), row_vals.end(), 0.0);
        }
    };

    bool factorized = false;
    double used_shift = 0.0;
    constexpr int max_shift_attempts = 16;

    for (int attempt = 0; attempt < max_shift_attempts && !factorized; ++attempt) {
        const double shift =
            (attempt == 0) ? 0.0 : diag_scale * std::pow(10.0, attempt - 13);
        clear_vals();

        bool breakdown = false;
        for (int i = 0; i < n && !breakdown; ++i) {
            const int idiag = diag_pos[i];

            for (int p = 0; p < idiag; ++p) {
                const int j = cols[i][p];
                double lij = lower_values[i][j];

                for (int q = 0; q < p; ++q) {
                    const int k = cols[i][q];
                    const auto jk_it = pos[j].find(k);
                    if (jk_it != pos[j].end()) {
                        lij -= vals[i][q] * vals[j][jk_it->second];
                    }
                }

                const double ljj = vals[j][diag_pos[j]];
                if (!(std::abs(ljj) > pivot_floor) || !std::isfinite(ljj)) {
                    breakdown = true;
                    break;
                }
                vals[i][p] = lij / ljj;
                if (!std::isfinite(vals[i][p])) {
                    breakdown = true;
                    break;
                }
            }

            double diag = lower_values[i][i] + shift;
            for (int p = 0; p < idiag; ++p) {
                diag -= vals[i][p] * vals[i][p];
            }
            if (!(diag > pivot_floor) || !std::isfinite(diag)) {
                breakdown = true;
                break;
            }
            vals[i][idiag] = std::sqrt(diag);
        }

        if (!breakdown) {
            factorized = true;
            used_shift = shift;
        }
    }

    int boosted_pivots = 0;
    if (!factorized) {
        clear_vals();
        for (int i = 0; i < n; ++i) {
            const int idiag = diag_pos[i];

            for (int p = 0; p < idiag; ++p) {
                const int j = cols[i][p];
                double lij = lower_values[i][j];

                for (int q = 0; q < p; ++q) {
                    const int k = cols[i][q];
                    const auto jk_it = pos[j].find(k);
                    if (jk_it != pos[j].end()) {
                        lij -= vals[i][q] * vals[j][jk_it->second];
                    }
                }

                const double ljj = vals[j][diag_pos[j]];
                vals[i][p] = (std::abs(ljj) > pivot_floor && std::isfinite(ljj))
                    ? lij / ljj
                    : 0.0;
                if (!std::isfinite(vals[i][p])) {
                    vals[i][p] = 0.0;
                    ++boosted_pivots;
                }
            }

            double diag = lower_values[i][i];
            for (int p = 0; p < idiag; ++p) {
                diag -= vals[i][p] * vals[i][p];
            }
            const double row_pivot_floor =
                std::max(pivot_floor, row_pivot_floor_ratio * std::abs(lower_values[i][i]));
            if (!(diag > row_pivot_floor) || !std::isfinite(diag)) {
                for (int p = 0; p < idiag; ++p) {
                    vals[i][p] = 0.0;
                }
                diag = std::max(row_pivot_floor, std::abs(lower_values[i][i]));
                ++boosted_pivots;
            }
            vals[i][idiag] = std::sqrt(diag);
        }
    }

    if (used_shift > 0.0) {
        std::cerr << "[IC] Used diagonal shift " << used_shift << ".\n";
    }
    if (boosted_pivots > 0) {
        std::cerr << "[IC] Safeguarded " << boosted_pivots
                  << " small/non-positive diagonal pivot(s).\n";
    }

    CsrMatrix L;
    L.nrows = n;
    L.ncols = n;
    L.ia.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        L.ia[i + 1] = L.ia[i] + static_cast<int>(cols[i].size());
    }

    L.ja.resize(L.ia[n]);
    L.a.resize(L.ia[n]);
    for (int i = 0; i < n; ++i) {
        const int row_start = L.ia[i];
        for (int p = 0; p < static_cast<int>(cols[i].size()); ++p) {
            L.ja[row_start + p] = cols[i][p];
            L.a[row_start + p] = vals[i][p];
        }
    }

    return L;
}

int main(int argc, char *argv[]) {
    try {
        // 1. 读取矩阵
        std::string path = (argc > 1) ? argv[1] : "../data/bcsstk01/bcsstk01.mtx";
        CsrMatrix csr = read_mtx_to_csr(path);
        if(csr.nrows != csr.ncols) {
            std::cerr << "CG requires a square matrix, got " << csr.nrows << "x" << csr.ncols << '\n';
            return 1;
        }
        int n = csr.nrows;

        constexpr int max_iter = 1000000;
        constexpr double rel_tol = 1e-10;

        // 2. Setup 阶段：构造 A 和 shifted/safeguarded IC(0) 的 L
        sparse_matrix_t A = nullptr;
        check_sparse_status(
            mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO,
                                    n, n,
                                    csr.ia.data(), csr.ia.data() + 1,
                                    csr.ja.data(), csr.a.data()),
            "mkl_sparse_d_create_csr(A)");

        matrix_descr descr{};
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        check_sparse_status(mkl_sparse_optimize(A), "mkl_sparse_optimize(A)");

        CsrMatrix ic = incomplete_cholesky0(csr);
        sparse_matrix_t L = nullptr;
        check_sparse_status(
            mkl_sparse_d_create_csr(&L, SPARSE_INDEX_BASE_ZERO,
                                    n, n,
                                    ic.ia.data(), ic.ia.data() + 1,
                                    ic.ja.data(), ic.a.data()),
            "mkl_sparse_d_create_csr(L)");

        matrix_descr L_descr{};
        L_descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
        L_descr.mode = SPARSE_FILL_MODE_LOWER;
        L_descr.diag = SPARSE_DIAG_NON_UNIT;

        check_sparse_status(
            mkl_sparse_set_sv_hint(L, SPARSE_OPERATION_NON_TRANSPOSE, L_descr, max_iter),
            "mkl_sparse_set_sv_hint(L)");
        check_sparse_status(
            mkl_sparse_set_sv_hint(L, SPARSE_OPERATION_TRANSPOSE, L_descr, max_iter),
            "mkl_sparse_set_sv_hint(L^T)");
        check_sparse_status(mkl_sparse_optimize(L), "mkl_sparse_optimize(L)");

        // 3. 内存分配 (Setup 阶段，不计入求解时间)
        double *x  = (double *)mkl_malloc(n * sizeof(double), 64);
        double *y  = (double *)mkl_malloc(n * sizeof(double), 64);
        double *z  = (double *)mkl_malloc(n * sizeof(double), 64);
        double *b  = (double *)mkl_malloc(n * sizeof(double), 64);
        double *r  = (double *)mkl_malloc(n * sizeof(double), 64);
        double *p  = (double *)mkl_malloc(n * sizeof(double), 64);
        double *Ap = (double *)mkl_malloc(n * sizeof(double), 64);

        if (!x || !y || !z || !b || !r || !p || !Ap) {
            throw std::runtime_error("mkl_malloc failed");
        }

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
        check_sparse_status(
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A, descr, x, 1.0, r),
            "mkl_sparse_d_mv(A)");
        // r0 = b - A x0

        check_sparse_status(
            mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, L, L_descr, r, y),
            "mkl_sparse_d_trsv(L)");
        // sparse triangular solve L y = r
        check_sparse_status(
            mkl_sparse_d_trsv(SPARSE_OPERATION_TRANSPOSE, 1.0, L, L_descr, y, z),
            "mkl_sparse_d_trsv(L^T)");
        // sparse triangular solve L^T z = y

        cblas_dcopy(n, z, 1, p, 1); // p0 = z0

        double rho = cblas_ddot(n, r, 1, z, 1);
        if (!(rho > 0.0)) {
            throw std::runtime_error("PCG breakdown: r^T M^{-1} r is not positive");
        }

        const double b_norm = cblas_dnrm2(n, b, 1);
        const double target_res_norm = rel_tol * ((b_norm == 0.0) ? 1.0 : b_norm);
        double recursive_res_norm = cblas_dnrm2(n, r, 1);

        /* Iterative Process */
        int iter = 0;

        while (recursive_res_norm > target_res_norm && iter < max_iter) {
            /* Calculate Step Length */
            check_sparse_status(
                mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, p, 0.0, Ap),
                "mkl_sparse_d_mv(A)");
            const double pAp = cblas_ddot(n, p, 1, Ap, 1);
            if (!(pAp > 0.0)) {
                throw std::runtime_error("PCG breakdown: p^T A p is not positive");
            }
            const double alpha = rho / pAp;

            /* Update Intermediate Results */
            cblas_daxpy(n, alpha, p, 1, x, 1);      // x_{k+1} = x_k + alpha_k p_k
            cblas_daxpy(n, -alpha, Ap, 1, r, 1);    // r_{k+1} = r_k - alpha_k A p_k

            ++iter;
            recursive_res_norm = cblas_dnrm2(n, r, 1);
            if (recursive_res_norm <= target_res_norm) {
                break;
            }

            /* Apply Preconditioner */
            check_sparse_status(
                mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, L, L_descr, r, y),
                "mkl_sparse_d_trsv(L)");
            // sparse triangular solve L y = r
            check_sparse_status(
                mkl_sparse_d_trsv(SPARSE_OPERATION_TRANSPOSE, 1.0, L, L_descr, y, z),
                "mkl_sparse_d_trsv(L^T)");
            // sparse triangular solve L^T z = y

            const double rho_new = cblas_ddot(n, r, 1, z, 1);
            if (!(rho_new > 0.0)) {
                throw std::runtime_error("PCG breakdown: r^T M^{-1} r is not positive");
            }

            /* Update Search Direction */
            const double beta = rho_new / rho;
            cblas_dscal(n, beta, p, 1);
            cblas_daxpy(n, 1.0, z, 1, p, 1);

            rho = rho_new;
        }

        // ==========================================================
        // [计时结束] 记录求解阶段的结束时间并计算耗时
        // ==========================================================
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;

        // 输出求解统计信息
        std::cout << "==================================================\n";
        std::cout << "ICPCG Solver Summary\n";
        std::cout << "==================================================\n";
        std::cout << "Matrix Size         : " << n << '\n';
        std::cout << "A Nonzeros          : " << csr.ia[n] << '\n';
        std::cout << "L Nonzeros          : " << ic.ia[n] << '\n';
        std::cout << "Iterations          : " << iter << '\n';
        std::cout << "Solve Time          : " << elapsed_seconds.count() << " seconds\n";

        // ==========================================================
        // 真实残差校验 (Verification 阶段，通常不计入纯求解性能评估)
        // ==========================================================
        check_sparse_status(
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, 0.0, Ap),
            "mkl_sparse_d_mv(A)");

        double *r_true = (double *)mkl_malloc(n * sizeof(double), 64);
        if (!r_true) {
            throw std::runtime_error("mkl_malloc failed");
        }
        cblas_dcopy(n, b, 1, r_true, 1);
        cblas_daxpy(n, -1.0, Ap, 1, r_true, 1);

        double true_res_norm = cblas_dnrm2(n, r_true, 1);
        double rel_err = (b_norm == 0.0) ? true_res_norm : (true_res_norm / b_norm);

        std::cout << "Recursive Res Norm  : " << recursive_res_norm << '\n';
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
        mkl_sparse_destroy(L);
        mkl_sparse_destroy(A);
        mkl_free(r); mkl_free(p); mkl_free(Ap);
        mkl_free(y); mkl_free(z);
        mkl_free(x); mkl_free(b);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
