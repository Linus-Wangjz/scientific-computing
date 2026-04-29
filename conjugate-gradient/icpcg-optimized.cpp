#include <mkl.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "csr.hpp"

#ifdef USE_METIS
#include <metis.h>
#endif

struct SparseHandle {
    sparse_matrix_t handle = nullptr;

    SparseHandle() = default;
    SparseHandle(const SparseHandle&) = delete;
    SparseHandle& operator=(const SparseHandle&) = delete;

    ~SparseHandle() {
        if (handle) {
            mkl_sparse_destroy(handle);
        }
    }
};

struct MklBuffer {
    double *ptr = nullptr;

    explicit MklBuffer(int n) {
        ptr = static_cast<double *>(mkl_malloc(n * sizeof(double), 64));
        if (!ptr) {
            throw std::runtime_error("mkl_malloc failed");
        }
    }

    MklBuffer(const MklBuffer&) = delete;
    MklBuffer& operator=(const MklBuffer&) = delete;

    ~MklBuffer() {
        mkl_free(ptr);
    }

    double *data() { return ptr; }
    const double *data() const { return ptr; }
};

struct ColorSchedule {
    int num_colors = 0;
    std::vector<int> color_ptr;   // each color's starting offset in color_idx
    std::vector<int> color_idx;   // row indices ordered by color, then METIS block
    std::vector<int> row_color;   // row_color[i] is row i's dependency color
};

static void check_sparse_status(sparse_status_t status, const char *context) {
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(context) + " failed");
    }
}

static CsrMatrix transpose_csr(const CsrMatrix& L) {
    const int n = L.nrows;
    CsrMatrix U;
    U.nrows = n;
    U.ncols = n;
    U.ia.assign(n + 1, 0);
    U.ja.resize(L.ja.size());
    U.a.resize(L.a.size());

    for (int col : L.ja) {
        if (col < 0 || col >= n) {
            throw std::runtime_error("CSR column index out of range");
        }
        U.ia[col + 1]++;
    }
    for (int i = 0; i < n; ++i) {
        U.ia[i + 1] += U.ia[i];
    }

    std::vector<int> offset = U.ia;
    for (int i = 0; i < n; ++i) {
        for (int p = L.ia[i]; p < L.ia[i + 1]; ++p) {
            const int col = L.ja[p];
            const int dest = offset[col]++;
            U.ja[dest] = i;
            U.a[dest] = L.a[p];
        }
    }

    return U;
}

static CsrMatrix build_symmetric_structure(const CsrMatrix& A) {
    if (A.nrows != A.ncols) {
        throw std::runtime_error("graph structure requires a square matrix");
    }

    const int n = A.nrows;
    std::vector<std::vector<int>> adj(n);

    for (int i = 0; i < n; ++i) {
        for (int p = A.ia[i]; p < A.ia[i + 1]; ++p) {
            const int j = A.ja[p];
            if (j < 0 || j >= n) {
                throw std::runtime_error("CSR column index out of range");
            }
            if (i == j) {
                continue;
            }
            adj[i].push_back(j);
            adj[j].push_back(i);
        }
    }

    CsrMatrix G;
    G.nrows = n;
    G.ncols = n;
    G.ia.assign(n + 1, 0);

    for (int i = 0; i < n; ++i) {
        std::sort(adj[i].begin(), adj[i].end());
        adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
        G.ia[i + 1] = G.ia[i] + static_cast<int>(adj[i].size());
    }

    G.ja.resize(G.ia[n]);
    G.a.assign(G.ia[n], 1.0);
    for (int i = 0; i < n; ++i) {
        const int row_start = G.ia[i];
        for (int p = 0; p < static_cast<int>(adj[i].size()); ++p) {
            G.ja[row_start + p] = adj[i][p];
        }
    }

    return G;
}

static void optimize_color_with_metis(const CsrMatrix& graph, std::vector<int>& color_nodes) {
#ifdef USE_METIS
    const int num_nodes = static_cast<int>(color_nodes.size());
    const int nthreads = omp_get_max_threads();
    const int nparts_int = std::min(nthreads, num_nodes);

    if (num_nodes < nparts_int * 4 || nparts_int <= 1) {
        std::sort(color_nodes.begin(), color_nodes.end());
        return;
    }

    std::vector<int> local_id(graph.nrows, -1);
    for (int i = 0; i < num_nodes; ++i) {
        local_id[color_nodes[i]] = i;
    }

    std::vector<idx_t> xadj(num_nodes + 1, 0);
    std::vector<idx_t> adjncy;
    adjncy.reserve(num_nodes * 8);

    for (int i = 0; i < num_nodes; ++i) {
        const int global_u = color_nodes[i];
        for (int p = graph.ia[global_u]; p < graph.ia[global_u + 1]; ++p) {
            const int global_v = graph.ja[p];
            const int local_v = local_id[global_v];
            if (local_v != -1 && global_u != global_v) {
                adjncy.push_back(static_cast<idx_t>(local_v));
            }
        }
        xadj[i + 1] = static_cast<idx_t>(adjncy.size());
    }

    if (adjncy.empty()) {
        std::sort(color_nodes.begin(), color_nodes.end());
        return;
    }

    idx_t nvtxs = static_cast<idx_t>(num_nodes);
    idx_t ncon = 1;
    idx_t nparts = static_cast<idx_t>(nparts_int);
    idx_t objval = 0;
    std::vector<idx_t> part(num_nodes, 0);

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;

    const int ret = METIS_PartGraphKway(
        &nvtxs, &ncon, xadj.data(), adjncy.data(),
        nullptr, nullptr, nullptr, &nparts, nullptr,
        nullptr, options, &objval, part.data());

    if (ret != METIS_OK) {
        std::cerr << "[Warning] METIS partitioning failed for one color; using row-order sort.\n";
        std::sort(color_nodes.begin(), color_nodes.end());
        return;
    }

    std::vector<std::vector<int>> blocks(nparts_int);
    for (int i = 0; i < num_nodes; ++i) {
        const int block = static_cast<int>(part[i]);
        if (block >= 0 && block < nparts_int) {
            blocks[block].push_back(color_nodes[i]);
        }
    }

    int out = 0;
    for (auto& block : blocks) {
        std::sort(block.begin(), block.end());
        for (int row : block) {
            color_nodes[out++] = row;
        }
    }
    if (out != num_nodes) {
        std::sort(color_nodes.begin(), color_nodes.end());
    }
#else
    (void)graph;
    std::sort(color_nodes.begin(), color_nodes.end());
#endif
}

static ColorSchedule build_colored_schedule(
    const CsrMatrix& triangular,
    const CsrMatrix& graph,
    bool is_lower) {
    const int n = triangular.nrows;
    std::vector<int> row_color(n, 0);
    int max_color = 0;

    for (int pass = 0; pass < n; ++pass) {
        const int row = is_lower ? pass : (n - 1 - pass);
        int color = 0;

        for (int p = triangular.ia[row]; p < triangular.ia[row + 1]; ++p) {
            const int col = triangular.ja[p];
            if (is_lower && col < row) {
                color = std::max(color, row_color[col] + 1);
            } else if (!is_lower && col > row) {
                color = std::max(color, row_color[col] + 1);
            }
        }

        row_color[row] = color;
        max_color = std::max(max_color, color);
    }

    ColorSchedule schedule;
    schedule.num_colors = max_color + 1;
    schedule.row_color = std::move(row_color);
    schedule.color_ptr.assign(schedule.num_colors + 1, 0);

    std::vector<std::vector<int>> colors(schedule.num_colors);
    for (int i = 0; i < n; ++i) {
        colors[schedule.row_color[i]].push_back(i);
    }

    for (int color = 0; color < schedule.num_colors; ++color) {
        schedule.color_ptr[color] = static_cast<int>(schedule.color_idx.size());
        optimize_color_with_metis(graph, colors[color]);
        schedule.color_idx.insert(
            schedule.color_idx.end(),
            colors[color].begin(),
            colors[color].end());
    }
    schedule.color_ptr[schedule.num_colors] = static_cast<int>(schedule.color_idx.size());

    return schedule;
}

static void sptrsv_forward(
    const CsrMatrix& L,
    const ColorSchedule& schedule,
    const double *r,
    double *y) {
    for (int color = 0; color < schedule.num_colors; ++color) {
        const int start = schedule.color_ptr[color];
        const int end = schedule.color_ptr[color + 1];

        #pragma omp parallel for schedule(static)
        for (int p = start; p < end; ++p) {
            const int i = schedule.color_idx[p];
            double sum = r[i];
            double diag = 1.0;

            for (int q = L.ia[i]; q < L.ia[i + 1]; ++q) {
                const int col = L.ja[q];
                if (col < i) {
                    sum -= L.a[q] * y[col];
                } else if (col == i) {
                    diag = L.a[q];
                }
            }
            y[i] = sum / diag;
        }
    }
}

static void sptrsv_backward(
    const CsrMatrix& U,
    const ColorSchedule& schedule,
    const double *y,
    double *z) {
    for (int color = 0; color < schedule.num_colors; ++color) {
        const int start = schedule.color_ptr[color];
        const int end = schedule.color_ptr[color + 1];

        #pragma omp parallel for schedule(static)
        for (int p = start; p < end; ++p) {
            const int i = schedule.color_idx[p];
            double sum = y[i];
            double diag = 1.0;

            for (int q = U.ia[i]; q < U.ia[i + 1]; ++q) {
                const int col = U.ja[q];
                if (col > i) {
                    sum -= U.a[q] * z[col];
                } else if (col == i) {
                    diag = U.a[q];
                }
            }
            z[i] = sum / diag;
        }
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
        const std::string path = (argc > 1) ? argv[1] : "../data/bcsstk01/bcsstk01.mtx";
        CsrMatrix csr = read_mtx_to_csr(path);
        if (csr.nrows != csr.ncols) {
            std::cerr << "CG requires a square matrix, got "
                      << csr.nrows << "x" << csr.ncols << '\n';
            return 1;
        }
        const int n = csr.nrows;

        constexpr int max_iter = 1000000;
        constexpr double rel_tol = 1e-10;

        SparseHandle A;
        check_sparse_status(
            mkl_sparse_d_create_csr(&A.handle, SPARSE_INDEX_BASE_ZERO,
                                    n, n,
                                    csr.ia.data(), csr.ia.data() + 1,
                                    csr.ja.data(), csr.a.data()),
            "mkl_sparse_d_create_csr(A)");

        matrix_descr descr{};
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        check_sparse_status(mkl_sparse_optimize(A.handle), "mkl_sparse_optimize(A)");

        CsrMatrix L = incomplete_cholesky0(csr);
        CsrMatrix U = transpose_csr(L);
        CsrMatrix graph = build_symmetric_structure(csr);

        std::cout << "[Setup] Coloring triangular dependency graphs";
#ifdef USE_METIS
        std::cout << " with METIS partitioning";
#else
        std::cout << " with row-order fallback";
#endif
        std::cout << "...\n";

        ColorSchedule sched_L = build_colored_schedule(L, graph, true);
        ColorSchedule sched_U = build_colored_schedule(U, graph, false);

        std::cout << "[Setup] Dependency coloring complete. L colors: "
                  << sched_L.num_colors << ", U colors: " << sched_U.num_colors
                  << ", OpenMP threads: " << omp_get_max_threads() << '\n';

        MklBuffer x(n);
        MklBuffer y(n);
        MklBuffer z(n);
        MklBuffer b(n);
        MklBuffer r(n);
        MklBuffer p(n);
        MklBuffer Ap(n);

        for (int i = 0; i < n; ++i) {
            x.data()[i] = 0.0;
            b.data()[i] = 1.0;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        cblas_dcopy(n, b.data(), 1, r.data(), 1);
        check_sparse_status(
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0,
                            A.handle, descr, x.data(), 1.0, r.data()),
            "mkl_sparse_d_mv(A)");

        sptrsv_forward(L, sched_L, r.data(), y.data());
        sptrsv_backward(U, sched_U, y.data(), z.data());

        cblas_dcopy(n, z.data(), 1, p.data(), 1);

        double rho = cblas_ddot(n, r.data(), 1, z.data(), 1);
        if (!(rho > 0.0)) {
            throw std::runtime_error("PCG breakdown: r^T M^{-1} r is not positive");
        }

        const double b_norm = cblas_dnrm2(n, b.data(), 1);
        const double target_res_norm = rel_tol * ((b_norm == 0.0) ? 1.0 : b_norm);
        double recursive_res_norm = cblas_dnrm2(n, r.data(), 1);

        int iter = 0;
        while (recursive_res_norm > target_res_norm && iter < max_iter) {
            check_sparse_status(
                mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0,
                                A.handle, descr, p.data(), 0.0, Ap.data()),
                "mkl_sparse_d_mv(A)");

            const double pAp = cblas_ddot(n, p.data(), 1, Ap.data(), 1);
            if (!(pAp > 0.0)) {
                throw std::runtime_error("PCG breakdown: p^T A p is not positive");
            }
            const double alpha = rho / pAp;

            cblas_daxpy(n, alpha, p.data(), 1, x.data(), 1);
            cblas_daxpy(n, -alpha, Ap.data(), 1, r.data(), 1);

            ++iter;
            recursive_res_norm = cblas_dnrm2(n, r.data(), 1);
            if (recursive_res_norm <= target_res_norm) {
                break;
            }

            sptrsv_forward(L, sched_L, r.data(), y.data());
            sptrsv_backward(U, sched_U, y.data(), z.data());

            const double rho_new = cblas_ddot(n, r.data(), 1, z.data(), 1);
            if (!(rho_new > 0.0)) {
                throw std::runtime_error("PCG breakdown: r^T M^{-1} r is not positive");
            }

            const double beta = rho_new / rho;
            cblas_dscal(n, beta, p.data(), 1);
            cblas_daxpy(n, 1.0, z.data(), 1, p.data(), 1);

            rho = rho_new;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;

        check_sparse_status(
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0,
                            A.handle, descr, x.data(), 0.0, Ap.data()),
            "mkl_sparse_d_mv(A)");

        MklBuffer r_true(n);
        cblas_dcopy(n, b.data(), 1, r_true.data(), 1);
        cblas_daxpy(n, -1.0, Ap.data(), 1, r_true.data(), 1);

        const double true_res_norm = cblas_dnrm2(n, r_true.data(), 1);
        const double rel_err = (b_norm == 0.0) ? true_res_norm : (true_res_norm / b_norm);

        std::cout << "==================================================\n";
        std::cout << "ICPCG Optimized Solver Summary\n";
        std::cout << "==================================================\n";
        std::cout << "Matrix Size         : " << n << '\n';
        std::cout << "A Nonzeros          : " << csr.ia[n] << '\n';
        std::cout << "L Nonzeros          : " << L.ia[n] << '\n';
        std::cout << "L Colors            : " << sched_L.num_colors << '\n';
        std::cout << "U Colors            : " << sched_U.num_colors << '\n';
        std::cout << "OpenMP Threads      : " << omp_get_max_threads() << '\n';
        std::cout << "Iterations          : " << iter << '\n';
        std::cout << "Solve Time          : " << elapsed_seconds.count() << " seconds\n";
        std::cout << "Recursive Res Norm  : " << recursive_res_norm << '\n';
        std::cout << "True Res Norm       : " << true_res_norm << '\n';
        std::cout << "Relative Error      : " << rel_err << '\n';

        if (rel_err < 1e-6) {
            std::cout << "[PASS] Solution is numerically accurate.\n";
        } else {
            std::cout << "[WARN] Floating-point drift detected!\n";
        }
        std::cout << "==================================================\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
