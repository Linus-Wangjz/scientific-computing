#include <mkl.h>
#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
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
        ptr = static_cast<double *>(mkl_malloc(static_cast<size_t>(n) * sizeof(double), 64));
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
    std::vector<int> color_idx;   // row indices ordered by color, then optional METIS block
    std::vector<int> row_color;   // row_color[i] is row i's dependency color
};

static void check_sparse_status(sparse_status_t status, const char *context) {
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(context) + " failed");
    }
}

static void check_cl(cl_int err, const char *context) {
    if (err != CL_SUCCESS) {
        std::ostringstream oss;
        oss << context << " failed with OpenCL error " << err;
        throw std::runtime_error(oss.str());
    }
}

static std::string get_device_string(cl_device_id device, cl_device_info key) {
    size_t n = 0;
    check_cl(clGetDeviceInfo(device, key, 0, nullptr, &n), "clGetDeviceInfo(size)");
    std::string s(n, '\0');
    check_cl(clGetDeviceInfo(device, key, n, &s[0], nullptr), "clGetDeviceInfo(value)");
    while (!s.empty() && s.back() == '\0') {
        s.pop_back();
    }
    return s;
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

static void optimize_color_with_metis(
    const CsrMatrix& graph,
    std::vector<int>& color_nodes,
    int partition_count) {
#ifdef USE_METIS
    const int num_nodes = static_cast<int>(color_nodes.size());
    const int nparts_int = std::min(std::max(partition_count, 1), num_nodes);

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
    adjncy.reserve(static_cast<size_t>(num_nodes) * 8);

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
    (void)partition_count;
    std::sort(color_nodes.begin(), color_nodes.end());
#endif
}

static ColorSchedule build_colored_schedule(
    const CsrMatrix& triangular,
    const CsrMatrix& graph,
    bool is_lower,
    int partition_count) {
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
        optimize_color_with_metis(graph, colors[color], partition_count);
        schedule.color_idx.insert(
            schedule.color_idx.end(),
            colors[color].begin(),
            colors[color].end());
    }
    schedule.color_ptr[schedule.num_colors] = static_cast<int>(schedule.color_idx.size());

    return schedule;
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

static const char *kSptrsvKernels = R"CLC(
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void sptrsv_forward_color(
    __global const int *ia,
    __global const int *ja,
    __global const double *a,
    __global const int *color_idx,
    const int color_start,
    const int color_count,
    __global const double *r,
    __global double *y) {

    const int lid = (int)get_global_id(0);
    if (lid >= color_count) {
        return;
    }

    const int i = color_idx[color_start + lid];
    double sum = r[i];
    double diag = 1.0;

    for (int q = ia[i]; q < ia[i + 1]; ++q) {
        const int col = ja[q];
        if (col < i) {
            sum -= a[q] * y[col];
        } else if (col == i) {
            diag = a[q];
        }
    }

    y[i] = sum / diag;
}

__kernel void sptrsv_backward_color(
    __global const int *ia,
    __global const int *ja,
    __global const double *a,
    __global const int *color_idx,
    const int color_start,
    const int color_count,
    __global const double *y,
    __global double *z) {

    const int lid = (int)get_global_id(0);
    if (lid >= color_count) {
        return;
    }

    const int i = color_idx[color_start + lid];
    double sum = y[i];
    double diag = 1.0;

    for (int q = ia[i]; q < ia[i + 1]; ++q) {
        const int col = ja[q];
        if (col > i) {
            sum -= a[q] * z[col];
        } else if (col == i) {
            diag = a[q];
        }
    }

    z[i] = sum / diag;
}
)CLC";

struct OpenClSptrsv {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel forward_kernel = nullptr;
    cl_kernel backward_kernel = nullptr;

    cl_mem d_L_ia = nullptr;
    cl_mem d_L_ja = nullptr;
    cl_mem d_L_a = nullptr;
    cl_mem d_L_color_idx = nullptr;

    cl_mem d_U_ia = nullptr;
    cl_mem d_U_ja = nullptr;
    cl_mem d_U_a = nullptr;
    cl_mem d_U_color_idx = nullptr;

    cl_mem d_r = nullptr;
    cl_mem d_y = nullptr;
    cl_mem d_z = nullptr;

    int n = 0;
    cl_uint compute_units = 1;
    std::string device_name;

    OpenClSptrsv(
        const CsrMatrix& L,
        const ColorSchedule& sched_L,
        const CsrMatrix& U,
        const ColorSchedule& sched_U) {
        n = L.nrows;
        init_context();
        build_program();
        create_buffers(L, sched_L, U, sched_U);
    }

    OpenClSptrsv(const OpenClSptrsv&) = delete;
    OpenClSptrsv& operator=(const OpenClSptrsv&) = delete;

    ~OpenClSptrsv() {
        release_mem(d_z);
        release_mem(d_y);
        release_mem(d_r);
        release_mem(d_U_color_idx);
        release_mem(d_U_a);
        release_mem(d_U_ja);
        release_mem(d_U_ia);
        release_mem(d_L_color_idx);
        release_mem(d_L_a);
        release_mem(d_L_ja);
        release_mem(d_L_ia);
        if (backward_kernel) clReleaseKernel(backward_kernel);
        if (forward_kernel) clReleaseKernel(forward_kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    static void release_mem(cl_mem& m) {
        if (m) {
            clReleaseMemObject(m);
            m = nullptr;
        }
    }

    void init_context() {
        cl_uint num_platforms = 0;
        check_cl(clGetPlatformIDs(0, nullptr, &num_platforms), "clGetPlatformIDs(count)");
        if (num_platforms == 0) {
            throw std::runtime_error("No OpenCL platform found");
        }
        std::vector<cl_platform_id> platforms(num_platforms);
        check_cl(clGetPlatformIDs(num_platforms, platforms.data(), nullptr), "clGetPlatformIDs(list)");

        const cl_device_type preferred_types[] = {
            CL_DEVICE_TYPE_GPU,
#ifdef CL_DEVICE_TYPE_ACCELERATOR
            CL_DEVICE_TYPE_ACCELERATOR,
#endif
            CL_DEVICE_TYPE_DEFAULT,
            CL_DEVICE_TYPE_CPU
        };

        for (cl_device_type type : preferred_types) {
            for (cl_platform_id p : platforms) {
                cl_uint num_devices = 0;
                cl_int err = clGetDeviceIDs(p, type, 0, nullptr, &num_devices);
                if (err != CL_SUCCESS || num_devices == 0) {
                    continue;
                }
                std::vector<cl_device_id> devices(num_devices);
                err = clGetDeviceIDs(p, type, num_devices, devices.data(), nullptr);
                if (err != CL_SUCCESS) {
                    continue;
                }
                for (cl_device_id d : devices) {
                    cl_device_fp_config fp64 = 0;
                    err = clGetDeviceInfo(d, CL_DEVICE_DOUBLE_FP_CONFIG,
                                          sizeof(fp64), &fp64, nullptr);
                    if (err == CL_SUCCESS && fp64 != 0) {
                        platform = p;
                        device = d;
                        break;
                    }
                }
                if (device) break;
            }
            if (device) break;
        }

        if (!device) {
            throw std::runtime_error("No OpenCL device with double precision support found");
        }

        cl_int err = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        check_cl(err, "clCreateContext");

        queue = clCreateCommandQueue(context, device, 0, &err);
        check_cl(err, "clCreateCommandQueue");

        check_cl(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                                 sizeof(compute_units), &compute_units, nullptr),
                 "clGetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS)");
        device_name = get_device_string(device, CL_DEVICE_NAME);
    }

    void build_program() {
        cl_int err = CL_SUCCESS;
        const char *src = kSptrsvKernels;
        const size_t len = std::strlen(kSptrsvKernels);
        program = clCreateProgramWithSource(context, 1, &src, &len, &err);
        check_cl(err, "clCreateProgramWithSource");

        err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            if (log_size > 0) {
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                      log_size, &log[0], nullptr);
            }
            throw std::runtime_error("clBuildProgram failed:\n" + log);
        }

        forward_kernel = clCreateKernel(program, "sptrsv_forward_color", &err);
        check_cl(err, "clCreateKernel(sptrsv_forward_color)");
        backward_kernel = clCreateKernel(program, "sptrsv_backward_color", &err);
        check_cl(err, "clCreateKernel(sptrsv_backward_color)");
    }

    template <typename T>
    cl_mem make_readonly_buffer(const std::vector<T>& v, const char *name) {
        if (v.empty()) {
            throw std::runtime_error(std::string(name) + " is empty; empty buffers are not supported here");
        }
        cl_int err = CL_SUCCESS;
        cl_mem m = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(T) * v.size(),
                                  const_cast<T*>(v.data()),
                                  &err);
        check_cl(err, name);
        return m;
    }

    cl_mem make_readwrite_double_buffer(int len, const char *name) {
        cl_int err = CL_SUCCESS;
        cl_mem m = clCreateBuffer(context,
                                  CL_MEM_READ_WRITE,
                                  sizeof(double) * static_cast<size_t>(len),
                                  nullptr,
                                  &err);
        check_cl(err, name);
        return m;
    }

    void create_buffers(
        const CsrMatrix& L,
        const ColorSchedule& sched_L,
        const CsrMatrix& U,
        const ColorSchedule& sched_U) {
        d_L_ia = make_readonly_buffer(L.ia, "clCreateBuffer(L.ia)");
        d_L_ja = make_readonly_buffer(L.ja, "clCreateBuffer(L.ja)");
        d_L_a = make_readonly_buffer(L.a, "clCreateBuffer(L.a)");
        d_L_color_idx = make_readonly_buffer(sched_L.color_idx, "clCreateBuffer(L.color_idx)");

        d_U_ia = make_readonly_buffer(U.ia, "clCreateBuffer(U.ia)");
        d_U_ja = make_readonly_buffer(U.ja, "clCreateBuffer(U.ja)");
        d_U_a = make_readonly_buffer(U.a, "clCreateBuffer(U.a)");
        d_U_color_idx = make_readonly_buffer(sched_U.color_idx, "clCreateBuffer(U.color_idx)");

        d_r = make_readwrite_double_buffer(n, "clCreateBuffer(r)");
        d_y = make_readwrite_double_buffer(n, "clCreateBuffer(y)");
        d_z = make_readwrite_double_buffer(n, "clCreateBuffer(z)");
    }

    void enqueue_forward_color(int color_start, int color_count) {
        if (color_count <= 0) return;
        cl_int err = CL_SUCCESS;
        int arg = 0;
        err  = clSetKernelArg(forward_kernel, arg++, sizeof(cl_mem), &d_L_ia);
        err |= clSetKernelArg(forward_kernel, arg++, sizeof(cl_mem), &d_L_ja);
        err |= clSetKernelArg(forward_kernel, arg++, sizeof(cl_mem), &d_L_a);
        err |= clSetKernelArg(forward_kernel, arg++, sizeof(cl_mem), &d_L_color_idx);
        err |= clSetKernelArg(forward_kernel, arg++, sizeof(int), &color_start);
        err |= clSetKernelArg(forward_kernel, arg++, sizeof(int), &color_count);
        err |= clSetKernelArg(forward_kernel, arg++, sizeof(cl_mem), &d_r);
        err |= clSetKernelArg(forward_kernel, arg++, sizeof(cl_mem), &d_y);
        check_cl(err, "clSetKernelArg(sptrsv_forward_color)");

        const size_t global = static_cast<size_t>(color_count);
        check_cl(clEnqueueNDRangeKernel(queue, forward_kernel, 1, nullptr,
                                        &global, nullptr, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel(sptrsv_forward_color)");
    }

    void enqueue_backward_color(int color_start, int color_count) {
        if (color_count <= 0) return;
        cl_int err = CL_SUCCESS;
        int arg = 0;
        err  = clSetKernelArg(backward_kernel, arg++, sizeof(cl_mem), &d_U_ia);
        err |= clSetKernelArg(backward_kernel, arg++, sizeof(cl_mem), &d_U_ja);
        err |= clSetKernelArg(backward_kernel, arg++, sizeof(cl_mem), &d_U_a);
        err |= clSetKernelArg(backward_kernel, arg++, sizeof(cl_mem), &d_U_color_idx);
        err |= clSetKernelArg(backward_kernel, arg++, sizeof(int), &color_start);
        err |= clSetKernelArg(backward_kernel, arg++, sizeof(int), &color_count);
        err |= clSetKernelArg(backward_kernel, arg++, sizeof(cl_mem), &d_y);
        err |= clSetKernelArg(backward_kernel, arg++, sizeof(cl_mem), &d_z);
        check_cl(err, "clSetKernelArg(sptrsv_backward_color)");

        const size_t global = static_cast<size_t>(color_count);
        check_cl(clEnqueueNDRangeKernel(queue, backward_kernel, 1, nullptr,
                                        &global, nullptr, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel(sptrsv_backward_color)");
    }

    void apply(
        const ColorSchedule& sched_L,
        const ColorSchedule& sched_U,
        const double *r_host,
        double *z_host) {
        check_cl(clEnqueueWriteBuffer(queue, d_r, CL_FALSE, 0,
                                      sizeof(double) * static_cast<size_t>(n),
                                      r_host, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer(r)");

        // In-order queue gives a global barrier between color kernels.  This is the
        // OpenCL equivalent of: for color serially, parallel-for rows in this color.
        for (int color = 0; color < sched_L.num_colors; ++color) {
            const int start = sched_L.color_ptr[color];
            const int count = sched_L.color_ptr[color + 1] - start;
            enqueue_forward_color(start, count);
        }
        for (int color = 0; color < sched_U.num_colors; ++color) {
            const int start = sched_U.color_ptr[color];
            const int count = sched_U.color_ptr[color + 1] - start;
            enqueue_backward_color(start, count);
        }

        check_cl(clEnqueueReadBuffer(queue, d_z, CL_TRUE, 0,
                                     sizeof(double) * static_cast<size_t>(n),
                                     z_host, 0, nullptr, nullptr),
                 "clEnqueueReadBuffer(z)");
    }
};

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

        // Temporary context just to learn device properties before deciding METIS block count.
        // The real solver object is created after coloring because it also uploads schedules.
        cl_uint num_platforms = 0;
        check_cl(clGetPlatformIDs(0, nullptr, &num_platforms), "clGetPlatformIDs(count)");
        if (num_platforms == 0) {
            throw std::runtime_error("No OpenCL platform found");
        }

        // Use a conservative partition target for optional METIS ordering.  It does not
        // control OpenCL parallelism; OpenCL work-item count is the number of rows in a color.
        int metis_partition_count = 32;
        if (const char *env = std::getenv("OPENCL_COLOR_PARTS")) {
            const int v = std::atoi(env);
            if (v > 0) metis_partition_count = v;
        }

        std::cout << "[Setup] Coloring triangular dependency graphs";
#ifdef USE_METIS
        std::cout << " with METIS partitioning, parts=" << metis_partition_count;
#else
        std::cout << " with row-order fallback";
#endif
        std::cout << "...\n";

        ColorSchedule sched_L = build_colored_schedule(L, graph, true, metis_partition_count);
        ColorSchedule sched_U = build_colored_schedule(U, graph, false, metis_partition_count);

        OpenClSptrsv ocl_solver(L, sched_L, U, sched_U);

        std::cout << "[Setup] Dependency coloring complete. L colors: "
                  << sched_L.num_colors << ", U colors: " << sched_U.num_colors
                  << ", OpenCL device: " << ocl_solver.device_name
                  << ", compute units: " << ocl_solver.compute_units << '\n';

        MklBuffer x(n);
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

        ocl_solver.apply(sched_L, sched_U, r.data(), z.data());
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

            ocl_solver.apply(sched_L, sched_U, r.data(), z.data());

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
        std::cout << "ICPCG OpenCL SpTRSV Solver Summary\n";
        std::cout << "==================================================\n";
        std::cout << "Matrix Size         : " << n << '\n';
        std::cout << "A Nonzeros          : " << csr.ia[n] << '\n';
        std::cout << "L Nonzeros          : " << L.ia[n] << '\n';
        std::cout << "L Colors            : " << sched_L.num_colors << '\n';
        std::cout << "U Colors            : " << sched_U.num_colors << '\n';
        std::cout << "OpenCL Device       : " << ocl_solver.device_name << '\n';
        std::cout << "OpenCL Compute Units: " << ocl_solver.compute_units << '\n';
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
