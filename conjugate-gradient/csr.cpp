#include "csr.hpp"

static std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

CsrMatrix read_mtx_to_csr(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("cannot open " + path);

    std::string line;
    std::getline(fin, line);

    std::istringstream hs(line);
    std::string banner, object, format, field, symmetry;
    hs >> banner >> object >> format >> field >> symmetry;
    if (banner != "%%MatrixMarket" || object != "matrix" || format != "coordinate") {
        throw std::runtime_error("only Matrix Market coordinate format is supported");
    }

    while (std::getline(fin, line)) {
        line = trim(line);
        if (!line.empty() && line[0] != '%') break;
    }

    int nrows, ncols, nnz_file;
    {
        std::istringstream ds(line);
        ds >> nrows >> ncols >> nnz_file;
    }

    std::vector<std::tuple<int,int,double>> trips;
    trips.reserve(nnz_file);

    for (int k = 0; k < nnz_file; ++k) {
        std::getline(fin, line);
        line = trim(line);
        if (line.empty() || line[0] == '%') {
            --k;
            continue;
        }

        std::istringstream ls(line);
        int i, j;
        double v = 1.0;
        ls >> i >> j;
        if (field != "pattern") ls >> v;

        --i;  // Matrix Market is 1-based
        --j;
        trips.emplace_back(i, j, v);

        if (symmetry == "symmetric" && i != j) {
            trips.emplace_back(j, i, v);
        }
    }

    std::sort(trips.begin(), trips.end());

    CsrMatrix csr;
    csr.nrows = nrows;
    csr.ncols = ncols;
    csr.ia.assign(nrows + 1, 0);

    for (const auto& [i, j, v] : trips) {
        csr.ia[i + 1]++;
    }
    for (int i = 0; i < nrows; ++i) {
        csr.ia[i + 1] += csr.ia[i];
    }

    csr.ja.resize(trips.size());
    csr.a.resize(trips.size());

    std::vector<int> offset = csr.ia;
    for (const auto& [i, j, v] : trips) {
        int p = offset[i]++;
        csr.ja[p] = j;
        csr.a[p] = v;
    }

    return csr;
}
