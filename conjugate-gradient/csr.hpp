#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

struct CsrMatrix {
    int nrows = 0, ncols = 0;
    std::vector<int> ia;      // row pointers, size nrows + 1
    std::vector<int> ja;      // col indices, size nnz
    std::vector<double> a;    // values, size nnz
};

CsrMatrix read_mtx_to_csr(const std::string& path);
