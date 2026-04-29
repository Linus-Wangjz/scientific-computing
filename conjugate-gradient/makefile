# ==========================================
# 编译器与核心标志设定
# ==========================================
ONEAPI   = /opt/intel/oneapi
GCC_DIR  = /usr/lib/gcc/x86_64-linux-gnu/14

CXX      = $(ONEAPI)/compiler/latest/bin/icpx
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra \
           --gcc-install-dir=$(GCC_DIR) \
           -I$(ONEAPI)/mkl/latest/include
OPENMPFLAGS = -qopenmp
METIS_PREFIX ?= $(HOME)/local
METIS_CXXFLAGS ?= -I$(METIS_PREFIX)/include
METIS_LDFLAGS ?= -L$(METIS_PREFIX)/lib
METIS_LDLIBS ?= -lmetis -lGKlib -lm

# ==========================================
# 链接器标志设定 (MKL 魔法指令)
# ==========================================
LDFLAGS  = -L$(ONEAPI)/mkl/latest/lib \
           -Wl,-rpath,$(ONEAPI)/mkl/latest/lib \
           -Wl,-rpath,$(ONEAPI)/compiler/latest/lib \
           -qmkl=parallel

# ==========================================
# 文件与目标设定
# ==========================================
TARGETS  = cg_solver icpcg_solver icpcg_optimized_solver
CG_OBJS  = cg.o csr.o
ICPCG_OBJS = icpcg.o csr.o
ICPCG_OPT_OBJS = icpcg-optimized.o csr.o

# ==========================================
# 编译规则
# ==========================================
all: $(TARGETS)

cg_solver: $(CG_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

icpcg_solver: $(ICPCG_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

icpcg_optimized_solver: $(ICPCG_OPT_OBJS)
	$(CXX) $(CXXFLAGS) $(OPENMPFLAGS) -o $@ $^ $(LDFLAGS)

icpcg-optimized.o: icpcg-optimized.cpp
	$(CXX) $(CXXFLAGS) $(OPENMPFLAGS) -c $< -o $@

icpcg-optimized-metis.o: icpcg-optimized.cpp
	$(CXX) $(CXXFLAGS) $(OPENMPFLAGS) -DUSE_METIS $(METIS_CXXFLAGS) -c $< -o $@

icpcg_optimized_metis_solver: icpcg-optimized-metis.o csr.o
	$(CXX) $(CXXFLAGS) $(OPENMPFLAGS) -o $@ $^ $(LDFLAGS) $(METIS_LDFLAGS) $(METIS_LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f cg.o icpcg.o icpcg-optimized.o icpcg-optimized-metis.o csr.o $(TARGETS) icpcg_optimized_metis_solver

.PHONY: all clean
