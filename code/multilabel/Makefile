GNN_HOME=../../graphnn
include $(GNN_HOME)/make_common
USE_GPU = 0

include_dirs = $(CUDA_HOME)/include $(MKL_ROOT)/include $(GNN_HOME)/include ./include

CXXFLAGS += $(addprefix -I,$(include_dirs)) -Wno-unused-local-typedef
cpp_files = $(shell $(FIND) src/lib -name "*.cpp" -printf "%P\n")
cxx_obj_files = $(subst .cpp,.o,$(cpp_files))
objs = $(addprefix $(obj_build_root)/cxx/,$(cxx_obj_files))

ifeq ($(USE_GPU), 1)
    CXXFLAGS += -DUSE_GPU
    NVCCFLAGS += -DUSE_GPU
    NVCCFLAGS += $(addprefix -I,$(include_dirs))
    NVCCFLAGS += -std=c++11 --use_fast_math
    cu_files = $(shell $(FIND) src/lib -name "*.cu" -printf "%P\n")
    cu_obj_files = $(subst .cu,.o,$(cu_files))
    objs += $(addprefix $(obj_build_root)/cuda/,$(cu_obj_files))
    lib_dir = $(GNN_HOME)/build/lib
else
    lib_dir = $(GNN_HOME)/build_cpuonly/lib
endif

gnn_lib = $(lib_dir)/libgnn.a

obj_build_root = build/objs

DEPS = $(objs:.o=.d)

target = build/main
target_dep = $(addsuffix .d,$(target))

.PRECIOUS: $(obj_build_root)/cuda/%.o $(obj_build_root)/cxx/%.o

all: $(target)

build/%: src/%.cpp $(gnn_lib) $(objs)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -MMD -o $@ $(filter %.cpp %.o, $^) -L$(lib_dir) -lgnn $(LDFLAGS)

DEPS += $(target_dep)

ifeq ($(USE_GPU), 1)
$(obj_build_root)/cuda/%.o: src/lib/%.cu
	$(dir_guard)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -M $< -o ${@:.o=.d} -odir $(@D)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@
endif

$(obj_build_root)/cxx/%.o: src/lib/%.cpp
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -MMD -c -o $@ $(filter %.cpp, $^)

clean:
	rm -rf build

-include $(DEPS)
