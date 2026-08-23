# neutron-star — build (PLAN.md §3.3)
# Two modes:  make          (release)
#             make debug    (-O1 -g -DNS_DEBUG)
# Targets:    all bench test debug clean
# HARD RULE (PLAN §0.2): --offload-arch=gfx1201 exactly. Never gfx1200/gfx12-generic.

HIPCC    := /opt/rocm/bin/hipcc
ARCH     := --offload-arch=gfx1201
WARN     := -Wall -Wextra
RELFLAGS := -O3 -std=c++17
DBGFLAGS := -O1 -g -std=c++17 -DNS_DEBUG

MODE ?= release
ifeq ($(MODE),debug)
  OPT      := $(DBGFLAGS)
  BUILDDIR := build/debug
else
  OPT      := $(RELFLAGS)
  BUILDDIR := build/release
endif

FLAGS := $(ARCH) $(OPT) $(WARN) -Isrc -Itests -fopenmp
GIT_COMMIT := $(shell git rev-parse --short=12 HEAD 2>/dev/null || printf unknown)
GIT_DIRTY  := $(shell git diff --quiet HEAD -- 2>/dev/null || printf -- -dirty)

# ---- sources -----------------------------------------------------------------
SRCS   := $(wildcard src/*.cpp) $(wildcard src/kernels/*.hip)
OBJS   := $(patsubst %,$(BUILDDIR)/%.o,$(SRCS))
DEPS   := $(OBJS:.o=.d)
NS     := $(BUILDDIR)/ns

TESTSRCS := $(wildcard tests/*.cpp) $(wildcard tests/*.hip)
TESTBINS := $(patsubst tests/%,$(BUILDDIR)/tests/%,$(basename $(TESTSRCS)))

BENCHSRCS := $(wildcard bench/*.hip)
BENCHBINS := $(patsubst bench/%.hip,$(BUILDDIR)/%,$(BENCHSRCS))

# Oracle tools link llama.cpp's ggml (PLAN §9.1). Not part of `all` or `test`:
# ns itself depends on the HIP runtime only (§3.4).
LLAMA_DIR   ?= $(HOME)/dev/inference-engines/llama.cpp
GGML_LIBDIR := $(LLAMA_DIR)/build_vulkan/bin
TOOLSRCS    := $(wildcard tools/*.cpp)
TOOLBINS    := $(patsubst tools/%.cpp,$(BUILDDIR)/tools/%,$(TOOLSRCS))

# ---- top level ---------------------------------------------------------------
ifeq ($(strip $(SRCS)),)
all: bench
	@echo "note: no engine sources in src/ yet (Stage 0) — built benchmarks only"
else
all: $(NS)
endif

.PHONY: all bench test tools debug clean help
debug:
	@$(MAKE) --no-print-directory MODE=debug all bench test

bench: $(BENCHBINS)

# ---- engine ------------------------------------------------------------------
$(NS): $(OBJS)
	@mkdir -p $(dir $@)
	$(HIPCC) $(FLAGS) $^ -o $@

$(BUILDDIR)/%.o: %
	@mkdir -p $(dir $@)
	$(HIPCC) $(FLAGS) -MMD -MP -c $< -o $@

# Refresh build identity even when main.cpp itself is unchanged. Benchmark JSONL
# must pin the executable's source revision (PLAN section 9.4).
.PHONY: FORCE
$(BUILDDIR)/src/main.cpp.o: src/main.cpp FORCE
	@mkdir -p $(dir $@)
	$(HIPCC) $(FLAGS) -DNS_BUILD_COMMIT=\"$(GIT_COMMIT)$(GIT_DIRTY)\" \
		-MMD -MP -c $< -o $@

# ---- benchmarks --------------------------------------------------------------
$(BUILDDIR)/gemv_bench: bench/gemv_bench.hip $(filter-out $(BUILDDIR)/src/main.cpp.o,$(OBJS))
	@mkdir -p $(dir $@)
	$(HIPCC) $(FLAGS) $< -x none $(filter %.o,$^) -o $@

$(BUILDDIR)/gdn_bench: bench/gdn_bench.hip $(filter-out $(BUILDDIR)/src/main.cpp.o,$(OBJS))
	@mkdir -p $(dir $@)
	$(HIPCC) $(FLAGS) $< -x none $(filter %.o,$^) -o $@

$(BUILDDIR)/%: bench/%.hip
	@mkdir -p $(dir $@)
	$(HIPCC) $(FLAGS) $< -o $@

# ---- tests -------------------------------------------------------------------
# note: hipcc puts "-x hip" ahead of every input, so the object list needs
# "-x none" to stop it being compiled as HIP source.
$(BUILDDIR)/tests/%: tests/%.cpp $(filter-out $(BUILDDIR)/src/main.cpp.o,$(OBJS))
	@mkdir -p $(dir $@)
	$(HIPCC) $(FLAGS) $< -x none $(filter %.o,$^) -o $@

$(BUILDDIR)/tests/%: tests/%.hip $(filter-out $(BUILDDIR)/src/main.cpp.o,$(OBJS))
	@mkdir -p $(dir $@)
	$(HIPCC) $(FLAGS) $< -x none $(filter %.o,$^) -o $@

ifeq ($(strip $(TESTSRCS)),)
test:
	@echo "no tests yet (Stage 1 adds them)"
else
test: $(TESTBINS)
	@fail=0; for t in $(TESTBINS); do printf '%-40s' "$$t"; \
	  if $$t > /tmp/ns_test.log 2>&1; then echo PASS; else echo FAIL; cat /tmp/ns_test.log; fail=1; fi; \
	done; \
	for t in $(wildcard tests/*.py); do printf '%-40s' "$$t"; \
	  if python3 $$t > /tmp/ns_test.log 2>&1; then echo PASS; else echo FAIL; cat /tmp/ns_test.log; fail=1; fi; \
	done; exit $$fail
endif

tools: $(TOOLBINS)

ORACLE_BINS := $(BUILDDIR)/tools/oracle_logits $(BUILDDIR)/tools/oracle_activations
$(ORACLE_BINS): $(BUILDDIR)/tools/%: tools/%.cpp
	@mkdir -p $(dir $@)
	$(HIPCC) $(FLAGS) -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include $< \
	  -L$(GGML_LIBDIR) -lllama -lggml -lggml-base -Wl,-rpath,$(GGML_LIBDIR) -o $@

$(BUILDDIR)/tools/%: tools/%.cpp $(filter-out $(BUILDDIR)/src/main.cpp.o,$(OBJS))
	@mkdir -p $(dir $@)
	$(HIPCC) $(FLAGS) $< -x none $(filter %.o,$^) \
	  -L$(GGML_LIBDIR) -lggml-cpu -lggml-base -Wl,-rpath,$(GGML_LIBDIR) -o $@

clean:
	rm -rf build

help:
	@echo "make            release build (engine when src/ is non-empty, else benches)"
	@echo "make debug      -O1 -g -DNS_DEBUG build of everything"
	@echo "make bench      build bench/*.hip -> $(BUILDDIR)/"
	@echo "make test       build+run tests/"
	@echo "make tools      build tools/ (links llama.cpp ggml; oracle checks)"
	@echo "make clean"

-include $(DEPS)
