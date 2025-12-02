## Top-level Makefile for the pickleball project
# Builds all .cpp files in src/ using mpic++ (or the compiler set in CXX)

CXX ?= mpic++
CXXFLAGS ?= -Wall -Wextra -O3 -std=c++20 -I./src

SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin
TARGET := $(BIN_DIR)/ballflight
RENDERER_TARGET := $(BIN_DIR)/renderer

# Get all .cpp files except renderer.cpp for ballflight target
ALL_SRCS := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/cuda/*.cpp)
SRCS := $(filter-out $(SRC_DIR)/renderer.cpp,$(ALL_SRCS))
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
RENDERER_SRC := $(SRC_DIR)/renderer.cpp
RENDERER_OBJ := $(BUILD_DIR)/renderer.o

# OpenGL libraries - using pkg-config for GLFW
ifeq ($(shell uname -s),Darwin)
LDFLAGS_GLFW := $(shell pkg-config --libs glfw3) -framework OpenGL
else
LDFLAGS_GLFW := $(shell pkg-config --libs glfw3) -lGLEW -lGL -ldl
RENDERER_CXXFLAGS += $(shell pkg-config --cflags glew)
endif
RENDERER_CXXFLAGS += $(shell pkg-config --cflags glfw3)

# Use regular g++ for renderer (not mpic++)
RENDERER_CXX ?= g++
RENDERER_CXXFLAGS ?= -Wall -Wextra -O3 -std=c++17 -I./src

ifeq ($(DEBUG),1)
RENDERER_CXXFLAGS += -DDEBUG -g -O0
endif

# Optional CUDA support: pass CUDA=1 to enable. This compiles .cu files with nvcc
# and links CUDA objects into the renderer target. If CUDA=1 is not set the
# default build is unchanged.
ifeq ($(CUDA),1)
NVCC ?= nvcc
CUDA_ARCH ?= sm_60
CUDA_CFLAGS ?= -O3 -arch=$(CUDA_ARCH) -Xcompiler "-fPIC"
CUDA_LDFLAGS ?= -L/usr/local/cuda/lib64 -lcudart -Wl,-rpath,/usr/local/cuda/lib64
CUDA_SRCS := $(wildcard $(SRC_DIR)/cuda/*.cu)
CUDA_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.cu.o,$(CUDA_SRCS))
RENDERER_CXXFLAGS += -DHAVE_CUDA
endif

.PHONY: all clean run format renderer run-renderer

all: $(TARGET)

renderer: $(RENDERER_TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(BUILD_DIR)/renderer.o: $(RENDERER_SRC) | $(BUILD_DIR)
	$(RENDERER_CXX) $(RENDERER_CXXFLAGS) -c -o $@ $<

# Rule to build CUDA object files (only used when CUDA=1)
$(BUILD_DIR)/%.cu.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(NVCC) $(CUDA_CFLAGS) -c -o $@ $<

RENDERER_EXTRA_OBJS := $(BUILD_DIR)/cuda/cpu_vecadd.o

$(RENDERER_TARGET): $(RENDERER_OBJ) $(RENDERER_EXTRA_OBJS) $(CUDA_OBJS) | $(BIN_DIR)
	$(RENDERER_CXX) $(RENDERER_CXXFLAGS) -o $@ $^ $(LDFLAGS_GLFW) $(CUDA_LDFLAGS)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) *.o

run: all
	./$(TARGET)

run-renderer: renderer
	./$(RENDERER_TARGET)

# Convenience: format sources with clang-format if installed
format:
	command -v clang-format >/dev/null 2>&1 && clang-format -i $(SRCS) || echo "clang-format not found"

.SILENT: 
