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
ALL_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
SRCS := $(filter-out $(SRC_DIR)/renderer.cpp,$(ALL_SRCS))
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
RENDERER_SRC := $(SRC_DIR)/renderer.cpp
RENDERER_OBJ := $(BUILD_DIR)/renderer.o

# OpenGL libraries - using pkg-config for GLFW
LDFLAGS_GLFW := $(shell pkg-config --libs glfw3) -framework OpenGL
RENDERER_CXXFLAGS += $(shell pkg-config --cflags glfw3)

# Use regular g++ for renderer (not mpic++)
RENDERER_CXX ?= g++
RENDERER_CXXFLAGS ?= -Wall -Wextra -O3 -std=c++17 -I./src

.PHONY: all clean run format renderer run-renderer

all: $(TARGET)

renderer: $(RENDERER_TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(BUILD_DIR)/renderer.o: $(RENDERER_SRC) | $(BUILD_DIR)
	$(RENDERER_CXX) $(RENDERER_CXXFLAGS) -c -o $@ $<

$(RENDERER_TARGET): $(RENDERER_OBJ) | $(BIN_DIR)
	$(RENDERER_CXX) $(RENDERER_CXXFLAGS) -o $@ $< $(LDFLAGS_GLFW)

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
