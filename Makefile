## Top-level Makefile for the pickleball project
# Builds all .cpp files in src/ using mpic++ (or the compiler set in CXX)

CXX ?= mpic++
CXXFLAGS ?= -Wall -Wextra -O3 -std=c++20 -I./src

SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin
TARGET := $(BIN_DIR)/ballflight

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

.PHONY: all clean run format

all: $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) *.o

run: all
	./$(TARGET)

# Convenience: format sources with clang-format if installed
format:
	command -v clang-format >/dev/null 2>&1 && clang-format -i $(SRCS) || echo "clang-format not found"

.SILENT: 
