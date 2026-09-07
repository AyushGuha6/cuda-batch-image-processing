IDIR=include
LIBDIR=lib
SRCDIR=src
BINDIR=bin
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I$(LIBDIR) -I/usr/local/cuda/include -lcudart --std=c++17 -O2
NVCC_MAJOR := $(shell nvcc --version | sed -n 's/.*release \([0-9]*\)\.*/\1/p')
ifeq ($(shell test $(NVCC_MAJOR) -lt 13 && echo yes),yes)
COMPILER_FLAGS += -gencode arch=compute_61,code=sm_61
endif
COMPILER_FLAGS += -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86

TARGET=$(BINDIR)/batch_image_processor.exe
SOURCES=$(SRCDIR)/main.cpp $(SRCDIR)/cli_parser.cpp $(SRCDIR)/image_processor.cu

.PHONY: clean build run setup data

setup:
	bash scripts/fetch_dependencies.sh

build: setup $(TARGET)

$(TARGET): $(SOURCES) $(IDIR)/cli_parser.h $(IDIR)/image_processor.h
	mkdir -p $(BINDIR)
	$(COMPILER) $(COMPILER_FLAGS) $(SOURCES) -o $(TARGET)

data:
	python3 scripts/generate_sample_images.py --output data/input --count 128

run: build data
	bash run.sh

clean:
	rm -f $(TARGET)
	rm -rf data/output/*
	rm -f results/execution_log.txt

all: clean build
