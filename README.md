# Collaborative-Optimization-Strategies-for-GEMV-on-PIMsimulator

## Contents
  [overview](#-overview)  
  [Setup](#-setup)  

## Overview
This repository contains the accompanying simulation code for the paper "Collaborative Optimization Strategies for Matrix Multiplication-Accumulation Operators on Commercial Processing-In-Memory Architectures". It implements core optimization strategies including dynamic bank allocation, parity-bank interleaving mapping, and partitioned virtualization computing. This work is modified based on the original project https://github.com/SAITPublic/PIMSimulator.

贺煜凯, 谢童欣, 朱振华, 高岚, 李冰. 面向商用存算一体架构矩阵乘算子协同优化策略研究[J]. 电子与信息学报, 2025, 47(9): 3187-3197. doi: 10.11999/JEIT250364

## Setup

### Prerequisites
* `Scons` tool for compiling PIMSimulator:
```bash
sudo apt install scons
```
* `gtest` for running test cases:
```bash
sudo apt install libgtest-dev
```

### Installing
* To Install PIMSimulator:
```bash
# compile
scons
```

### Launch a Test Run
* Show a list of test cases
```bash
./sim --gtest_list_tests

# Example
PIMKernelFixture.
  gemv_tree
  gemv
  mul
  add
  relu
MemBandwidthFixture.
  hbm_read_bandwidth
  hbm_write_bandwidth
PIMBenchFixture.
  gemv
  mul
  add
  relu
```

* Test Running
```bash
# Running: functionality test (GEMV)
./sim --gtest_filter=PIMKernelFixture.gemv

# Running: functionality test (MUL)
./sim --gtest_filter=PIMKernelFixture.mul

# Running: performance test (GEMV)
./sim --gtest_filter=PIMBenchFixture.gemv

# Running: performance test (ADD)
./sim --gtest_filter=PIMBenchFixture.add
```
