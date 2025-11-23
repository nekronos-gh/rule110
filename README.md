# Rule 110 Fast Algorithm

A high-performance implementation of Rule 110 cellular automaton using AVX2 SIMD instructions and advanced bit manipulation techniques.

## Overview

This project implements an optimized Rule 110 simulator that leverages Intel AVX2 intrinsics to achieve significant performance improvements over traditional lookup-table approaches. The implementation focuses on minimizing operations through Boolean algebra simplification and efficient bit-level parallelism.

[Video Explanation](https://youtu.be/CbsLsx1AKcs?si=-kGaAxW40ml2VPKn)

## Performance Characteristics

- **Memory Access**: Aligned 256-bit loads/stores for optimal cache performance
- **Bit Operations**: Reduced to 3 logical operations per cell (XOR, ANDNOT, OR)
- **Parallelism**: Processes 256 cells simultaneously per AVX2 operation
- **Scalability**: OpenMP parallelization across multiple groups


```text
┌──────────┬──────────┬──────────┬──────────┐
│  Lane 0  │  Lane 1  │  Lane 2  │  Lane 3  │
│ (64 bits)│ (64 bits)│ (64 bits)│ (64 bits)│
└──────────┴──────────┴──────────┴──────────┘
↑----------------- 256 bits ----------------↑
```

## Requirements
- **C++20 compiler** (GCC 11+ or Clang 12+)
- **AVX2-capable CPU** (Intel Haswell 2013+ or AMD Excavator 2015+)
- **OpenMP support**

