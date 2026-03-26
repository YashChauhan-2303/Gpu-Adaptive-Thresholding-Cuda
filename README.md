# GPU-Accelerated Adaptive Thresholding for Document Binarization (CUDA)

## Overview
This mini-project implements **adaptive thresholding** for grayscale document images in **PGM (P2 ASCII)** format.

Two versions are provided in one CUDA source file:
- CPU implementation (baseline)
- GPU implementation using CUDA kernel (parallel)

The program reads a grayscale image, converts it to binary (black/white), and writes:
- `output_cpu.pgm`
- `output_gpu.pgm`

It also reports execution times for CPU and GPU and prints speedup.

## Problem Being Solved
Global thresholding often fails when document images have:
- uneven lighting
- shadows
- faded text

Adaptive thresholding solves this by computing a **local threshold per pixel** using neighboring pixels.

## Algorithm Used
### 1) Adaptive Thresholding (Local Mean)
For each pixel at `(x, y)`:
1. Take a local window of size `W x W` centered at `(x, y)`.
2. Compute local mean intensity:

   `mean(x, y) = sum(window pixels) / number_of_pixels_in_window`

3. Compute threshold:

   `T(x, y) = mean(x, y) - C`

4. Binarize:
- if `pixel(x, y) > T(x, y)` -> `255` (white)
- else -> `0` (black)

### 2) Image Processing View (Convolution / Box Filter Idea)
The local mean step is equivalent to applying a **box filter** (uniform averaging filter).
- Kernel size: `W x W`
- Kernel values: all equal (normalized by number of pixels)

So conceptually this is similar to image convolution with an averaging kernel, then thresholding the original image using that local average.

### 3) CPU vs GPU Strategy
- CPU: nested loops over all pixels and their local windows.
- GPU: one CUDA thread handles one output pixel.
  - Thread index is computed with `blockIdx`, `blockDim`, `threadIdx`.
  - Each thread safely clamps its window bounds at image borders.

## CUDA Implementation Notes
- Memory management uses:
  - `cudaMalloc`
  - `cudaMemcpy`
  - `cudaFree`
- GPU timing uses `cudaEvent_t`.
- CPU timing uses `std::chrono`.
- Recommended block size: `16 x 16`.

## Parallelism Strategy

In the GPU implementation, each thread is assigned to process exactly one pixel of the output image. Since each pixel operation is independent, the algorithm is highly parallelizable.

Threads are organized in a 2D grid of blocks (typically 16×16 threads per block), allowing thousands of pixels to be processed simultaneously.

## Performance Metric

Speedup is calculated as:

Speedup = CPU Execution Time / GPU Execution Time

## Key Observations

- GPU performance is slower for very small images due to overheads such as memory transfer and kernel launch latency.
- For larger images, GPU significantly outperforms CPU due to parallel processing.
- Increasing window size (W) increases computational load, which benefits GPU performance more than CPU.
- CPU and GPU outputs match exactly, confirming correctness of implementation.

## File Format
Input image must be **PGM P2 (ASCII)** grayscale.

Example header:
```
P2
640 480
255
... pixel values ...
```

## Build Instructions
Compile with nvcc:

```bash
nvcc main.cu -o run
```

## Run Instructions
Basic run:

```bash
./run input.pgm
```

Run with custom parameters:

```bash
./run input.pgm 15 7
```

Where:
- first argument: input PGM file
- second argument: window size `W` (odd value preferred)
- third argument: constant `C`

## Program Output
Console output includes:
- image size
- chosen `W` and `C`
- CPU time (ms)
- GPU time (ms)
- Speedup = CPU / GPU
- mismatch count between CPU and GPU outputs

Generated files:
- `output_cpu.pgm`
- `output_gpu.pgm`

## Example Input
A small sample is included as `sample.pgm`.

You can test quickly with:

```bash
./run sample.pgm 7 5
```

## Notes on Performance
For very small images, GPU may look slower due to kernel launch and memory transfer overhead.
For larger images, GPU parallelism provides significant performance improvements for larger images.
Special care is taken at image boundaries to ensure that window operations do not access out-of-bounds memory.

## Source
Main implementation file:
- `main.cu`

## License
This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this code for academic and personal purposes, provided that the original copyright and license notice are retained.
