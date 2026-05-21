# 🚀 GPU-Accelerated Adaptive Thresholding using CUDA

A CUDA-based implementation of Adaptive Thresholding for document image binarization, designed to demonstrate the performance benefits of GPU parallelism for image processing workloads.

## Highlights

✅ Achieved up to **193× speedup** over CPU implementation

✅ Implemented pixel-level parallelism using **CUDA kernels**

✅ Verified CPU and GPU outputs for correctness

✅ Demonstrates efficient GPU acceleration for adaptive image thresholding

---

## Overview

Adaptive Thresholding is a document image preprocessing technique used to convert grayscale images into binary images.

Unlike global thresholding, which applies a single threshold value to the entire image, adaptive thresholding computes a local threshold for every pixel based on its surrounding neighborhood.

This makes it highly effective for:

- Uneven illumination
- Shadows
- Low-contrast documents
- Faded or degraded text

---

## Performance Results

| Image Size | Window Size | CPU Time (ms) | GPU Time (ms) | Speedup |
|------------|------------|---------------|---------------|----------|
| 8 × 8 | 7 | 0.007 | 1.33 | 0.005× |
| 700 × 1145 | 7 | 109.63 | 1.68 | 65× |
| 700 × 1145 | 15 | 330.52 | 1.70 | 193× |

### Key Observations

- GPU overhead dominates for extremely small images.
- Performance gains increase significantly as image size grows.
- Larger window sizes benefit more from parallel execution.
- CPU and GPU outputs were verified to be identical.

---

## Processing Pipeline

```text
Input Image (.avif)
        ↓
Convert to Grayscale PGM
        ↓
Adaptive Thresholding
        ↓
CPU Implementation
        ↓
GPU CUDA Implementation
        ↓
Performance Comparison
        ↓
Binary Output Image
```

---

## Problem Statement

Traditional thresholding techniques struggle when document images contain:

- Non-uniform lighting
- Background noise
- Shadows
- Faded text

Adaptive thresholding addresses these limitations by computing a threshold locally for each pixel.

---

## Algorithm

For every pixel `(x, y)`:

### Step 1

Select a local window of size:

```text
W × W
```

around the pixel.

### Step 2

Compute the local mean intensity:

```text
mean = sum(window pixels) / count
```

### Step 3

Compute the adaptive threshold:

```text
T = mean - C
```

where:

- `T` = threshold value
- `C` = constant offset

### Step 4

Apply binarization:

```text
pixel > T  → 255 (white)
otherwise  → 0 (black)
```

---

## CUDA Parallelization Strategy

The algorithm is highly parallel because each output pixel can be computed independently.

### CPU Version

- Sequential execution
- Nested loops
- One pixel processed at a time

### GPU Version

- One CUDA thread per pixel
- Thousands of threads execute concurrently
- Exploits massive parallelism available on modern GPUs

### Thread Organization

```text
Block Size : 16 × 16
Grid Size  : Covers entire image
```

This design makes adaptive thresholding an **embarrassingly parallel problem**, ideal for GPU acceleration.

---

## Sample Results

### Original Image

![Input Image](input/img.avif)

### CPU Output

![CPU Output](docs/cpu_output.png)

### GPU Output

![GPU Output](docs/gpu_output.png)

---

## Technologies Used

- CUDA
- C++
- NVIDIA GPU Programming
- Image Processing
- PGM Image Format

---

## Repository Structure

```text
.
├── src/
│   └── main.cu
│
├── input/
│   └── img.avif
│
├── data/
│   ├── img.pgm
│   └── sample.pgm
│
├── output/
│   ├── output_cpu.pgm
│   ├── output_gpu.pgm
│   └── output.png
│
├── docs/
│   ├── result.md
│   ├── result.png
│   ├── cpu_output.png
│   └── gpu_output.png
│
├── scripts/
│   ├── commands.txt
│   └── run
│
└── README.md
```

---

## Build Instructions

### Compile

```bash
nvcc src/main.cu -o adaptive_threshold
```

### Run

```bash
./adaptive_threshold data/img.pgm 7 5
```

Parameters:

```text
<input_image> <window_size> <constant_C>
```

Example:

```bash
./adaptive_threshold data/img.pgm 15 5
```

---

## Image Conversion

Convert AVIF image to grayscale PGM:

```bash
convert input/img.avif -colorspace Gray data/img.pgm
```

Convert output images for visualization:

```bash
convert output/output_cpu.pgm docs/cpu_output.png
convert output/output_gpu.pgm docs/gpu_output.png
```

---

## Future Improvements

- Shared memory optimization
- Integral image acceleration
- Multi-GPU execution
- CUDA stream-based overlapping
- Real-time document processing pipeline

---

## Author

**Yash Chauhan**

B.Tech CSE, MIT Manipal

GitHub: https://github.com/YashChauhan-2303