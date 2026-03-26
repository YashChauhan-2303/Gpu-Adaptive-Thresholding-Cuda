# Results and Performance Analysis

## Experiment Setup
- Project: GPU-Accelerated Adaptive Thresholding for Document Binarization
- Input image size: 700 x 1145 (PGM grayscale)
- Program: [main.cu](main.cu)
- Executable run command format:
  - ./run <input.pgm> <window_size> <C>

## Recorded Runs

### Run 1
- Command: ./run img.pgm 7 5
- Window size (W): 7
- Constant (C): 5
- CPU time: 94.4076 ms
- GPU time: 1.40429 ms
- Speedup (CPU/GPU): 67.2281x
- Mismatched pixels (CPU vs GPU): 0

### Run 2
- Command: ./run img.pgm 15 5
- Window size (W): 15
- Constant (C): 5
- CPU time: 337.795 ms
- GPU time: 0.572544 ms
- Speedup (CPU/GPU): 589.99x
- Mismatched pixels (CPU vs GPU): 0

## Summary Table

| Input Size | W | C | CPU Time (ms) | GPU Time (ms) | Speedup | Pixel Mismatch |
|---|---:|---:|---:|---:|---:|---:|
| 700 x 1145 | 7  | 5 | 94.4076  | 1.40429  | 67.2281x | 0 |
| 700 x 1145 | 15 | 5 | 337.795  | 0.572544 | 589.99x  | 0 |

## Observations
1. GPU implementation is significantly faster than CPU in both runs.
2. CPU time increased strongly when the window size increased from 7 to 15, because each pixel processes a larger neighborhood.
3. Output correctness is confirmed by zero mismatch between CPU and GPU outputs.
4. The larger window run showed higher speedup, indicating better benefit from parallel execution for heavier per-pixel computation.

## Conclusion
The CUDA-based adaptive thresholding implementation provides major acceleration while preserving output correctness. For this test image, the measured speedup ranges from 67x to 590x depending on window size, demonstrating the effectiveness of GPU parallelization for local-window image operations.
