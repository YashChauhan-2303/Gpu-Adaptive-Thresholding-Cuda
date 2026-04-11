# 🚀 GPU-Accelerated Adaptive Thresholding using CUDA

## 📌 Overview
This project implements **Adaptive Thresholding for Document Binarization** using both:
- 🟢 CPU (sequential)
- 🔴 GPU (parallel using CUDA)

The objective is to demonstrate how **GPU parallelism significantly improves performance** for computationally intensive image processing tasks.

---

## 🧠 Problem Statement
Traditional global thresholding fails for images with:
- uneven lighting
- shadows
- faded text

👉 Adaptive thresholding solves this by computing a **local threshold for each pixel**.

---

## ⚙️ Algorithm

For each pixel `(x, y)`:

1. Take a window of size `W × W`
2. Compute mean intensity:

mean = sum(window pixels) / count

3. Compute threshold:

T = mean - C

4. Apply:
- `pixel > T → 255 (white)`
- else → `0 (black)`

---

## ⚡ CPU vs GPU

### 🟢 CPU
- Sequential processing using nested loops  
- Processes one pixel at a time  
- Computationally expensive  

### 🔴 GPU
- One thread per pixel  
- Thousands of threads run in parallel  
- Significant speedup for large images  

---

## 🧵 Parallelism Strategy
Each CUDA thread processes one pixel independently.

Threads are organized as:
- Blocks → `16 × 16`
- Grid → covers the entire image

👉 This makes the problem **embarrassingly parallel**

---

## 📊 Performance Metric

Speedup = CPU Time / GPU Time

---

## 🖼️ Input Image

### 📥 Original Image
![Input](img.avif)

---

## 🧾 Output Images

### 🟢 CPU Output
![CPU Output](output_cpu.pgm)

### 🔴 GPU Output
![GPU Output](output_gpu.pgm)

---

## 📈 Results

| Image Size | W  | CPU Time (ms) | GPU Time (ms) | Speedup |
|------------|----|---------------|---------------|---------|
| 8×8        | 7  | 0.007         | 1.33          | 0.005x  |
| 700×1145   | 7  | 109.63        | 1.68          | 65x     |
| 700×1145   | 15 | 330.52        | 1.70          | 193x    |

---

## 🔍 Key Observations

- GPU is slower for very small images due to overhead (memory transfer + kernel launch)
- GPU achieves massive speedup for large images
- Increasing window size improves GPU efficiency
- CPU and GPU outputs match exactly (correctness verified)

---

## 🛠️ Tech Stack
- C++
- CUDA
- PGM Image Processing

---

## 📁 Project Structure

```
.
├── main.cu
├── img.pgm
├── img.avif
├── output_cpu.pgm
├── output_gpu.pgm
├── sample.pgm
├── README.md
```

---

## 🧪 Build & Run

### 🔧 Compile
```bash
nvcc main.cu -o run
```

### ▶️ Run
```bash
./run img.pgm 7 5
```

### ⚠️ Notes
- Input must be PGM (P2 format)
- Convert images using ImageMagick:
```bash
convert img.avif -colorspace Gray img.pgm
```
- Convert output for viewing:
```bash
convert output_cpu.pgm output.png
convert output_gpu.pgm result.png
```
