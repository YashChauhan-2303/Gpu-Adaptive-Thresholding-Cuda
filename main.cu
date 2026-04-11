#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Read next token from PGM file while skipping comments starting with '#'.
static int readNextToken(FILE* in, char* token) {
    int c;
    while ((c = fgetc(in)) != EOF) {
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
            continue;
        }
        if (c == '#') {
            while ((c = fgetc(in)) != EOF && c != '\n') {
                // skip comment until newline
            }
            continue;
        }
        ungetc(c, in);
        if (fscanf(in, "%255s", token) == 1) {
            return 1;
        }
        return 0;
    }
    return 0;
}

int readPGM(const char* filename, unsigned char** image_out, int* width, int* height) {
    FILE* in = fopen(filename, "r");
    if (!in) {
        fprintf(stderr, "Error: Cannot open input file: %s\n", filename);
        return 0;
    }

    char token[256];
    if (!readNextToken(in, token) || strcmp(token, "P2") != 0) {
        fprintf(stderr, "Error: Unsupported or invalid PGM format. Expected P2.\n");
        fclose(in);
        return 0;
    }

    if (!readNextToken(in, token)) {
        fprintf(stderr, "Error: Could not read image width.\n");
        fclose(in);
        return 0;
    }
    *width = atoi(token);

    if (!readNextToken(in, token)) {
        fprintf(stderr, "Error: Could not read image height.\n");
        fclose(in);
        return 0;
    }
    *height = atoi(token);

    if (*width <= 0 || *height <= 0) {
        fprintf(stderr, "Error: Invalid image dimensions.\n");
        fclose(in);
        return 0;
    }

    if (!readNextToken(in, token)) {
        fprintf(stderr, "Error: Could not read max gray value.\n");
        fclose(in);
        return 0;
    }
    int maxVal = atoi(token);
    if (maxVal <= 0 || maxVal > 255) {
        fprintf(stderr, "Error: Unsupported max gray value (must be 1..255).\n");
        fclose(in);
        return 0;
    }

    size_t size = (size_t)(*width) * (size_t)(*height);
    *image_out = (unsigned char*)malloc(size);
    if (!*image_out) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        fclose(in);
        return 0;
    }

    for (int i = 0; i < *width * *height; ++i) {
        if (!readNextToken(in, token)) {
            fprintf(stderr, "Error: Not enough pixel values in file.\n");
            free(*image_out);
            *image_out = NULL;
            fclose(in);
            return 0;
        }
        int value = atoi(token);
        if (value < 0) {
            value = 0;
        }
        if (value > maxVal) {
            value = maxVal;
        }
        // Scale to 0..255 if maxVal is not 255.
        (*image_out)[i] = (unsigned char)((value * 255) / maxVal);
    }

    fclose(in);
    return 1;
}

int writePGM(const char* filename, const unsigned char* image, int width, int height) {
    if (!image) {
        fprintf(stderr, "Error: Invalid image array.\n");
        return 0;
    }

    FILE* out = fopen(filename, "w");
    if (!out) {
        fprintf(stderr, "Error: Cannot open output file: %s\n", filename);
        return 0;
    }

    fprintf(out, "P2\n");
    fprintf(out, "%d %d\n", width, height);
    fprintf(out, "255\n");

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            fprintf(out, "%d", (int)image[y * width + x]);
            if (x < width - 1) {
                fputc(' ', out);
            }
        }
        fputc('\n', out);
    }

    fclose(out);
    return 1;
}

void adaptiveThresholdCPU(const unsigned char* input,
                          unsigned char* output,
                          int width,
                          int height,
                          int windowSize,
                          int C) {
    int radius = windowSize / 2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            int count = 0;

            int yStart = (y - radius < 0) ? 0 : (y - radius);
            int yEnd = (y + radius >= height) ? (height - 1) : (y + radius);
            int xStart = (x - radius < 0) ? 0 : (x - radius);
            int xEnd = (x + radius >= width) ? (width - 1) : (x + radius);

            for (int ny = yStart; ny <= yEnd; ++ny) {
                for (int nx = xStart; nx <= xEnd; ++nx) {
                    sum += input[ny * width + nx];
                    ++count;
                }
            }

            int mean = sum / count;
            int threshold = mean - C;
            int idx = y * width + x;
            output[idx] = (input[idx] > threshold) ? 255 : 0;
        }
    }
}

__global__ void adaptiveThresholdKernel(const unsigned char* input,
                                        unsigned char* output,
                                        int width,
                                        int height,
                                        int windowSize,
                                        int C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int radius = windowSize / 2;
    int yStart = (y - radius < 0) ? 0 : (y - radius);
    int yEnd = (y + radius >= height) ? (height - 1) : (y + radius);
    int xStart = (x - radius < 0) ? 0 : (x - radius);
    int xEnd = (x + radius >= width) ? (width - 1) : (x + radius);

    int sum = 0;
    int count = 0;

    for (int ny = yStart; ny <= yEnd; ++ny) {
        for (int nx = xStart; nx <= xEnd; ++nx) {
            sum += input[ny * width + nx];
            ++count;
        }
    }

    int mean = sum / count;
    int threshold = mean - C;
    int idx = y * width + x;
    output[idx] = (input[idx] > threshold) ? 255 : 0;
}

// Wrapper that handles device memory, kernel launch, and timing.
float adaptiveThresholdGPU(const unsigned char* input,
                           unsigned char* output,
                           int width,
                           int height,
                           int windowSize,
                           int C,
                           dim3 blockSize) {
    size_t bytes = (size_t)width * (size_t)height * sizeof(unsigned char);
    unsigned char* d_input = NULL;
    unsigned char* d_output = NULL;

    cudaError_t err;

    err = cudaMalloc((void**)&d_input, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (cudaMalloc d_input): %s\n", cudaGetErrorString(err));
        return -1.0f;
    }

    err = cudaMalloc((void**)&d_output, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (cudaMalloc d_output): %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        return -1.0f;
    }

    err = cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (cudaMemcpy H2D): %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return -1.0f;
    }

    dim3 gridSize;
    gridSize.x = (width + blockSize.x - 1) / blockSize.x;
    gridSize.y = (height + blockSize.y - 1) / blockSize.y;
    gridSize.z = 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    adaptiveThresholdKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, windowSize, C);
    cudaEventRecord(stop, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (Kernel launch): %s\n", cudaGetErrorString(err));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_input);
        cudaFree(d_output);
        return -1.0f;
    }

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (cudaEventSynchronize): %s\n", cudaGetErrorString(err));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_input);
        cudaFree(d_output);
        return -1.0f;
    }

    float gpuMs = 0.0f;
    cudaEventElapsedTime(&gpuMs, start, stop);

    err = cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (cudaMemcpy D2H): %s\n", cudaGetErrorString(err));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_input);
        cudaFree(d_output);
        return -1.0f;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);

    return gpuMs;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: ./run <input.pgm> [windowSize] [C]\n");
        printf("Example: ./run input.pgm 15 7\n");
        return 1;
    }

    const char* inputFile = argv[1];
    int windowSize = 15;
    int C = 7;

    if (argc >= 3) {
        windowSize = atoi(argv[2]);
    }
    if (argc >= 4) {
        C = atoi(argv[3]);
    }

    if (windowSize <= 0) {
        fprintf(stderr, "Error: windowSize must be positive.\n");
        return 1;
    }
    if (windowSize % 2 == 0) {
        // Make it odd so the window has a clear center.
        windowSize += 1;
        printf("Info: windowSize changed to odd value: %d\n", windowSize);
    }

    unsigned char* inputImage = NULL;
    int width = 0;
    int height = 0;

    if (!readPGM(inputFile, &inputImage, &width, &height)) {
        return 1;
    }

    printf("Loaded image: %d x %d\n", width, height);
    printf("Window size: %d, C: %d\n", windowSize, C);

    size_t size = (size_t)width * (size_t)height;
    unsigned char* cpuOutput = (unsigned char*)malloc(size);
    unsigned char* gpuOutput = (unsigned char*)malloc(size);

    if (!cpuOutput || !gpuOutput) {
        fprintf(stderr, "Error: Memory allocation failed for output images.\n");
        if (inputImage) free(inputImage);
        if (cpuOutput) free(cpuOutput);
        if (gpuOutput) free(gpuOutput);
        return 1;
    }

    clock_t cpuStart = clock();
    adaptiveThresholdCPU(inputImage, cpuOutput, width, height, windowSize, C);
    clock_t cpuEnd = clock();

    double cpuMs = (double)(cpuEnd - cpuStart) / CLOCKS_PER_SEC * 1000.0;

    dim3 blockSize;
    blockSize.x = 16;
    blockSize.y = 16;
    blockSize.z = 1;

    float gpuMs = adaptiveThresholdGPU(inputImage, gpuOutput, width, height, windowSize, C, blockSize);
    if (gpuMs < 0.0f) {
        free(inputImage);
        free(cpuOutput);
        free(gpuOutput);
        return 1;
    }

    if (!writePGM("output_cpu.pgm", cpuOutput, width, height)) {
        free(inputImage);
        free(cpuOutput);
        free(gpuOutput);
        return 1;
    }
    if (!writePGM("output_gpu.pgm", gpuOutput, width, height)) {
        free(inputImage);
        free(cpuOutput);
        free(gpuOutput);
        return 1;
    }

    int mismatchCount = 0;
    for (int i = 0; i < width * height; ++i) {
        if (cpuOutput[i] != gpuOutput[i]) {
            ++mismatchCount;
        }
    }

    printf("CPU time: %f ms\n", cpuMs);
    printf("GPU time: %f ms\n", gpuMs);
    if (gpuMs > 0.0f) {
        printf("Speedup (CPU/GPU): %fx\n", cpuMs / gpuMs);
    }
    printf("Mismatched pixels (CPU vs GPU): %d\n", mismatchCount);
    printf("Saved output_cpu.pgm and output_gpu.pgm\n");

    free(inputImage);
    free(cpuOutput);
    free(gpuOutput);

    return 0;
}