#include <cuda_runtime.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;
using std::vector;

// Read next token from PGM file while skipping comments starting with '#'.
static bool readNextToken(ifstream& in, string& token) {
    while (in >> token) {
        if (!token.empty() && token[0] == '#') {
            string discard;
            std::getline(in, discard);
            continue;
        }
        return true;
    }
    return false;
}

bool readPGM(const string& filename, vector<unsigned char>& image, int& width, int& height) {
    ifstream in(filename.c_str());
    if (!in.is_open()) {
        cerr << "Error: Cannot open input file: " << filename << endl;
        return false;
    }

    string token;
    if (!readNextToken(in, token) || token != "P2") {
        cerr << "Error: Unsupported or invalid PGM format. Expected P2." << endl;
        return false;
    }

    if (!readNextToken(in, token)) {
        cerr << "Error: Could not read image width." << endl;
        return false;
    }
    width = std::stoi(token);

    if (!readNextToken(in, token)) {
        cerr << "Error: Could not read image height." << endl;
        return false;
    }
    height = std::stoi(token);

    if (width <= 0 || height <= 0) {
        cerr << "Error: Invalid image dimensions." << endl;
        return false;
    }

    if (!readNextToken(in, token)) {
        cerr << "Error: Could not read max gray value." << endl;
        return false;
    }
    int maxVal = std::stoi(token);
    if (maxVal <= 0 || maxVal > 255) {
        cerr << "Error: Unsupported max gray value (must be 1..255)." << endl;
        return false;
    }

    image.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (int i = 0; i < width * height; ++i) {
        if (!readNextToken(in, token)) {
            cerr << "Error: Not enough pixel values in file." << endl;
            return false;
        }
        int value = std::stoi(token);
        if (value < 0) {
            value = 0;
        }
        if (value > maxVal) {
            value = maxVal;
        }
        // Scale to 0..255 if maxVal is not 255.
        image[i] = static_cast<unsigned char>((value * 255) / maxVal);
    }

    return true;
}

bool writePGM(const string& filename, const vector<unsigned char>& image, int width, int height) {
    if (static_cast<int>(image.size()) != width * height) {
        cerr << "Error: Output image size does not match dimensions." << endl;
        return false;
    }

    ofstream out(filename.c_str());
    if (!out.is_open()) {
        cerr << "Error: Cannot open output file: " << filename << endl;
        return false;
    }

    out << "P2\n";
    out << width << " " << height << "\n";
    out << "255\n";

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            out << static_cast<int>(image[y * width + x]);
            if (x < width - 1) {
                out << ' ';
            }
        }
        out << '\n';
    }

    return true;
}

void adaptiveThresholdCPU(const vector<unsigned char>& input,
                          vector<unsigned char>& output,
                          int width,
                          int height,
                          int windowSize,
                          int C) {
    output.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
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
float adaptiveThresholdGPU(const vector<unsigned char>& input,
                           vector<unsigned char>& output,
                           int width,
                           int height,
                           int windowSize,
                           int C,
                           dim3 blockSize = dim3(16, 16)) {
    output.resize(static_cast<size_t>(width) * static_cast<size_t>(height));

    const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(unsigned char);
    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    cudaError_t err;

    err = cudaMalloc(reinterpret_cast<void**>(&d_input), bytes);
    if (err != cudaSuccess) {
        cerr << "CUDA Error (cudaMalloc d_input): " << cudaGetErrorString(err) << endl;
        return -1.0f;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_output), bytes);
    if (err != cudaSuccess) {
        cerr << "CUDA Error (cudaMalloc d_output): " << cudaGetErrorString(err) << endl;
        cudaFree(d_input);
        return -1.0f;
    }

    err = cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "CUDA Error (cudaMemcpy H2D): " << cudaGetErrorString(err) << endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1.0f;
    }

    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    adaptiveThresholdKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, windowSize, C);
    cudaEventRecord(stop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA Error (Kernel launch): " << cudaGetErrorString(err) << endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_input);
        cudaFree(d_output);
        return -1.0f;
    }

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) {
        cerr << "CUDA Error (cudaEventSynchronize): " << cudaGetErrorString(err) << endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_input);
        cudaFree(d_output);
        return -1.0f;
    }

    float gpuMs = 0.0f;
    cudaEventElapsedTime(&gpuMs, start, stop);

    err = cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "CUDA Error (cudaMemcpy D2H): " << cudaGetErrorString(err) << endl;
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
        cout << "Usage: ./run <input.pgm> [windowSize] [C]\n";
        cout << "Example: ./run input.pgm 15 7\n";
        return 1;
    }

    string inputFile = argv[1];
    int windowSize = 15;
    int C = 7;

    if (argc >= 3) {
        windowSize = std::stoi(argv[2]);
    }
    if (argc >= 4) {
        C = std::stoi(argv[3]);
    }

    if (windowSize <= 0) {
        cerr << "Error: windowSize must be positive." << endl;
        return 1;
    }
    if (windowSize % 2 == 0) {
        // Make it odd so the window has a clear center.
        windowSize += 1;
        cout << "Info: windowSize changed to odd value: " << windowSize << endl;
    }

    vector<unsigned char> inputImage;
    int width = 0;
    int height = 0;

    if (!readPGM(inputFile, inputImage, width, height)) {
        return 1;
    }

    cout << "Loaded image: " << width << " x " << height << endl;
    cout << "Window size: " << windowSize << ", C: " << C << endl;

    vector<unsigned char> cpuOutput;
    vector<unsigned char> gpuOutput;

    auto cpuStart = std::chrono::high_resolution_clock::now();
    adaptiveThresholdCPU(inputImage, cpuOutput, width, height, windowSize, C);
    auto cpuEnd = std::chrono::high_resolution_clock::now();

    double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    dim3 blockSize(16, 16);
    float gpuMs = adaptiveThresholdGPU(inputImage, gpuOutput, width, height, windowSize, C, blockSize);
    if (gpuMs < 0.0f) {
        return 1;
    }

    if (!writePGM("output_cpu.pgm", cpuOutput, width, height)) {
        return 1;
    }
    if (!writePGM("output_gpu.pgm", gpuOutput, width, height)) {
        return 1;
    }

    int mismatchCount = 0;
    for (int i = 0; i < width * height; ++i) {
        if (cpuOutput[i] != gpuOutput[i]) {
            ++mismatchCount;
        }
    }

    cout << "CPU time: " << cpuMs << " ms" << endl;
    cout << "GPU time: " << gpuMs << " ms" << endl;
    if (gpuMs > 0.0f) {
        cout << "Speedup (CPU/GPU): " << (cpuMs / gpuMs) << "x" << endl;
    }
    cout << "Mismatched pixels (CPU vs GPU): " << mismatchCount << endl;
    cout << "Saved output_cpu.pgm and output_gpu.pgm" << endl;

    return 0;
}