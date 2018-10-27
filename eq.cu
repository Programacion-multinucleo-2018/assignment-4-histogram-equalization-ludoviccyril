#include "eq.h"

__global__ void build_histogram(unsigned char *original, int *histogram,
                                int rows, int cols, int step) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int i = y * step + x;

  int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

  __shared__ int sub_histogram[256];

  if (thread_id < 256) {
    sub_histogram[thread_id] = 0;
  }

  __syncthreads();

  if ((x < cols) && (y < rows)) {
    atomicAdd(&sub_histogram[original[i]], 1);
  }

  __syncthreads();

  if (thread_id < 256) {
    atomicAdd(&histogram[thread_id], sub_histogram[thread_id]);
  }
}

__global__ void equalize_histogram(int *histogram, int *equalized_histogram,
                                   int rows, int cols) {
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
  if (thread_id < 256) {
    int sum = 0;
    for (int i = 0; i <= thread_id; i++) {
      sum += histogram[i];
    }
    equalized_histogram[thread_id] = sum * 255 / (rows * cols);
  }
}

__global__ void build_image(unsigned char *original, unsigned char *equalized,
                            int *equalized_histogram, int rows, int cols,
                            int step) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int i = y * step + x;

  if ((x < cols) && (y < rows)) {
    equalized[i] = equalized_histogram[original[i]];
  }
}

void equalize(Mat &original, Mat &equalized) {

  int bytes = original.step * original.rows;
  int histogram[256] = {};

  unsigned char *d_original, *d_equalized;
  int *d_histogram, *d_equalized_histogram;

  // Allocate device memory
  SAFE_CALL(cudaMalloc<unsigned char>(&d_original, bytes),
            "CUDA Malloc failed");
  SAFE_CALL(cudaMalloc<unsigned char>(&d_equalized, bytes),
            "CUDA Malloc failed");
  SAFE_CALL(cudaMalloc<int>(&d_histogram, sizeof(int) * 256),
            "CUDA Malloc failed");
  SAFE_CALL(cudaMalloc<int>(&d_equalized_histogram, sizeof(int) * 256),
            "CUDA Malloc failed");

  SAFE_CALL(
      cudaMemcpy(d_original, original.ptr(), bytes, cudaMemcpyHostToDevice),
      "CUDA memcpy host to device failed");
  cudaMemcpy(d_equalized, equalized.ptr(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_histogram, histogram, sizeof(int) * 256, cudaMemcpyHostToDevice);
  cudaMemcpy(d_equalized_histogram, histogram, sizeof(int) * 256,
             cudaMemcpyHostToDevice);

  auto start = chrono::high_resolution_clock::now();

  const dim3 block(32, 32);
  const dim3 grid((int)ceil((float)original.cols / block.x),
                  (int)ceil((float)original.rows / block.y));

  // STEP 1

  build_histogram<<<grid, block>>>(d_original, d_histogram, original.rows,
                                   original.cols, original.step);

  // STEP 2

  equalize_histogram<<<1, block>>>(d_histogram, d_equalized_histogram,
                                   original.rows, original.cols);

  // STEP 3

  build_image<<<grid, block>>>(d_original, d_equalized, d_equalized_histogram,
                               original.rows, original.cols, original.step);

  SAFE_CALL(cudaDeviceSynchronize(), "Kernel launch failed");

  auto end = chrono::high_resolution_clock::now();

  SAFE_CALL(
      cudaMemcpy(equalized.ptr(), d_equalized, bytes, cudaMemcpyDeviceToHost),
      "CUDA memcpy host to device failed");

  SAFE_CALL(cudaFree(d_original), "CUDA free failed");
  SAFE_CALL(cudaFree(d_equalized), "CUDA free failed");
  SAFE_CALL(cudaFree(d_histogram), "CUDA free failed");
  SAFE_CALL(cudaFree(d_equalized_histogram), "CUDA free failed");

  chrono::duration<float, std::milli> duration = end - start;

  cout << "Duration on GPU: " << duration.count() << "ms." << endl;
}

int main(int argc, char *argv[]) {
  string path;

  if (argc > 1) {
    path += argv[1];
  } else {
    cout << "Please specify an image path." << endl;
    return 0;
  }

  Mat original = imread(path, IMREAD_GRAYSCALE);
  Mat equalized = original.clone();

  equalize(original, equalized);

  // imshow("ORIGINAL", original);
  // imshow("EQUALIZED", equalized);

  imwrite("output.jpg", equalized);

  // waitKey();

  return 0;
}
