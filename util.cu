#include "eq.h"

void safely_call(cudaError err, const char *msg, const char *file_name,
                 const int line_number) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg,
            file_name, line_number, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}