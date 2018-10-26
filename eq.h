#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

using namespace std;
using namespace cv;

void safely_call(cudaError err, const char *msg, const char *file_name,
                 const int line_number);

#define SAFE_CALL(call, msg) safely_call(call, msg, __FILE__, __LINE__)
