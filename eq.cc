#include <chrono>
#include <iostream>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void equalize(Mat *original, Mat *equalized) {
  int histogram[256] = {};

  for (int i = 0; i < original->step * original->rows; i++) {
    histogram[original->data[i]]++;
  }

  int sum = 0;
  for (int i = 0; i < 256; i++) {
    sum += histogram[i];
    histogram[i] = sum * 255 / (original->step * original->rows);
  }

  for (int i = 0; i < original->step * original->rows; i++) {
    equalized->data[i] = histogram[original->data[i]];
  }
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

  auto start = std::chrono::high_resolution_clock::now();
  equalize(&original, &equalized);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<float, std::milli> duration = end - start;
  cout << "Duration on CPU: " << duration.count() << "ms." << endl;

  imshow("ORIGINAL", original);
  imshow("EQUALIZED", equalized);

  waitKey();

  return 0;
}