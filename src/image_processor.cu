#include "image_processor.h"
#include "platform_compat.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <cuda_runtime.h>

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#ifdef LoadImage
#undef LoadImage
#endif

namespace batch_image {
namespace {

constexpr int kRgbChannels = 3;
constexpr int kGrayChannels = 1;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " "    \
                << cudaGetErrorString(err) << "\n";                          \
      return false;                                                          \
    }                                                                        \
  } while (0)

__global__ void RgbToGrayscaleKernel(const unsigned char* rgb,
                                     unsigned char* gray,
                                     int width,
                                     int height) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_pixels = width * height;
  if (idx >= num_pixels) {
    return;
  }

  int base = idx * kRgbChannels;
  float r = static_cast<float>(rgb[base]);
  float g = static_cast<float>(rgb[base + 1]);
  float b = static_cast<float>(rgb[base + 2]);
  gray[idx] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
}

__global__ void HorizontalBlurKernel(const unsigned char* input,
                                     unsigned char* output,
                                     int width,
                                     int height,
                                     int channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_pixels = width * height;
  if (idx >= num_pixels) {
    return;
  }

  int row = idx / width;
  int col = idx % width;
  int base = idx * channels;

  for (int c = 0; c < channels; ++c) {
    if (col == 0 || col == width - 1) {
      output[base + c] = input[base + c];
      continue;
    }

    int left = (idx - 1) * channels + c;
    int center = base + c;
    int right = (idx + 1) * channels + c;
    int sum = static_cast<int>(input[left]) + static_cast<int>(input[center]) +
              static_cast<int>(input[right]);
    output[base + c] = static_cast<unsigned char>(sum / 3);
  }
}

bool HasSupportedExtension(const std::string& filename) {
  const std::string lower = [&filename]() {
    std::string value = filename;
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
  }();

  return lower.find(".png") != std::string::npos ||
         lower.find(".jpg") != std::string::npos ||
         lower.find(".jpeg") != std::string::npos;
}

bool EnsureDirectoryExists(const std::string& path) {
  struct stat info {};
  if (stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode)) {
    return true;
  }
  return PlatformMkdir(path.c_str(), 0755) == 0;
}

double ElapsedMs(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

ImageProcessor::ImageProcessor(int threads_per_block)
    : threads_per_block_(threads_per_block) {}

bool ImageProcessor::LoadImage(const std::string& path, ImageBuffer* image) {
  if (image == nullptr) {
    return false;
  }

  int width = 0;
  int height = 0;
  int channels = 0;
  unsigned char* data =
      stbi_load(path.c_str(), &width, &height, &channels, kRgbChannels);
  if (data == nullptr) {
    std::cerr << "Failed to load image: " << path << "\n";
    return false;
  }

  image->width = width;
  image->height = height;
  image->channels = kRgbChannels;
  image->rgb.assign(data, data + width * height * kRgbChannels);
  image->gray.assign(width * height, 0);
  stbi_image_free(data);
  return true;
}

bool ImageProcessor::SaveImage(const std::string& path,
                               const ImageBuffer& image,
                               bool save_grayscale) {
  if (save_grayscale) {
    return stbi_write_png(path.c_str(), image.width, image.height,
                          kGrayChannels, image.gray.data(),
                          image.width * kGrayChannels) != 0;
  }

  return stbi_write_png(path.c_str(), image.width, image.height, kRgbChannels,
                        image.rgb.data(),
                        image.width * kRgbChannels) != 0;
}

void ImageProcessor::RunGrayscaleKernel(unsigned char* device_rgb,
                                        unsigned char* device_gray,
                                        int width,
                                        int height,
                                        cudaStream_t stream) {
  int num_pixels = width * height;
  int blocks = (num_pixels + threads_per_block_ - 1) / threads_per_block_;
  RgbToGrayscaleKernel<<<blocks, threads_per_block_, 0, stream>>>(
      device_rgb, device_gray, width, height);
}

void ImageProcessor::RunBlurKernel(unsigned char* device_in,
                                   unsigned char* device_out,
                                   int width,
                                   int height,
                                   int channels,
                                   cudaStream_t stream) {
  int num_pixels = width * height;
  int blocks = (num_pixels + threads_per_block_ - 1) / threads_per_block_;
  HorizontalBlurKernel<<<blocks, threads_per_block_, 0, stream>>>(
      device_in, device_out, width, height, channels);
}

bool ImageProcessor::ProcessImageFile(const std::string& input_path,
                                      const std::string& output_path,
                                      bool apply_grayscale,
                                      bool apply_blur,
                                      double* gpu_ms,
                                      double* io_ms) {
  auto io_start = std::chrono::steady_clock::now();
  ImageBuffer image;
  if (!LoadImage(input_path, &image)) {
    return false;
  }
  auto io_after_load = std::chrono::steady_clock::now();

  const int num_pixels = image.width * image.height;
  const size_t rgb_bytes = num_pixels * kRgbChannels;
  const size_t gray_bytes = num_pixels * kGrayChannels;

  unsigned char* device_rgb = nullptr;
  unsigned char* device_work_a = nullptr;
  unsigned char* device_work_b = nullptr;
  unsigned char* device_gray = nullptr;

  CUDA_CHECK(cudaMalloc(&device_rgb, rgb_bytes));
  CUDA_CHECK(cudaMalloc(&device_work_a, rgb_bytes));
  CUDA_CHECK(cudaMalloc(&device_work_b, rgb_bytes));
  CUDA_CHECK(cudaMalloc(&device_gray, gray_bytes));

  CUDA_CHECK(cudaMemcpy(device_rgb, image.rgb.data(), rgb_bytes,
                        cudaMemcpyHostToDevice));

  auto gpu_start = std::chrono::steady_clock::now();

  unsigned char* current_rgb = device_rgb;
  unsigned char* blur_src = device_work_a;
  unsigned char* blur_dst = device_work_b;

  if (apply_blur) {
    CUDA_CHECK(cudaMemcpy(blur_src, device_rgb, rgb_bytes,
                          cudaMemcpyDeviceToDevice));
    RunBlurKernel(blur_src, blur_dst, image.width, image.height, kRgbChannels,
                  0);
    CUDA_CHECK(cudaGetLastError());
    current_rgb = blur_dst;
  }

  if (apply_grayscale) {
    RunGrayscaleKernel(current_rgb, device_gray, image.width, image.height, 0);
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  auto gpu_end = std::chrono::steady_clock::now();

  if (apply_grayscale) {
    CUDA_CHECK(cudaMemcpy(image.gray.data(), device_gray, gray_bytes,
                          cudaMemcpyDeviceToHost));
  } else {
    CUDA_CHECK(cudaMemcpy(image.rgb.data(), current_rgb, rgb_bytes,
                          cudaMemcpyDeviceToHost));
  }

  cudaFree(device_rgb);
  cudaFree(device_work_a);
  cudaFree(device_work_b);
  cudaFree(device_gray);

  if (!SaveImage(output_path, image, apply_grayscale)) {
    std::cerr << "Failed to save image: " << output_path << "\n";
    return false;
  }

  auto io_end = std::chrono::steady_clock::now();
  if (gpu_ms != nullptr) {
    *gpu_ms = ElapsedMs(gpu_start, gpu_end);
  }
  if (io_ms != nullptr) {
    *io_ms = ElapsedMs(io_start, io_after_load) + ElapsedMs(gpu_end, io_end);
  }
  return true;
}

int ImageProcessor::ProcessDirectory(const std::string& input_dir,
                                     const std::string& output_dir,
                                     bool apply_grayscale,
                                     bool apply_blur,
                                     int max_images,
                                     BatchStats* stats) {
  if (stats == nullptr) {
    return 0;
  }

  if (!EnsureDirectoryExists(output_dir)) {
    std::cerr << "Unable to create output directory: " << output_dir << "\n";
    return 0;
  }

  DIR* directory = opendir(input_dir.c_str());
  if (directory == nullptr) {
    std::cerr << "Unable to open input directory: " << input_dir << "\n";
    return 0;
  }

  std::vector<std::string> filenames;
  dirent* entry = nullptr;
  while ((entry = readdir(directory)) != nullptr) {
    std::string name = entry->d_name;
    if (name == "." || name == "..") {
      continue;
    }
    if (HasSupportedExtension(name)) {
      filenames.push_back(name);
    }
  }
  closedir(directory);
  std::sort(filenames.begin(), filenames.end());

  if (max_images >= 0 &&
      static_cast<int>(filenames.size()) > max_images) {
    filenames.resize(max_images);
  }

  for (const std::string& filename : filenames) {
    const std::string input_path = input_dir + "/" + filename;
    const std::string output_path = output_dir + "/" + filename;

    double gpu_ms = 0.0;
    double io_ms = 0.0;
    if (ProcessImageFile(input_path, output_path, apply_grayscale, apply_blur,
                         &gpu_ms, &io_ms)) {
      stats->images_processed += 1;
      stats->total_gpu_ms += gpu_ms;
      stats->total_io_ms += io_ms;
      std::cout << "Processed " << filename << " gpu_ms=" << gpu_ms
                << " io_ms=" << io_ms << "\n";
    } else {
      stats->images_failed += 1;
      std::cerr << "Failed " << filename << "\n";
    }
  }

  return stats->images_processed;
}

}  // namespace batch_image
