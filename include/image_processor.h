#ifndef CUDA_BATCH_IMAGE_PROCESSOR_INCLUDE_IMAGE_PROCESSOR_H_
#define CUDA_BATCH_IMAGE_PROCESSOR_INCLUDE_IMAGE_PROCESSOR_H_

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace batch_image {

struct ImageBuffer {
  std::vector<unsigned char> rgb;
  std::vector<unsigned char> gray;
  int width = 0;
  int height = 0;
  int channels = 0;
};

struct BatchStats {
  int images_processed = 0;
  int images_failed = 0;
  double total_gpu_ms = 0.0;
  double total_io_ms = 0.0;
};

class ImageProcessor {
 public:
  explicit ImageProcessor(int threads_per_block);

  bool ProcessImageFile(const std::string& input_path,
                        const std::string& output_path,
                        bool apply_grayscale,
                        bool apply_blur,
                        double* gpu_ms,
                        double* io_ms);

  int ProcessDirectory(const std::string& input_dir,
                       const std::string& output_dir,
                       bool apply_grayscale,
                       bool apply_blur,
                       int max_images,
                       BatchStats* stats);

 private:
  bool LoadImage(const std::string& path, ImageBuffer* image);
  bool SaveImage(const std::string& path, const ImageBuffer& image,
                 bool save_grayscale);

  void RunGrayscaleKernel(unsigned char* device_rgb,
                          unsigned char* device_gray,
                          int width,
                          int height,
                          cudaStream_t stream);

  void RunBlurKernel(unsigned char* device_in,
                     unsigned char* device_out,
                     int width,
                     int height,
                     int channels,
                     cudaStream_t stream);

  int threads_per_block_;
};

}  // namespace batch_image

#endif  // CUDA_BATCH_IMAGE_PROCESSOR_INCLUDE_IMAGE_PROCESSOR_H_
