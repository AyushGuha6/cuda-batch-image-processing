#include "cli_parser.h"
#include "image_processor.h"
#include "platform_compat.h"

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <string>

namespace {

bool EnsureParentDirectoryExists(const std::string& file_path) {
  const std::size_t slash = file_path.find_last_of("/\\");
  if (slash == std::string::npos) {
    return true;
  }
  const std::string directory = file_path.substr(0, slash);
  struct stat info {};
  if (stat(directory.c_str(), &info) == 0 && S_ISDIR(info.st_mode)) {
    return true;
  }
  return PlatformMkdir(directory.c_str(), 0755) == 0;
}

void WriteLogFile(const batch_image::CliOptions& options,
                  const batch_image::BatchStats& stats,
                  int device_id) {
  cudaDeviceProp properties {};
  cudaGetDeviceProperties(&properties, device_id);

  if (!EnsureParentDirectoryExists(options.log_file)) {
    std::cerr << "Warning: unable to create log directory for "
              << options.log_file << "\n";
  }

  std::ofstream log_stream(options.log_file.c_str());
  if (!log_stream.is_open()) {
    std::cerr << "Warning: unable to write log file " << options.log_file
              << "\n";
    return;
  }

  log_stream << "CUDA Batch Image Processor Execution Log\n";
  log_stream << "GPU: " << properties.name << "\n";
  log_stream << "Input directory: " << options.input_dir << "\n";
  log_stream << "Output directory: " << options.output_dir << "\n";
  log_stream << "Images processed: " << stats.images_processed << "\n";
  log_stream << "Images failed: " << stats.images_failed << "\n";
  log_stream << "Total GPU ms: " << stats.total_gpu_ms << "\n";
  log_stream << "Total IO ms: " << stats.total_io_ms << "\n";
  if (stats.images_processed > 0) {
    log_stream << "Average GPU ms per image: "
               << (stats.total_gpu_ms / stats.images_processed) << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  batch_image::CliOptions options;
  if (!batch_image::ParseCommandLine(argc, argv, &options)) {
    batch_image::PrintUsage(argv[0]);
    return options.help ? 0 : 1;
  }

  bool apply_grayscale = options.mode == batch_image::ProcessingMode::kGrayscale ||
                         options.mode == batch_image::ProcessingMode::kBoth;
  bool apply_blur = options.mode == batch_image::ProcessingMode::kBlur ||
                    options.mode == batch_image::ProcessingMode::kBoth;

  batch_image::ImageProcessor processor(options.threads_per_block);
  batch_image::BatchStats stats;

  cudaSetDevice(0);

  std::cout << "Starting batch GPU image processing\n";
  std::cout << "Input: " << options.input_dir << "\n";
  std::cout << "Output: " << options.output_dir << "\n";

  processor.ProcessDirectory(options.input_dir, options.output_dir,
                             apply_grayscale, apply_blur, options.max_images,
                             &stats);

  WriteLogFile(options, stats, 0);

  std::cout << "Finished. processed=" << stats.images_processed
            << " failed=" << stats.images_failed
            << " total_gpu_ms=" << stats.total_gpu_ms << "\n";
  std::cout << "Log written to " << options.log_file << "\n";

  return stats.images_failed > 0 ? 1 : 0;
}
