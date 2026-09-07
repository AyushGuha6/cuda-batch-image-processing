#ifndef CUDA_BATCH_IMAGE_PROCESSOR_INCLUDE_CLI_PARSER_H_
#define CUDA_BATCH_IMAGE_PROCESSOR_INCLUDE_CLI_PARSER_H_

#include <string>

namespace batch_image {

enum class ProcessingMode {
  kGrayscale = 0,
  kBlur = 1,
  kBoth = 2,
};

struct CliOptions {
  std::string input_dir = "data/input";
  std::string output_dir = "data/output";
  std::string log_file = "results/execution_log.txt";
  ProcessingMode mode = ProcessingMode::kBoth;
  int threads_per_block = 256;
  int max_images = -1;
  bool help = false;
};

bool ParseCommandLine(int argc, char** argv, CliOptions* options);

void PrintUsage(const char* program_name);

}  // namespace batch_image

#endif  // CUDA_BATCH_IMAGE_PROCESSOR_INCLUDE_CLI_PARSER_H_
