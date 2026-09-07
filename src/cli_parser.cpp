#include "cli_parser.h"

#include <iostream>
#include <string>

namespace batch_image {

namespace {

ProcessingMode ParseMode(const std::string& mode_text) {
  if (mode_text == "grayscale") {
    return ProcessingMode::kGrayscale;
  }
  if (mode_text == "blur") {
    return ProcessingMode::kBlur;
  }
  return ProcessingMode::kBoth;
}

}  // namespace

void PrintUsage(const char* program_name) {
  std::cout
      << "Usage: " << program_name << " [options]\n\n"
      << "Options:\n"
      << "  --input <dir>       Input image directory (default: data/input)\n"
      << "  --output <dir>      Output image directory (default: data/output)\n"
      << "  --log <file>        Execution log file (default: results/execution_log.txt)\n"
      << "  --mode <name>       grayscale | blur | both (default: both)\n"
      << "  --threads <n>       CUDA threads per block (default: 256)\n"
      << "  --max-images <n>    Process at most n images (-1 = all, default: -1)\n"
      << "  --help              Show this message\n";
}

bool ParseCommandLine(int argc, char** argv, CliOptions* options) {
  if (options == nullptr) {
    return false;
  }

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      options->help = true;
      return false;
    }
    if (arg == "--input" && i + 1 < argc) {
      options->input_dir = argv[++i];
      continue;
    }
    if (arg == "--output" && i + 1 < argc) {
      options->output_dir = argv[++i];
      continue;
    }
    if (arg == "--log" && i + 1 < argc) {
      options->log_file = argv[++i];
      continue;
    }
    if (arg == "--mode" && i + 1 < argc) {
      options->mode = ParseMode(argv[++i]);
      continue;
    }
    if (arg == "--threads" && i + 1 < argc) {
      options->threads_per_block = std::stoi(argv[++i]);
      continue;
    }
    if (arg == "--max-images" && i + 1 < argc) {
      options->max_images = std::stoi(argv[++i]);
      continue;
    }

    std::cerr << "Unknown or incomplete argument: " << arg << "\n";
    return false;
  }

  if (options->threads_per_block <= 0) {
    std::cerr << "threads per block must be positive\n";
    return false;
  }

  return true;
}

}  // namespace batch_image
