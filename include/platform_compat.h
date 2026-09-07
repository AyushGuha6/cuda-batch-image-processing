#pragma once

#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>

#ifndef S_ISDIR
#define S_ISDIR(mode) (((mode) & _S_IFDIR) != 0)
#endif

inline int PlatformMkdir(const char* path, int mode) {
  (void)mode;
  return _mkdir(path);
}
#else
inline int PlatformMkdir(const char* path, int mode) {
  return mkdir(path, mode);
}
#endif
