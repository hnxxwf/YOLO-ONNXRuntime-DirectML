#ifndef YOLOORTDML_EXPORT_H
#define YOLOORTDML_EXPORT_H

#if defined(_WIN32)
#  if defined(YOLOORTDML_BUILD_SHARED)
#    define YOLOORTDML_API __declspec(dllexport)
#  elif defined(YOLOORTDML_USE_SHARED)
#    define YOLOORTDML_API __declspec(dllimport)
#  else
#    define YOLOORTDML_API
#  endif
#else
#  if defined(YOLOORTDML_BUILD_SHARED)
#    define YOLOORTDML_API __attribute__((visibility("default")))
#  else
#    define YOLOORTDML_API
#  endif
#endif

#endif // YOLOORTDML_EXPORT_H
