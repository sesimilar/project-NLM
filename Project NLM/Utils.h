#include <fstream>
#include <opencv.hpp>
#include <iostream>
#include <sstream>

#ifdef _APPLE_
#include <OpenCL/cl.h>
#else
#include <CL\cl.h>
#include <CL\cl.hpp>
#endif


void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem imageObjects[2], cl_sampler sampler);

size_t RoundUp(int groupSize, int globalSize);

