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

/**
*@Author JoelChen
*create cl_context by device
*@return cl_context
*/
cl_context CreateContext();

cl_command_queue CreateCommandQueue(cl_context context , cl_device_id* device);

cl_program CreateProgram(cl_context context,cl_device_id device , std::string fileName);

cl_bool ImageSupport(cl_device_id device);



