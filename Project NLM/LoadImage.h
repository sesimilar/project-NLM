#include <fstream>
#include <iostream>
#include <opencv.hpp>

#ifdef __APPLE__  
#include <OpenCL/cl.h>  
#else  
#include <CL/cl.h>  
#endif  

#pragma warning( disable : 4996 )

cl_mem LoadImage(cl_context context, std::string fileName, int &width, int &height);