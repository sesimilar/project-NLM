//#include <stdio.h>
//#include <stdlib.h>
//#ifdef __APPLE__
//#include <OpenCL/cl.h>
//#else
//#include <CL/cl.h>
//#endif
//#define VECTOR_SIZE 1024
//
//const char* _kernel =
//"__kernel \n"
//"void saxpy_kernel(float alpha, \n"
//" __global float *A, \n"
//" __global float *B, \n"
//" __global float *C) \n"
//"{ \n"
//" //Get the index of the work-item \n"
//" int index = get_global_id(0); \n"
//" C[index] = alpha* A[index] + B[index]; \n"
//"} \n";
//
//int main(int argc, char** argv){
//	int i;
//	float alpha = 2.0;
//	float* A = (float*)malloc(sizeof(float)*VECTOR_SIZE);
//	float* B = (float*)malloc(sizeof(float)*VECTOR_SIZE);
//	float* C = (float*)malloc(sizeof(float)*VECTOR_SIZE);
//	for (i = 0; i < VECTOR_SIZE; i++){
//		A[i] = i;
//		B[i] = VECTOR_SIZE - i;
//		C[i] = 0;
//	}
//
//	cl_platform_id* platforms = NULL;
//	cl_uint num_platforms = 0;
//	cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
//	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*num_platforms);
//
//	clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);
//
//	cl_device_id* device_list = NULL;
//	cl_uint num_device = 0;
//
//	clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_device);
//	device_list = (cl_device_id*)malloc(sizeof(cl_device_id)*num_device);
//	clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_device, device_list, NULL);
//
//	cl_context context = NULL;
//	context = clCreateContext(NULL, num_device, device_list, NULL, NULL, &clStatus);
//
//	cl_command_queue command_queue = clCreateCommandQueue(context, device_list[1], 0, &clStatus);
//
//	cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);
//	cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);
//	cl_mem C_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);
//
//	clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, VECTOR_SIZE*sizeof(float), A, 0, NULL, NULL);
//	clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, VECTOR_SIZE*sizeof(float), B, 0, NULL, NULL);
//
//	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&_kernel, NULL, &clStatus);
//
//	clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
//
//	cl_kernel kernel = clCreateKernel(program, "_kernel", &clStatus);
//
//	clStatus = clSetKernelArg(kernel, 0, sizeof(float), (void*)&alpha);
//	clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&A_clmem);
//	clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&B_clmem);
//	clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&C_clmem);
//
//	size_t global_size = VECTOR_SIZE;
//	size_t local_size = 64;
//	clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
//	clStatus = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0, VECTOR_SIZE*sizeof(float), C, 0, NULL, NULL);
//	clStatus = clFlush(command_queue);
//	clStatus = clFinish(command_queue);
//
//	for (i = 0; i < VECTOR_SIZE; i++){
//		printf("%f * %f + %f = %f\n", alpha, A[i], B[i], C[i]);
//	}
//
//	clStatus = clReleaseKernel(kernel);
//	clStatus = clReleaseProgram(program);
//	clStatus = clReleaseMemObject(A_clmem);
//	clStatus = clReleaseMemObject(B_clmem);
//	clStatus = clReleaseMemObject(C_clmem);
//	clStatus = clReleaseCommandQueue(command_queue);
//	clStatus = clReleaseContext(context);
//	free(A);
//	free(B);
//	free(C);
//	free(platforms);
//	free(device_list);
//	return 0;
//}