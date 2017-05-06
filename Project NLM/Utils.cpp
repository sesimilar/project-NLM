#include "Utils.h"

void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem imageObjects[2], cl_sampler sampler)
{
	for (int i = 0; i < 2; i++)
	{
		if (imageObjects[i] != 0)
			clReleaseMemObject(imageObjects[i]);
	}
	if (commandQueue != 0)
		clReleaseCommandQueue(commandQueue);

	if (kernel != 0)
		clReleaseKernel(kernel);

	if (program != 0)
		clReleaseProgram(program);

	if (sampler != 0)
		clReleaseSampler(sampler);

	if (context != 0)
		clReleaseContext(context);

}

size_t RoundUp(int groupSize, int globalSize)
{
	int r = globalSize % groupSize;
	if (r == 0)
	{
		return globalSize;
	}
	else
	{
		return globalSize + groupSize - r;
	}
}