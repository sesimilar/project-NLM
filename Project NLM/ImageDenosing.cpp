#include "getDevice.h"
#include "LoadImage.h"
#include "Utils.h"
#include <time.h>
using namespace std;

static float nlmNoise = 1.45f;//图像噪声
double getPSNR(const cv::Mat& I1, const cv::Mat& I2);

int main(int argc, char** argv)
{
	float nosie = 1.0f / (nlmNoise * nlmNoise);
	float lerpC = 0.2f;
	cl_context context = 0;
	cl_command_queue commandQueue = 0;
	cl_program program = 0;
	cl_device_id device = 0;
	cl_kernel kernel = 0;
	cl_mem imageObjects[2] = { 0, 0 }; //一个原图像 一个目标图像
	cl_mem cl_width = NULL;
	cl_mem cl_height = NULL;
	cl_mem cl_lerpC = NULL;
	cl_mem cl_noise = NULL;
	cl_sampler sampler = 0;
	cl_int errNum;
	string cl_kernel_file = "ImageDenosing_NLM_kernel.cl";//OpenCL 文件路径
	// 获取设备 
	context = CreateContext();
	if (context == NULL)
	{
		cerr << "Failed to create OpenCL context." << endl;
		cin.get();
	}

	//创建队列
	commandQueue = CreateCommandQueue(context, &device);
	if (commandQueue == NULL)
	{
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		cin.get();
		return 1;
	}
	// 确保计算设备能够支持图片  
	if (ImageSupport(device) != CL_TRUE)
	{
		cerr << "OpenCL device does not support images." << endl;
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		cin.get();
		return 1;
	}

	// 将图片载入OpenCL设备
	int width, height; //在LoadImage函数改变了其值
	string src0 = "images/portrait_noise.bmp";
	imageObjects[0] = LoadImage(context, src0, width, height);
	if (imageObjects[0] == 0)
	{
		cerr << "Error loading: " << string(src0) << endl;
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		cin.get();
		return 1;
	}

	// 创建目标图像 （处理得到的）
	cl_image_format clImageFormat;
	clImageFormat.image_channel_order = CL_RGBA;
	clImageFormat.image_channel_data_type = CL_UNORM_INT8;
	imageObjects[1] = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &clImageFormat, width, height, 0, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
		cerr << "Error creating CL output image object." << endl;
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		return 1;
	}

	// Create sampler for sampling image object  
	sampler = clCreateSampler(context,
		CL_FALSE, // Non-normalized coordinates  
		CL_ADDRESS_CLAMP_TO_EDGE,
		CL_FILTER_NEAREST,
		&errNum);

	if (errNum != CL_SUCCESS)
	{
		cerr << "Error creating CL sampler object." << endl;
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		return 1;
	}

	// 创建函数项
	program = CreateProgram(context, device, cl_kernel_file);
	if (program == NULL)
	{
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		cin.get();
		return 1;
	}

	//创建一个OpenCL中的函数
	kernel = clCreateKernel(program, "NLMFiltering", NULL);
	if (kernel == NULL)
	{
		cerr << "Failed to create kernel" << endl;
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		cin.get();
		return 1;
	}
	cl_width  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_mem), NULL, &errNum);
	cl_height = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_mem), NULL, &errNum);
	cl_noise  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_mem), NULL, &errNum);
	cl_lerpC  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_mem), NULL, &errNum);
	
	errNum = clEnqueueWriteBuffer(commandQueue, cl_width, CL_TRUE, 0, sizeof(int), (void*)&width, 0, NULL, NULL);
	errNum = clEnqueueWriteBuffer(commandQueue, cl_height, CL_TRUE, 0, sizeof(int), (void*)&height, 0, NULL, NULL);
	errNum = clEnqueueWriteBuffer(commandQueue, cl_noise, CL_TRUE, 0, sizeof(float), (void*)&nosie, 0, NULL, NULL);
	errNum = clEnqueueWriteBuffer(commandQueue, cl_lerpC, CL_TRUE, 0, sizeof(float), (void*)&lerpC, 0, NULL, NULL);

	// 传入参数
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageObjects[0]);
 	errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), &imageObjects[1]);
	errNum = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_width);
	errNum = clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_height);
	errNum = clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_noise);
	errNum = clSetKernelArg(kernel, 5, sizeof(cl_mem), &cl_lerpC);
	if (errNum != CL_SUCCESS)
	{
		cerr << "Error setting kernel arguments." << endl;
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		return 1;
	}

	//貌似是空间大小？
	size_t localWorkSize[2] = { 8, 8 };
	size_t globalWorkSize[2] = { 
		// RoundUp(localWorkSize[0], width),
		// RoundUp(localWorkSize[1], height)
		width,
		height
		 };
	//开始运算
	clock_t t1, t2;
	t1 = clock();
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);


	if (errNum != CL_SUCCESS)
	{
		cerr << "Error queuing kernel for execution." << endl;
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		cin.get();
		return 1;
	}
	
	// Read the output buffer back to the Host  
	char *buffer = new char[width * height * 4];
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { width, height, 1 };
	errNum = clEnqueueReadImage(commandQueue, imageObjects[1], CL_TRUE,
		origin, region, 0, 0, buffer,
		0, NULL, NULL);
	t2 = clock();
	cout << "OpenCL - NLMDenosing:      " << t2 - t1 << "ms" << endl;
	if (errNum != CL_SUCCESS)
	{
		cerr << "Error reading result buffer." << endl;
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		cin.get();
		return 1;
	}

	cv::Mat imageColor = cv::imread(src0);
	// t1 = clock();
	// cv::cvtColor(imageColor, imageColor, CV_BGR2GRAY);
	// t2 = clock();
	// cout << "OpenCV - BGR2GRAY:      " << t2 - t1 << "ms" << endl;

	// cv::imshow("OpenCV-BGR2GRAY", imageColor);
	cv::Mat imageColor1 = cv::imread(src0);
	cv::Mat imageColor2;
	imageColor2.create(imageColor.rows, imageColor.cols, imageColor1.type());
	int w = 0;
	for (int v = imageColor2.rows - 1; v >= 0; v--)
	{
		for (int u = 0; u <imageColor2.cols; u++)
		{
			imageColor2.at<cv::Vec3b>(v, u)[0] = buffer[w++];
			imageColor2.at<cv::Vec3b>(v, u)[1] = buffer[w++];
			imageColor2.at<cv::Vec3b>(v, u)[2] = buffer[w++];
			w++;
		}
	}
	cv::imshow("OpenCL-NLMDenosing", imageColor2);
	cout << "PSNR:      " << getPSNR(cv::imread(src0),imageColor2) << endl;
	cv::waitKey(0);
	delete[] buffer;
	Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
	return 0;
}
double getPSNR(const cv::Mat& I1, const cv::Mat& I2)
{
	cv::Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2

	cv::Scalar s = sum(s1);         // sum elements per channel

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // for small values return zero
		return 0;
	else
	{
		double  mse = sse / (double)(I1.channels() * I1.total());
		double psnr = 10.0*log10((255 * 255) / mse);
		return psnr;
	}
}