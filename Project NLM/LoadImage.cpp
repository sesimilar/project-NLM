#include "LoadImage.h"

cl_mem LoadImage(cl_context context, std::string fileName, int &width, int &height)
{
	cv::Mat image1 = cv::imread(fileName);
	width = image1.cols;
	height = image1.rows;
	char *buffer = new char[width * height * 4];

	//数据传入方式：一个像素一个像素，按照B G R顺序，中间空一格 就像： 
	// 12 237 34  221 88 99  22 33 99
	int w = 0;
	for (int v = height - 1; v >= 0; v--)
	{
		for (int u = 0; u <width; u++)
		{
			buffer[w++] = image1.at<cv::Vec3b>(v, u)[0];
			buffer[w++] = image1.at<cv::Vec3b>(v, u)[1];
			buffer[w++] = image1.at<cv::Vec3b>(v, u)[2];
			w++;
		}
	}


	// Create OpenCL image  
	cl_image_format clImageFormat;
	clImageFormat.image_channel_order = CL_RGBA;
	clImageFormat.image_channel_data_type = CL_UNORM_INT8;

	cl_int errNum;
	cl_mem clImage;
	clImage = clCreateImage2D(context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		&clImageFormat,
		width,
		height,
		0,
		buffer,
		&errNum);

	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error creating CL image object" << std::endl;
		return 0;
	}

	return clImage;
}