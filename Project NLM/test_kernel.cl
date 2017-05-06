const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
kernel void bgr2gray(read_only image2d_t src, write_only image2d_t dst)
{
	int x = (int)get_global_id(0);
	int y = (int)get_global_id(1);
	if (x >= get_image_width(src) || y >= get_image_height(src))
		return;
	float3 pixel_ = read_imagef(src, sampler, (int2)(x, y)).xyz;
		float dst_ = 0.11*pixel_.x + 0.59*pixel_.y + 0.30*pixel_.z;
	write_imagef(dst, (int2)(x, y), (float4)(dst_, dst_, dst_, 1.0f));
}