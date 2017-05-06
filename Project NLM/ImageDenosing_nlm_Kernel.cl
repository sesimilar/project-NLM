#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8
#define NLM_WINDOW_RADIUS   3
#define NLM_WINDOW_AREA     ( (2 * NLM_WINDOW_RADIUS + 1) * (2 * NLM_WINDOW_RADIUS + 1) )
#define NLM_WEIGHT_THRESHOLD    0.10f
#define NLM_LERP_THRESHOLD      0.10f
#define INV_NLM_WINDOW_AREA ( 1.0f / (float)NLM_WINDOW_AREA )

inline float vecLen(float3 a, float3 b)
{
	return (
		(b.x - a.x) * (b.x - a.x) +
		(b.y - a.y) * (b.y - a.y) +
		(b.z - a.z) * (b.z - a.z)
		);
}

inline float lerpf(float a, float b, float c)
{
	return a + (b - a) * c;
}

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void NLMFiltering(read_only image2d_t src , 
	                     write_only image2d_t dst,
	                     __global int* width,
	                     __global int* height,
	                     __global float* lerpC,
	                     __global float* noise)
{
	__local float fweights[BLOCKDIM_X * BLOCKDIM_Y];

//	const int ix = get_local_size(0) * get_group_id(0) + get_local_id(0);
//	const int iy = get_local_size(1) * get_group_id(1) + get_local_id(1);
	const int ix = get_global_id(0);
	const int iy = get_global_id(1);
	const float x = (float)ix + 0.5f;
	const float y = (float)iy + 0.5f;

	const int cx = get_local_size(0) * get_global_id(0) + NLM_WINDOW_RADIUS ;
	const int cy = get_local_size(1) * get_global_id(1) + NLM_WINDOW_RADIUS ;

	if(ix < *width && iy < *height){
		float weight = 0;

		for(float n = -NLM_WINDOW_RADIUS ; n <= NLM_WINDOW_RADIUS ; n++){
			for(float m = -NLM_WINDOW_RADIUS ; m <= -NLM_WINDOW_RADIUS ; m++){
				weight += vecLen(
					   (float3)read_imagef(src , sampler , (int2)(cx + m , cy + n)).xyz,
                       (float3)read_imagef(src , sampler , (int2)(ix + m , iy + n)).xyz

					);
			}
		}
	    float dist = (get_local_id(0) - NLM_WINDOW_RADIUS) * (get_local_id(0) - NLM_WINDOW_RADIUS) +
	                 (get_local_id(1) - NLM_WINDOW_RADIUS) * (get_local_id(1) - NLM_WINDOW_RADIUS);

	    weight = native_exp(-(weight * (*noise) + dist * INV_NLM_WINDOW_AREA));

	    fweights[get_local_id(1) * BLOCKDIM_X + get_local_id(0)] = weight;

	    barrier(CLK_LOCAL_MEM_FENCE);

	    float fCount = 0;

	    float sumWeights = 0;

	    float3 clr = {0,0,0};

	    int idx = 0;

		for (float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS + 1; i++){
			for (float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS + 1; j++){
	    		float weightij = fweights[idx++];

	    		float3 clrij = read_imagef(src, sampler, (int2)(ix + j , iy + i)).xyz;
	    		clr.x       += clrij.x * weightij;
	    		clr.y       += clrij.y * weightij;
	    		clr.z       += clrij.z * weightij;

				sumWeights += weightij;

				fCount += (weightij > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
	    	}
	    }

		sumWeights = 1.0f / sumWeights;
		clr.x *= sumWeights;
		clr.y *= sumWeights;
		clr.z *= sumWeights;

		float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? (*lerpC) : 1.0f - *lerpC;

        float3 clr00 = read_imagef(src, sampler, (int2)(ix , iy)).xyz;
        clr.x = lerpf(clr.x, clr00.x, lerpQ);
        clr.y = lerpf(clr.y, clr00.y, lerpQ);
        clr.z = lerpf(clr.z, clr00.z, lerpQ);
        write_imagef(dst, (int2)(ix, iy), (float4)(clr.x , clr.y, clr.z, 1.0f));
	}
}

