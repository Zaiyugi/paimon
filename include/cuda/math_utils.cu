#ifndef __MATH_UTILS_CU__
#define __MATH_UTILS_CU__

#include <float.h>
#include <helper_math.h>
#include <math_constants.h>

#include <curand.h>
#include <curand_kernel.h>

#define CUDART_PI_F 3.141592654f

#define step(a, b) ((b<a) ? 0.0 : 1.0)

// template <class T>
// __device__ void cuswap(T& a, T &b) { T c(a); a = b; b = c; }

__device__ void cuswap(float& a, float& b)
{
	float c = a;
	a = b;
	b = c;
}

__device__ inline float4 max(float4 v, float a)
{
	return make_float4(
		(v.x < a) ? a : v.x,
		(v.y < a) ? a : v.y,
		(v.z < a) ? a : v.z,
		(v.w < a) ? a : v.w
	);
}

__device__ inline float fract(float x)
{
	return x - truncf(x);
}

__device__ inline float mod(float x, float y)
{
	return (x - y * floorf(x/y));
}

__device__ inline float4 floor(float4 a){
   return make_float4( floorf(a.x), floorf(a.y), floorf(a.z), floorf(a.w) );
}

__device__ inline float dot3(float4 a, float4 b)
{
	return (a.x*b.x + a.y*b.y + a.z*b.z);
}

__device__ inline float4 cross(float4 a, float4 b)
{
    return make_float4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0.0);
}

__device__ float4 unitize(float4 v)
{
	float len = sqrtf(v.x*v.x+v.y*v.y+v.z*v.z);
	if(len < 0.000005)
		return make_float4(0.0, 1.0, 0.0, 0.0);
	return make_float4(v.x/len, v.y/len, v.z/len, 0.0);
}

__device__ inline float4 abs(float4 &v)
{
	float4 r;
	r.x = abs(v.x); r.y = abs(v.y); r.z = abs(v.z); r.w = abs(v.w);
	return r;
}

// Convert floating point to 8-bit ints
__device__ int rgbToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b)<<16) | (int(g)<<8) | int(r);
}

__device__ float2 rand2n(float4& seed)
{
    seed += make_float4(-1.0, 1.0, 0.0, 0.0) * (seed.z + 1.0);
	// implementation based on: lumina.sourceforge.net/Tutorials/Noise.html
    return make_float2(
    	fract(sin( seed.x*12.9898+seed.y*78.233 ) * 43758.5453),
		fract(cos( seed.x*4.9898+seed.y*7.23 ) * 23421.631)
	);
};

__device__ float random (float vx, float vy) {
	float x = vx * 12.9898 + vy * 78.233;
	return fract(sin(x) * 43758.5453);
}

__device__ float smin( float a, float b, float k )
{
	float h = clamp(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
	return lerp(b, a, h) - k * h * (1.0 - h);
}

__device__ int iseq( float a, float b, float c )
{
	return ( fabs(a - b) < c );
}

// Kernels

__global__ void initRandomStates_krnl(unsigned int seed, curandState_t *states)
{
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(seed, tid, 0, &states[tid]);
}

#endif
