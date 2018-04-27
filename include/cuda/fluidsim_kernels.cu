#include <float.h>
#include <cstdint>
#include <helper_math.h>
#include <math_constants.h>
#include "cuda/math_utils.cu"

__device__ bool isValid(int x, int y, int Nx, int Ny)
{
   return ((x >= 0 && x < Nx) && (y >= 0 && y < Ny));
}

__device__ int coord(int x, int y, int Nx, int Ny)
{
	if(!isValid(x, y, Nx, Ny))
	   return -1;

	return x + Nx * y;
}

__device__ void sampleFields(
	float px, float py, int ndx, int Nx, int Ny, float Dx,
	float* density, float2* velocity,
	float* density_new, float2* velocity_new
)
{
	int i = px / Dx;
	int j = py / Dx;

	int x = coord(i, j, Nx, Ny);
	int y = coord(i, j+1, Nx, Ny);
	int z = coord(i+1, j+1, Nx, Ny);
	int w = coord(i+1, j, Nx, Ny);

	float x2x = (i + 1) * Dx - px;
	float xx1 = px - i * Dx;
	float y2y = (j + 1) * Dx - py;
	float yy1 = py - j * Dx;

	float scales[4] = {x2x * y2y, xx1 * y2y, x2x * yy1, xx1 * yy1};
	float dim = 1.0f / (Dx * Dx);

	float q11, q12, q22, q21;

	// Density
	{
		float q11 = ( x != -1 ) ? density[x] : 0.0;
		float q12 = ( y != -1 ) ? density[y] : 0.0;
		float q22 = ( z != -1 ) ? density[z] : 0.0;
		float q21 = ( w != -1 ) ? density[w] : 0.0;

		density_new[ndx] = dim * (q11 * scales[0] + q21 * scales[1] + q12 * scales[2] + q22 * scales[3]);
	}

	// Velocity
	{
		q11 = ( x != -1 ) ? velocity[x].x : 0.0;
		q12 = ( y != -1 ) ? velocity[y].x : 0.0;
		q22 = ( z != -1 ) ? velocity[z].x : 0.0;
		q21 = ( w != -1 ) ? velocity[w].x : 0.0;
		velocity_new[ndx].x = dim * (q11 * scales[0] + q21 * scales[1] + q12 * scales[2] + q22 * scales[3]);

		q11 = ( x != -1 ) ? velocity[x].y : 0.0;
		q12 = ( y != -1 ) ? velocity[y].y : 0.0;
		q22 = ( z != -1 ) ? velocity[z].y : 0.0;
		q21 = ( w != -1 ) ? velocity[w].y : 0.0;
		velocity_new[ndx].y = dim * (q11 * scales[0] + q21 * scales[1] + q12 * scales[2] + q22 * scales[3]);
	}

}

__global__ void advect_krnl(
	int Nx, int Ny, float dt, float Dx,
	float* density, 	float2* velocity,
	float* density_new, float2* velocity_new
)
{
	uint gbl_tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gbl_tid < Nx * Ny)
	{
		int tx = gbl_tid % (uint)(Nx);
		int ty = gbl_tid / (uint)(Nx);

		int ndx = coord(tx, ty, Nx, Ny);
		float px = tx * Dx - velocity[ndx].x * dt;
		float py = ty * Dx - velocity[ndx].y * dt;

		sampleFields(px, py, ndx, Nx, Ny, Dx, density, velocity, density_new, velocity_new);
	}
}

__global__ void sources_krnl(
	int Nx, int Ny, float dt, 
	float* density, float2* velocity, float2* force
)
{
	uint gbl_tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gbl_tid < Nx * Ny)
	{
		int tx = gbl_tid % (uint)(Nx);
		int ty = gbl_tid / (uint)(Nx);

		int ndx = coord(tx, ty, Nx, Ny);

		velocity[ndx] += force[ndx] * density[ndx] * dt;
	}
}

__global__ void boundaries_krnl(
	int Nx, int Ny,
	float* density, float2* velocity, float* obstruction
)
{
	uint gbl_tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gbl_tid < Nx * Ny)
	{
		int tx = gbl_tid % (uint)(Nx);
		int ty = gbl_tid / (uint)(Nx);

		int ndx = coord(tx, ty, Nx, Ny);

		if(tx == 0 || tx == Nx-1)
		{
			velocity[ndx].x = 0.0f;
			density[ndx] = 0.0f;
		}
		
		if(ty == 0 || ty == Ny-1)
		{
			velocity[ndx].y = 0.0f;
			density[ndx] = 0.0f;	
		}

		density[ndx] *= obstruction[ndx];
		velocity[ndx].x *= obstruction[ndx];
		velocity[ndx].y *= obstruction[ndx];
	}
}

__global__ void calculateDivergence_krnl(
	int Nx, int Ny, float Dx,
	float2* velocity, float* diverge
)
{
	uint gbl_tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gbl_tid < Nx * Ny)
	{
		int tx = gbl_tid % (uint)(Nx);
		int ty = gbl_tid / (uint)(Nx);

		int ndx = coord(tx, ty, Nx, Ny);

		if(tx > 0 && tx < Nx-1)
			if(ty > 0 && ty < Ny-1)
			{
				float div = 0.0f, v;
				v = velocity[coord(tx+1, ty, Nx, Ny)].x;
				div += v;
				v = velocity[coord(tx-1, ty, Nx, Ny)].x;
				div -= v;

				v = velocity[coord(tx, ty+1, Nx, Ny)].y;
				div += v;
				v = velocity[coord(tx, ty-1, Nx, Ny)].y;
				div -= v;

				div /= 2.0 * Dx;
				diverge[ndx] += div * Dx * Dx;
			}
	}
}

__global__ void calculatePressure_krnl(
	int Nx, int Ny, float Dx,
	float* diverge, float* pressure, float* pressure_new
)
{
	uint gbl_tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gbl_tid < Nx * Ny)
	{
		int tx = gbl_tid % (uint)(Nx);
		int ty = gbl_tid / (uint)(Nx);

		int ndx = coord(tx, ty, Nx, Ny);

		if(tx > 0 && tx < Nx-1)
			if(ty > 0 && ty < Ny-1)
			{
				float p = 
					  pressure[coord(tx+1, ty, Nx, Ny)]
					+ pressure[coord(tx-1, ty, Nx, Ny)]
					+ pressure[coord(tx, ty+1, Nx, Ny)]
					+ pressure[coord(tx, ty-1, Nx, Ny)];
				pressure_new[ndx] = (p - diverge[ndx]) / 4.0;
			}

	}
}

__global__ void calculateVelocityFromPressure_krnl(
	int Nx, int Ny, float Dx,
	float2* velocity, float* pressure
)
{
	uint gbl_tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gbl_tid < Nx * Ny)
	{
		int tx = gbl_tid % (uint)(Nx);
		int ty = gbl_tid / (uint)(Nx);

		int ndx = coord(tx, ty, Nx, Ny);

		if(tx > 0 && tx < Nx-1)
			if(ty > 0 && ty < Ny-1)
			{
				velocity[ndx].x -= (pressure[coord(tx+1, ty, Nx, Ny)] - pressure[coord(tx-1, ty, Nx, Ny)]) / (2.0 * Dx);
				velocity[ndx].y -= (pressure[coord(tx, ty+1, Nx, Ny)] - pressure[coord(tx, ty-1, Nx, Ny)]) / (2.0 * Dx);
			}

	}

}

__global__ void initAll_krnl(
	int Nx, int Ny,
	float* density, 	float2* velocity,
	float* density_new, float2* velocity_new, 
	float* obstruction, float2* force,
	float2 gravity
)
{
	uint gbl_tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gbl_tid < Nx * Ny)
	{
		density[gbl_tid] = 0.0f;
		density_new[gbl_tid] = 0.0f;

		velocity[gbl_tid].x = 0.0f;
		velocity[gbl_tid].y = 0.0f;
		velocity_new[gbl_tid].x = 0.0f;
		velocity_new[gbl_tid].y = 0.0f;

		obstruction[gbl_tid] = 1.0f;

		force[gbl_tid].x = gravity.x;
		force[gbl_tid].y = gravity.y;
	}
}

__global__ void initPressureDivergence_krnl(
	int Nx, int Ny,
	float* diverge, float* pressure, float* pressure_new
)
{
	uint gbl_tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gbl_tid < Nx * Ny)
	{
		int tx = gbl_tid % (uint)(Nx);
		int ty = gbl_tid / (uint)(Nx);

		int ndx = coord(tx, ty, Nx, Ny);

		diverge[ndx] = 0.0f;
		pressure[ndx] = 0.0f;
		pressure_new[ndx] = 0.0f;
	}
}

__global__ void setColorToDensity_krnl(
	int Nx, int Ny,
	float* density, float4* color
)
{
	uint gbl_tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gbl_tid < Nx * Ny)
	{
		int tx = gbl_tid % (uint)(Nx);
		int ty = gbl_tid / (uint)(Nx);

		int ndx = coord(tx, ty, Nx, Ny);

		float d = clamp(density[ndx], 0.0, 1.0);
		color[ndx].x = 1.0; // R
		color[ndx].y = 1.0; // G
		color[ndx].z = 1.0; // B
		color[ndx].w = d;   // A
	}
}

__global__ void setColorToVelocity_krnl(
	int Nx, int Ny, float dt,
	float2* velocity, float4* color
)
{
	uint gbl_tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gbl_tid < Nx * Ny)
	{
		int tx = gbl_tid % (uint)(Nx);
		int ty = gbl_tid / (uint)(Nx);

		int ndx = coord(tx, ty, Nx, Ny);

		float d = velocity[ndx].x * velocity[ndx].x + velocity[ndx].y * velocity[ndx].y;
		d = pow(d, 0.5f) + 1.0e-8f;

		color[ndx].x = ((velocity[ndx].x / d) + 1.0f) / 2.0f;
		color[ndx].y = ((velocity[ndx].y / d) + 1.0f) / 2.0f;
		color[ndx].z = d * dt;
		color[ndx].w = 1.0f;
	}
}

__global__ void setColorToColormappedDensity_krnl(
	int Nx, int Ny, float4 cm_a, float4 cm_b, float4 cm_c, float4 cm_d,
	float* density, float4* color
)
{
	uint gbl_tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gbl_tid < Nx * Ny)
	{
		int tx = gbl_tid % (uint)(Nx);
		int ty = gbl_tid / (uint)(Nx);

		int ndx = coord(tx, ty, Nx, Ny);

		float t = density[ndx];//clamp(density[ndx], 0.0, 1.0);
		color[ndx].x = cm_a.x + cm_b.x * cos(2.0 * CUDART_PI_F * (cm_c.x * t + cm_d.x) );
		color[ndx].y = cm_a.y + cm_b.y * cos(2.0 * CUDART_PI_F * (cm_c.y * t + cm_d.y) );
		color[ndx].z = cm_a.z + cm_b.z * cos(2.0 * CUDART_PI_F * (cm_c.z * t + cm_d.z) );
		color[ndx].w = clamp(density[ndx], 0.0, 1.0);
	}
}

__global__ void setColorToMappedTemperature_krnl(
	int Nx, int Ny, float kelvin_per_d,
	float* density, float4* color
)
{
	uint gbl_tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gbl_tid < Nx * Ny)
	{
		int tx = gbl_tid % (uint)(Nx);
		int ty = gbl_tid / (uint)(Nx);

		int ndx = coord(tx, ty, Nx, Ny);

		float T = fmaxf(density[ndx], 0.0) * kelvin_per_d;
		float sen(1000.0), mil(exp10(6.0)), bil(exp10(9.0));

		float a = bil / (T * T * T);
		float b = mil / (T * T);
		float c = sen / T;
		float d = 0.0;

		if( T <= 4000 )
		{
			a *= -0.2661239;
			b *= -0.2343580;
			c *=  0.8776956;
			d  =  0.179910;
		}
		else
		{
			a *= -3.0258469;
			b *=  2.1070379;
			c *=  0.2226347;
			d  =  0.24039;
		}
		float x_c = a + b + c + d;

		d = 0;
		c = x_c;
		b = c * x_c;
		a = b * x_c;

		if( T <= 2222 )
		{
			a *= -1.10638140;
			b *= -1.34811020;
			c *=  2.18555832;
			d  = -0.20219683;	
		}
		else if( T <= 4000 )
		{
			a *= -0.95494760;
			b *= -1.37418593;
			c *=  2.09137015;
			d  = -0.16748867;
		}
		else
		{
			a *=  3.08175800;
			b *= -5.87338670;
			c *=  3.75112997;
			d  = -0.37001483;	
		}
		float y_c = a + b + c + d;

		float z_c = 1.0 - x_c - y_c;
		x_c = x_c / y_c;
		z_c = z_c / y_c;
		y_c = 1.0;

		color[ndx].x =  0.418470   * x_c - 0.15866   * y_c - 0.082835 * z_c;
		color[ndx].y = -0.091169   * x_c + 0.25243   * y_c + 0.015708 * z_c;
		color[ndx].z =  0.00092090 * x_c - 0.0025498 * y_c + 0.178600 * z_c;
		color[ndx].w = clamp(density[ndx], 0.0, 1.0);

		color[ndx].x *= 3.0;
		color[ndx].y *= 3.0;
		color[ndx].z *= 3.0;

		// Gamma
		// color[ndx].x = pow(color[ndx].x, 0.45);
		// color[ndx].y = pow(color[ndx].y, 0.45);
		// color[ndx].z = pow(color[ndx].z, 0.45);

		color[ndx].x = clamp(color[ndx].x, 0.0, 1.0);
		color[ndx].y = clamp(color[ndx].y, 0.0, 1.0);
		color[ndx].z = clamp(color[ndx].z, 0.0, 1.0);
	}
}

__global__ void packColorToByteArray_krnl(
	int Nx, int Ny,
	float4* color, unsigned char* byte_tex
)
{
	uint gbl_tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gbl_tid < Nx * Ny)
	{
		int tx = gbl_tid % (uint)(Nx);
		int ty = gbl_tid / (uint)(Nx);

		int ndx = coord(tx, ty, Nx, Ny);

		unsigned char r = (unsigned char)(255 * color[ndx].x);
		unsigned char g = (unsigned char)(255 * color[ndx].y);
		unsigned char b = (unsigned char)(255 * color[ndx].z);
		unsigned char a = (unsigned char)(255 * color[ndx].w);
		byte_tex[4*ndx]   = a;
		byte_tex[4*ndx+1] = b;
		byte_tex[4*ndx+2] = g;
		byte_tex[4*ndx+3] = r;
	}
}
