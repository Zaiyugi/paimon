/*
Paimon: An Eulerian fluid solver with CUDA acceleration
-------------------------------------------------------
Copyright (C) 2018 Zachary E. Shore

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or 
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
-------------------------------------------------------

float_variants.h
-------------
Variants of the float datatype: float4 and float2

Provides a definition for these types when Paimon is not compiled 
with nvcc. Allows for including Paimon headers in non-CUDA projects
without required nvcc.

*/


#ifndef __FLOAT_VARS_H__
#define __FLOAT_VARS_H__

namespace goetia
{

# ifndef __CUDACC__

// If not compiling with nvcc, define some variants of float

class float4
{
public:
   float4() : x(0.0), y(0.0), z(0.0), w(0.0) {}
   
   float4(float r, float g, float b, float a) : x(r), y(g), z(b), w(a) {}

   float4(float r, float g, float b) : x(r), y(g), z(b), w(0.0) {}

   float4(float4& flt) : x(flt.x), y(flt.y), z(flt.z), w(flt.w) {}

   float4& operator=(float4 flt)
   {
      x = flt.x; y = flt.y; z = flt.z; w = flt.w;
      return *this;
   }

   float x;
   float y;
   float z;
   float w;
};

class float2
{
public:
   float2() : x(0.0), y(0.0) {}
   
   float2(float r, float g) : x(r), y(g) {}

   float2(float r) : x(r), y(0.0) {}

   float2(float2& flt) : x(flt.x), y(flt.y) {}

   float2& operator=(float2 flt)
   {
      x = flt.x; y = flt.y;
      return *this;
   }

   float x;
   float y;
};

# endif

}

#endif
