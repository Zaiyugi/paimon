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

MemUtils.cuh
-------------
Various memory utility functions for working with host/device arrays

*/

#ifndef __MEMUTILS_CUH__
#define __MEMUTILS_CUH__

#include <iostream>
#include <cstdlib>
#include <cuda.h>

namespace goetia
{

namespace util
{

template <typename T>
T* hostMalloc(size_t size)
{
   return new T[size];
}

template <typename T>
T* deviceMalloc(size_t size)
{
   T* pMemory = nullptr;
   cudaError_t res = cudaMalloc(&pMemory, size * sizeof(T));
   if (res != cudaSuccess)
   {
      std::cout << "CUDA Error: " << cudaGetErrorString(res) << std::endl;
      // throw; // No exception handling for now
   }
   return pMemory;
}

template <typename T>
T* memcpyDeviceToHost(T* dest, T* src, size_t size)
{
   cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyDeviceToHost);
   return dest;
}

template <typename T>
T* memcpyHostToDevice(T* dest, T* src, size_t size)
{
   cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyHostToDevice);
   return dest;
}

}

}

#endif
