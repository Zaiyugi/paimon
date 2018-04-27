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

CudaDeleter.cu
--------------
Definition for the CudaDeleter functor

See CudaDeleter.h for details

*/

#include "CudaDeleter.h"
#include <iostream>
#include <cuda.h>

namespace goetia
{

namespace util
{

void CudaDeleter::operator()(void *p)
{
     //std::cerr << "NOTE: CudaDeleter: Free..." << std::endl;
	 if(p != nullptr)
	 {
	     cudaError_t res = cudaFree(p);
	     if (res != cudaSuccess)
	     {
	         std::cerr << "ERROR: CUDA Error at delete: " << cudaGetErrorString(res) << std::endl;
	     }
 	 }
}

}

}
