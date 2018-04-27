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

ExecutionPolicy.cuh
-------------------
Specification for the ExecutionPolicy

Provides a wrapper for the dim3 objects used to specify grid/block
sizes for CUDA kernels.

*/

#ifndef __EXECUTIONPOLICY_CUH__
#define __EXECUTIONPOLICY_CUH__

#include <iostream>
#include <cstdlib>
#include <cuda.h>

namespace goetia
{

namespace util
{

class ExecutionPolicy
{
   public:
      ExecutionPolicy() : _block_size(), _grid_size() {}

      ExecutionPolicy(size_t b_x, size_t g_x) : _block_size(b_x), _grid_size(g_x) {}

      ExecutionPolicy(size_t b_x, size_t b_y, size_t g_x, size_t g_y) : 
         _block_size(b_x, b_y), 
         _grid_size(g_x, g_y)
      {}

      ExecutionPolicy(size_t b_x, size_t b_y, size_t b_z, size_t g_x, size_t g_y, size_t g_z) : 
         _block_size(b_x, b_y, b_z), 
         _grid_size(g_x, g_y, g_z)
      {}

      ExecutionPolicy(ExecutionPolicy& ep) : 
         _block_size(ep._block_size),
         _grid_size(ep._grid_size)
      {}

      const dim3& blockSize() const { return _block_size; }
      const dim3& gridSize() const { return _grid_size; }

   private:
      dim3 _block_size;
      dim3 _grid_size;
};

}

}

#endif
