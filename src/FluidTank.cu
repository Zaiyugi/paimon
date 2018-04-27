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

FluidTank.cu
------------
Definitions for the FluidTank class

See FluidTank.h for details

*/

#include "FluidTank.h"
#include "DataObject.h"
#include "ExecutionPolicy.cuh"

#include "cuda/fluidsim_kernels.cu"

namespace goetia
{

namespace paimon
{

void FluidTank::init()
{
   // Init density, velocity and color to 0
   // Init force to constant force
   // Init obstruction to none
   _threads_per_block_x = 128;

   // Round _Nx to next multiple of 128
   if(_Nx % 128)
      _Nx = _Nx + (128 - _Nx % 128);

   // Round _Ny to next multiple of 128
   if(_Ny % 128)
      _Ny = _Ny + (128 - _Ny % 128);

   _grid_size_x = (_Nx * _Ny) / _threads_per_block_x;
   if(_grid_size_x * _threads_per_block_x < static_cast<size_t>(_Nx * _Ny)) _grid_size_x++;
   std::cout << "Paimon (LOG): Grid Size: " << _grid_size_x << std::endl;
   std::cout << "Paimon (LOG): Thread block size: " << _threads_per_block_x << std::endl;

   _density[0] = DataObject<float>(_Nx * _Ny);
   _density[1] = DataObject<float>(_Nx * _Ny);

   _velocity[0] = DataObject<float2>(_Nx * _Ny);
   _velocity[1] = DataObject<float2>(_Nx * _Ny);

   _pressure[0] = DataObject<float>(_Nx * _Ny, goetia::DataObject<float>::DEVICEONLY);
   _pressure[1] = DataObject<float>(_Nx * _Ny, goetia::DataObject<float>::DEVICEONLY);
   _diverge = DataObject<float>(_Nx * _Ny, goetia::DataObject<float>::DEVICEONLY);

   _color = DataObject<float4>(_Nx * _Ny, goetia::DataObject<float4>::DEVICEONLY);
   _force = DataObject<float2>(_Nx * _Ny);
   _obstruction = DataObject<float>(_Nx * _Ny);
   
   _byte_tex = new DataObject<unsigned char>(_Nx * _Ny * 4);

   _gravity_constant[0] = 0.0;
   _gravity_constant[1] = 15.0;

   goetia::util::ExecutionPolicy policy(_threads_per_block_x, _grid_size_x);
   initAll_krnl<<<policy.gridSize(), policy.blockSize()>>>(
      _Nx, _Ny,
      _density[0].getDevicePointer(),
      _velocity[0].getDevicePointer(),
      
      _density[1].getDevicePointer(),
      _velocity[1].getDevicePointer(),

      _obstruction.getDevicePointer(),
      _force.getDevicePointer(),
      make_float2(_gravity_constant[0], _gravity_constant[1])
   );

   _density[0].updateHost();
   _velocity[0].updateHost();

   _density[1].updateHost();
   _velocity[1].updateHost();

   _obstruction.updateHost();
   _force.updateHost();
}

FluidTank::~FluidTank()
{
   if( _byte_tex != nullptr )
      delete _byte_tex;
}

void FluidTank::resetSystem()
{
   // Clear Color and Density arrays
   goetia::util::ExecutionPolicy policy(_threads_per_block_x, _grid_size_x);
   initAll_krnl<<<policy.gridSize(), policy.blockSize()>>>(
      _Nx, _Ny,
      _density[0].getDevicePointer(),
      _velocity[0].getDevicePointer(),
      
      _density[1].getDevicePointer(),
      _velocity[1].getDevicePointer(),

      _obstruction.getDevicePointer(),
      _force.getDevicePointer(),
      make_float2(_gravity_constant[0], _gravity_constant[1])
   );

   _density[0].updateHost();
   _velocity[0].updateHost();

   _density[1].updateHost();
   _velocity[1].updateHost();

   _obstruction.updateHost();
   _force.updateHost();

}

void FluidTank::addDensity(int px, int py, float opacity)
{
   if( !isValid(px, py) )
      return;

   int ndx = coord(px, py);

   // Inject density
   (*_density[_density_frontbuf])[ndx] += opacity * (*_obstruction)[ndx];

   _added_paint = true;
}

void FluidTank::addVelocity(int px, int py, float vx, float vy)
{
   if( !isValid(px, py) )
      return;

   int ndx = coord(px, py);

   // Inject velocity
   (*_velocity[_velocity_frontbuf])[ndx].x += vx;
   (*_velocity[_velocity_frontbuf])[ndx].y += vy;

   _added_paint = true;
}

void FluidTank::addObstruction(int px, int py, float opacity) 
{
   if( !isValid(px, py) )
      return;

   int ndx = coord(px, py);

   // Inject obstruction
   (*_obstruction)[ndx] *= opacity;

   _added_paint = true;
}

void FluidTank::addForce(int px, int py, float fx, float fy)
{
   if( !isValid(px, py) )
      return;

   int ndx = coord(px, py);

   (*_force)[ndx].x += fx;
   (*_force)[ndx].y += fy;

   _added_paint = true;
}

void FluidTank::update(uint32_t ticks)
{
   float timestep = _dt * ticks;
   goetia::util::ExecutionPolicy policy(_threads_per_block_x, _grid_size_x);

   // If user has changed one of the fields, update device memory
   if( _added_paint )
   {
      _density[_density_frontbuf].updateDevice();
      _velocity[_velocity_frontbuf].updateDevice();
      _obstruction.updateDevice();
      _force.updateDevice();
   }

   // Advect density, velocity and color fields
   advect_krnl<<< policy.gridSize(), policy.blockSize() >>>(
      _Nx, _Ny, timestep, _dx,
      _density[_density_frontbuf].getDevicePointer(), 
      _velocity[_velocity_frontbuf].getDevicePointer(), 
      _density[1-_density_frontbuf].getDevicePointer(), 
      _velocity[1-_velocity_frontbuf].getDevicePointer()
   );

   _density_frontbuf = 1 - _density_frontbuf;
   _velocity_frontbuf = 1 - _velocity_frontbuf;

   // Add sources
   sources_krnl<<< policy.gridSize(), policy.blockSize() >>>(
      _Nx, _Ny, timestep,
      _density[_density_frontbuf].getDevicePointer(), 
      _velocity[_velocity_frontbuf].getDevicePointer(), 
      _force.getDevicePointer()
   );

   // Break stuff
   for(int i = 0; i < _Niop; ++i)
   {
      // Handle boundary conditions
      boundaries_krnl<<< policy.gridSize(), policy.blockSize() >>>(
         _Nx, _Ny,
         _density[_density_frontbuf].getDevicePointer(), 
         _velocity[_velocity_frontbuf].getDevicePointer(), 
         _obstruction.getDevicePointer()
      );

      // Project to divergence-free velocity field 
      // 1. Calculate divergence from velocity
      // 2. Project pressure to divergence-free
      // 3. Convert pressure gradient to velocity

      initPressureDivergence_krnl<<< policy.gridSize(), policy.blockSize() >>>(
         _Nx, _Ny,
         _diverge.getDevicePointer(), 
         _pressure[_pressure_frontbuf].getDevicePointer(), 
         _pressure[1-_pressure_frontbuf].getDevicePointer()
      );
      
      calculateDivergence_krnl<<< policy.gridSize(), policy.blockSize() >>>(
         _Nx, _Ny, _dx,
         _velocity[_velocity_frontbuf].getDevicePointer(), 
         _diverge.getDevicePointer()
      );

      for(int j = 0; j < _Nprojections; ++j)
      {
         calculatePressure_krnl<<< policy.gridSize(), policy.blockSize() >>>(
            _Nx, _Ny, _dx,
            _diverge.getDevicePointer(), 
            _pressure[_pressure_frontbuf].getDevicePointer(), 
            _pressure[1-_pressure_frontbuf].getDevicePointer()
         );

         _pressure_frontbuf = 1 - _pressure_frontbuf;
      }

      calculateVelocityFromPressure_krnl<<< policy.gridSize(), policy.blockSize() >>>(
         _Nx, _Ny, _dx,
         _velocity[_velocity_frontbuf].getDevicePointer(), 
         _pressure[_pressure_frontbuf].getDevicePointer()
      );
   }

   if( _color_mode == PAIMON_COLOR_DENSITY )
   {
      setColorToDensity_krnl<<< policy.gridSize(), policy.blockSize() >>>(
         _Nx, _Ny, 
         _density[_density_frontbuf].getDevicePointer(), 
         _color.getDevicePointer()
      );
   }
   else if( _color_mode == PAIMON_COLOR_COLORMAP )
   {
      setColorToColormappedDensity_krnl<<< policy.gridSize(), policy.blockSize() >>>(
         _Nx, _Ny, _colormap.A, _colormap.B, _colormap.C, _colormap.D,
         _density[_density_frontbuf].getDevicePointer(), 
         _color.getDevicePointer()
      );
   }
   else if( _color_mode == PAIMON_COLOR_BLACKBODY )
   { // Map density to temperature
      setColorToMappedTemperature_krnl<<< policy.gridSize(), policy.blockSize() >>>(
         _Nx, _Ny, _kelvin_per_d,
         _density[_density_frontbuf].getDevicePointer(), 
         _color.getDevicePointer()
      );
   }
   
   packColorToByteArray_krnl<<< policy.gridSize(), policy.blockSize() >>>(
      _Nx, _Ny, _color.getDevicePointer(), _byte_tex->getDevicePointer()
   );

   // Update host
   _density[_density_frontbuf].updateHost();
   _velocity[_velocity_frontbuf].updateHost();
   _byte_tex->updateHost();
}

void FluidTank::setGravity(float fx, float fy)
{
   _gravity_constant[0] = fx;
   _gravity_constant[1] = fy;
   resetSystem();
}

void FluidTank::setColormapMode(int gm)
{
   _selected_colormap = gm;
   switch(gm)
   {
      case PAIMON_COLORMAP_RAINBOW:
         _colormap.A = make_float4(0.50, 0.50, 0.50, 0.0);
         _colormap.B = make_float4(0.50, 0.50, 0.50, 0.0);
         _colormap.C = make_float4(1.00, 1.00, 1.00, 0.0);
         _colormap.D = make_float4(0.00, 0.33, 0.67, 0.0);
         break;

      case PAIMON_COLORMAP_DESERT:
         _colormap.A = make_float4(0.50, 0.50, 0.50, 0.0);
         _colormap.B = make_float4(0.50, 0.50, 0.50, 0.0);
         _colormap.C = make_float4(1.00, 1.00, 1.00, 0.0);
         _colormap.D = make_float4(0.00, 0.10, 0.20, 0.0);
         break;

      case PAIMON_COLORMAP_MEDICAL:
         _colormap.A = make_float4(0.50, 0.50, 0.50, 0.0);
         _colormap.B = make_float4(0.50, 0.50, 0.50, 0.0);
         _colormap.C = make_float4(1.00, 1.00, 1.00, 0.0);
         _colormap.D = make_float4(0.30, 0.20, 0.20, 0.0);
         break;

      case PAIMON_COLORMAP_NUCLEAR:
         _colormap.A = make_float4(0.50, 0.50, 0.50, 0.0);
         _colormap.B = make_float4(0.50, 0.50, 0.50, 0.0);
         _colormap.C = make_float4(1.00, 1.00, 0.50, 0.0);
         _colormap.D = make_float4(0.80, 0.90, 0.30, 0.0);
         break;

      case PAIMON_COLORMAP_SUNSET:
         _colormap.A = make_float4(0.50, 0.50, 0.50, 0.0);
         _colormap.B = make_float4(0.50, 0.50, 0.50, 0.0);
         _colormap.C = make_float4(1.00, 0.70, 0.40, 0.0);
         _colormap.D = make_float4(0.00, 0.15, 0.20, 0.0);
         break;

      case PAIMON_COLORMAP_CANDY:
         _colormap.A = make_float4(0.50, 0.50, 0.50, 0.0);
         _colormap.B = make_float4(0.50, 0.50, 0.50, 0.0);
         _colormap.C = make_float4(2.00, 1.00, 0.00, 0.0);
         _colormap.D = make_float4(0.50, 0.20, 0.25, 0.0);
         break;

      case PAIMON_COLORMAP_ROSE:
         _colormap.A = make_float4(0.80, 0.50, 0.40, 0.0);
         _colormap.B = make_float4(0.20, 0.40, 0.20, 0.0);
         _colormap.C = make_float4(2.00, 1.00, 1.00, 0.0);
         _colormap.D = make_float4(0.00, 0.25, 0.25, 0.0);
         break;

      case PAIMON_COLORMAP_NONE:
      default:
         ;
   }
}

void FluidTank::setCustomColormap(float4 A, float4 B, float4 C, float4 D)
{
   _selected_colormap = PAIMON_COLORMAP_CUSTOM;
   _colormap.A = A;
   _colormap.B = B;
   _colormap.C = C;
   _colormap.D = D;
}

const float2* FluidTank::getForce(const int x, const int y) const
{
   if( !isValid(x, y) )
      return nullptr;

   return (*_force) + (2 * coord(x, y));
}

bool FluidTank::isValid(int x, int y) const
{
   return ((x >= 0 && x < _Nx) && (y >= 0 && y < _Ny));
}

int FluidTank::coord(int x, int y) const
{
   if(!isValid(x, y))
      return -1;

   return x + _Nx * y;
}

} // paimon

} // goetia
