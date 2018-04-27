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

FluidTank.h
-------------
Specification for the FluidTank class

Provides all the heavy lifting for fluid simulation
Instantiated by the CFDSolver class defined in CFDSolver.h

*/

#ifndef __FLUIDTANK_H__
#define __FLUIDTANK_H__

#include <iostream>
#include <cstdlib>
#include <string>
#include <utility>
#include <cstdarg>

#include "float_variants.h"
#include "DataObject.h"

namespace goetia
{

namespace paimon
{

const static int PAIMON_COLOR_NONE        = -1;
const static int PAIMON_COLOR_DENSITY     =  0;
const static int PAIMON_COLOR_COLORMAP    =  1;
const static int PAIMON_COLOR_BLACKBODY   =  2;

const static int PAIMON_COLORMAP_NONE     = -1;
const static int PAIMON_COLORMAP_RAINBOW  =  0;
const static int PAIMON_COLORMAP_DESERT   =  1;
const static int PAIMON_COLORMAP_MEDICAL  =  2;
const static int PAIMON_COLORMAP_NUCLEAR  =  3;
const static int PAIMON_COLORMAP_SUNSET   =  4;
const static int PAIMON_COLORMAP_CANDY    =  5;
const static int PAIMON_COLORMAP_ROSE     =  6;
const static int PAIMON_COLORMAP_CUSTOM   =  7;

struct Colormap
{
   float4 A;
   float4 B;
   float4 C;
   float4 D;

   Colormap() : A(), B(), C(), D() {}
   Colormap(Colormap& cm) : A(cm.A), B(cm.B), C(cm.C), D(cm.D) {}

   Colormap& operator=(Colormap& cm)
   {
      A = cm.A;
      B = cm.B;
      C = cm.C;
      D = cm.D;
      return *this;
   }
};

class FluidTank
{
   public:
      FluidTank() = delete;

      FluidTank(float dt, int Nx, int Ny, float dx, int np, int ni) : 
         _dt(dt), _Nx(Nx), _Ny(Ny), _dx(dx), _Nprojections(np), _Niop(ni),
         _density(), 
         _velocity(),
         _color(), 
         _pressure(), 
         _diverge(),
         _obstruction(), 
         _force(),
         _gravity_constant(),
         _threads_per_block_x(128),
         _grid_size_x(1),
         _density_frontbuf(0),
         _velocity_frontbuf(0),
         _pressure_frontbuf(0),
         _byte_tex(nullptr),
         _added_paint(false),
         _color_mode(PAIMON_COLOR_DENSITY),
         _selected_colormap(PAIMON_COLORMAP_NONE),
         _kelvin_per_d(4000.0),
         _colormap()
      {
         init();
      }

      ~FluidTank();

      void resetSystem();

      void addDensity(int px, int py, float opacity);
      void addObstruction(int px, int py, float opacity);
      void addVelocity(int px, int py, float vx, float vy);
      void addForce(int px, int py, float fx, float fy);

      void update(uint32_t ticks);

      // Adjust constants
      void setGravity(float fx, float fy);

      // Colors
      void setColorMode(int cm) { _color_mode = cm; }
      void setColormapMode(int gm);
      void setCustomColormap(float4 A, float4 B, float4 C, float4 D);
      void setKelvinPerUnitDensity(float kpd) { _kelvin_per_d = kpd; }

      // Getters
      int getNx() const { return _Nx; }
      int getNy() const { return _Ny; }
      float getDx() const { return _dx; }

      const unsigned char* getTexture() const { return _byte_tex->getHostPointer(); }
      const float2* getForce(const int x, const int y) const;

      FluidTank(FluidTank& ft) = delete;
      FluidTank& operator=(const goetia::paimon::FluidTank&) = delete;

   private:
      float _dt;
      int _Nx, _Ny;
      float _dx;
      int _Nprojections;
      int _Niop;

      goetia::DataObject<float> _density[2]; // density
      goetia::DataObject<float2> _velocity[2]; // velocity
      goetia::DataObject<float4> _color; // Color

      goetia::DataObject<float> _pressure[2]; // Pressure
      goetia::DataObject<float> _diverge; // divergence
      
      goetia::DataObject<float> _obstruction; // Obstructions
      goetia::DataObject<float2> _force; // divergence

      // "Constants"
      float _gravity_constant[2];

      // CUDA
      size_t _threads_per_block_x;
      size_t _grid_size_x;

      size_t _density_frontbuf;
      size_t _velocity_frontbuf;
      size_t _pressure_frontbuf;

      // Byte array for SDL texture
      goetia::DataObject<unsigned char>* _byte_tex;

      // Flags
      bool _added_paint;

      // Color
      int _color_mode;
      int _selected_colormap;
      float _kelvin_per_d;

      Colormap _colormap;

      // Methods

      void init();

      // Utilities
      bool isValid(int x, int y) const;
      int coord(int x, int y) const;

};

} // paimon

} // goetia

#endif
