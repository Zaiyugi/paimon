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

CFDSolver.h
-----------
Specification for the CFDSolver class

Primary class for using Paimon.

*/

#ifndef __CFDSOLVER_H__
#define __CFDSOLVER_H__

#include <iostream>
#include <cstdlib>
#include <string>
#include <utility>

#include <SDL.h>

#include "FluidTank.h"
#include "float_variants.h"

namespace goetia
{

namespace paimon
{

class CFDSolver
{
   public:
      CFDSolver() = delete;

      CFDSolver(
         SDL_Renderer* rend, 
         float dt, int Nx, int Ny, float dx, int np, int ni,
         int px, int py, int ww, int wh
      );

      CFDSolver(CFDSolver& solv);

      ~CFDSolver();

      void regenerateTexture();

      void draw(int vx, int vy);
      void update(uint32_t ticks);

      void addDensity(int px, int py, float opacity);
      void addVelocity(int px, int py, float vx, float vy);
      void addObstruction(int px, int py, float opacity);

      // Adjust constants
      void setGravity(float fx, float fy);

      // Colors
      void setColorMode(int cm);
      void setColormapMode(int gm);
      // void setCustomColormap(float4 A, float4 B, float4 C, float4 D);
      void setKelvinPerUnitDensity(float kpd);

      // Getters
      int getNx() const { return _tank->getNx(); }
      int getNy() const { return _tank->getNy(); }
      float getDx() const { return _tank->getDx(); }
      int px() const { return _position_x; }
      int py() const { return _position_y; }

      // Operators
      CFDSolver& operator=(const CFDSolver& solv);

   private:
      FluidTank *_tank;

      SDL_Renderer* _renderer;
      SDL_Texture *_texture;

      int _position_x;
      int _position_y;

      SDL_Rect _world;
};

} // paimon

} // goetia

#endif
