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

CFDSolver.cpp
-------------
Definitions for the CFDSolver class

See CFDSolver.h for details

*/

#include "CFDSolver.h"

namespace goetia
{

namespace paimon
{

CFDSolver::~CFDSolver()
{
   if(_texture != nullptr) SDL_DestroyTexture(_texture);
   if( _tank != nullptr ) delete _tank;
}

CFDSolver::CFDSolver(
   SDL_Renderer* rend, 
   float dt, int Nx, int Ny, float dx, int np, int ni, 
   int px, int py, int ww, int wh
) :
   _tank(new FluidTank(dt, Nx, Ny, dx, np, ni)),
   _renderer(rend),
   _texture(nullptr),
   _position_x(px),
   _position_y(py),
   _world()
{
   _world = {0, 0, ww, wh};
   regenerateTexture();
}

CFDSolver::CFDSolver(CFDSolver& solv) :
   _tank(solv._tank), 
   _renderer(solv._renderer),
   _texture(solv._texture),
   _position_x(solv._position_x),
   _position_y(solv._position_y),
   _world(solv._world)
{}

void CFDSolver::regenerateTexture()
{
   if(_texture != nullptr) SDL_DestroyTexture(_texture);
   _texture = SDL_CreateTexture(
      _renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, 
      _tank->getNx(), _tank->getNy()
   );
   SDL_SetTextureBlendMode(_texture, SDL_BLENDMODE_BLEND);
}

void CFDSolver::draw(int vx, int vy)
{
   int x = _position_x - vx;
   int y = _position_y - vy;
   int tempHeight = _tank->getNy();
   int tempWidth  = _tank->getNx();
   SDL_Rect dest  = {x, y, tempWidth, tempHeight};

   SDL_RenderCopyEx(_renderer, _texture, &_world, &dest, 0.0, NULL, SDL_FLIP_NONE);
}

void CFDSolver::update(uint32_t ticks)
{
   _tank->update(ticks);

   unsigned char* pixels = nullptr;
   int pitch = 0;

   if( _tank->getTexture() != nullptr )
   {
      SDL_LockTexture(_texture, NULL, reinterpret_cast<void**>(&pixels), &pitch);
      std::memcpy(
         pixels, _tank->getTexture(),
         _tank->getNx() * _tank->getNy() * 4 * sizeof(unsigned char)
      );
      SDL_UnlockTexture(_texture);
   }
   
}

void CFDSolver::addDensity(int px, int py, float opacity)
{
   px = (px - _position_x);
   py = (py - _position_y);

   if( px > -1 && px < _tank->getNx() )
      if( py > -1 && py < _tank->getNy() )
         _tank->addDensity(px, py, opacity);
}

void CFDSolver::addVelocity(int px, int py, float vx, float vy) 
{
   px = (px - _position_x);
   py = (py - _position_y);

   if( px > -1 && px < _tank->getNx() )
      if( py > -1 && py < _tank->getNy() )
         _tank->addVelocity(px, py, vx, vy);
}

void CFDSolver::addObstruction(int px, int py, float opacity) 
{
   px = (px - _position_x);
   py = (py - _position_y);

   if( px > -1 && px < _tank->getNx() )
      if( py > -1 && py < _tank->getNy() )
         _tank->addObstruction(px, py, opacity);

   // px = (px / static_cast<float>(_world.w)) * _tank->getNx();
   // py = (py / static_cast<float>(_world.h)) * _tank->getNy();
}

void CFDSolver::setGravity(float fx, float fy)
{
   _tank->setGravity(fx, fy);
}

void CFDSolver::setColorMode(int cm)
{
   _tank->setColorMode(cm);
}

void CFDSolver::setColormapMode(int gm)
{
   _tank->setColormapMode(gm);
}

// void CFDSolver::setCustomColormap(float4 a, float4 b, float4 c, float4 d)
// {
//    // _tank->setCustomColormap(a, b, c, d);
// }

void CFDSolver::setKelvinPerUnitDensity(float kpd)
{
   _tank->setKelvinPerUnitDensity(kpd);
}

} // paimon

} // goetia
