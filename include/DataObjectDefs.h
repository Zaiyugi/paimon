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

DataObjectDefs.h
-------------
Forward declarations for the DataObject class

Used to hide CUDA dependent code from g++

*/

#ifndef __DATAOBJECTDEFS_H__
#define __DATAOBJECTDEFS_H__

namespace goetia
{

/* Datatype Policies */
template < typename T >
struct DataObjectType;

/* Ownership Policies */
template <typename T>
struct CreateUnique;

/* Class for storing both a host and device pointer */
template < typename T, typename PointerOwnershipPolicy = CreateUnique<T> >
class DataObject;

}

#endif
