/*
Goetia
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

DataObject.h
-------------
Specification for the DataObject class

DataObject is a template class providing a wrapper for managing 
both host and device arrays. 

*/

#ifndef __DATAOBJECT_H__
#define __DATAOBJECT_H__

#include <iostream>
#include <memory>
#include <cstring>
#include <utility>

// float2 and float4 types when not compiling with nvcc
#include "float_variants.h"

// Forward declarations of DataObject
#include "DataObjectDefs.h"

#include "CudaDeleter.h"

namespace goetia
{

/* Datatype Policies */
template < typename T >
struct DataObjectType
{
   typedef T hostDataType;
   typedef T deviceDataType;
};

/* Ownership Policies */
template <typename T>
struct CreateUnique
{
   typedef typename DataObjectType<T>::hostDataType hostDataType;
   typedef typename DataObjectType<T>::deviceDataType deviceDataType;

   typedef typename std::unique_ptr<hostDataType[]> hostPointerType;
   typedef typename std::unique_ptr<deviceDataType, util::CudaDeleter> devicePointerType;
};

/* Class for storing both a host and device pointer */
template < typename T, typename PointerOwnershipPolicy >
class DataObject
{
   public:
      // Data-types for host and device pointers
      typedef typename DataObjectType<T>::hostDataType hostDataType;
      typedef typename DataObjectType<T>::deviceDataType deviceDataType;

      typedef typename PointerOwnershipPolicy::hostPointerType hostPointerType;
      typedef typename PointerOwnershipPolicy::devicePointerType devicePointerType;

      static const int HOSTDEVICE = 0;
      static const int HOSTONLY = 1;
      static const int DEVICEONLY = 2;

      // Constructors
      DataObject() : _size(0), _memory_space(HOSTDEVICE), _host_ptr(nullptr), _device_ptr(nullptr) {}

      DataObject(size_t size, int ms = HOSTDEVICE);

      // Disallow copying
      DataObject(DataObject& mo) = delete;

      // Move constructor
      DataObject(DataObject&& mo) noexcept :
         _size(mo._size), _memory_space(mo._memory_space),
         _host_ptr( std::move(mo._host_ptr) ),
         _device_ptr( std::move(mo._device_ptr) )
      {}

      ~DataObject() { /*std::cerr << "NOTE: Destroying DataObject" << std::endl;*/ }

      // Methods

      void mirror();

      void resize(size_t new_size);
      void extend(size_t amt);

      void updateDevice();
      void updateHost();

      deviceDataType* copyHostToDevice();
      hostDataType* copyDeviceToHost();

      // Getters
      hostDataType* getHostPointer() const
      { return _host_ptr.get(); }

      deviceDataType* getDevicePointer() const
      { return _device_ptr.get(); }

      size_t size() const { return _size; }

      // Operators

      // Move assignment
      DataObject& operator=(DataObject&& mo);

      // Disallow copy assignment
      DataObject& operator=(DataObject& mo) = delete;

      typename PointerOwnershipPolicy::hostDataType* operator*() const;

   private:
      // Members
      size_t _size;
      int _memory_space;
      hostPointerType _host_ptr;
      devicePointerType _device_ptr;

};

/* Functors */
struct getHostPointer_functor;
struct getDevicePointer_functor;
struct memcpyHostToDevice_functor;
struct memcpyDeviceToHost_functor;

/* Common DataObjects */
// Explicitly declare these so 
// the compiler will create them
template class DataObject<double>;
template class DataObject<float>;
template class DataObject<int>;
template class DataObject<char>;
template class DataObject<unsigned int>;
template class DataObject<unsigned char>;
template class DataObject<long unsigned int>;

# ifndef __CUDACC__
template class DataObject<goetia::float2>;
template class DataObject<goetia::float4>;
# else
template class DataObject<float2>;
template class DataObject<float4>;
# endif

}

#endif
