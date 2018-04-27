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

DataObject.cu
-------------
Definitions for the DataObject class

See DataObject.h for details

*/

#include "DataObject.h"

// Include CUDA memory utilities
#include "MemUtils.cuh"

namespace goetia
{

/* Class for storing both a host and device pointer */
template < typename T, typename PointerOwnershipPolicy >
DataObject<T, PointerOwnershipPolicy>::DataObject(size_t size, int ms) :
   _size(size), _memory_space(ms), _host_ptr(), _device_ptr()
{
   _host_ptr = nullptr;
   _device_ptr = nullptr;

   if( _memory_space != DEVICEONLY )
      _host_ptr = typename PointerOwnershipPolicy::hostPointerType( util::hostMalloc<hostDataType>(_size) );

   if( _memory_space != HOSTONLY )
      _device_ptr = typename PointerOwnershipPolicy::devicePointerType( util::deviceMalloc<deviceDataType>(_size) );
}

// Methods

template < typename T, typename PointerOwnershipPolicy >
void DataObject<T, PointerOwnershipPolicy>::mirror()
{
   if( _memory_space == HOSTDEVICE )
   {
      std::cerr << "WARNING: DataObject already contains both host-side and device-side memory" << std::endl;
   } else if( _memory_space == DEVICEONLY )
   {
      std::cerr << "NOTE: Mirroring to host" << std::endl;
      _host_ptr = typename PointerOwnershipPolicy::hostPointerType( util::hostMalloc<hostDataType>(_size) );
      _memory_space = HOSTDEVICE;
   } else if( _memory_space == HOSTONLY )
   {
      std::cerr << "NOTE: Mirroring to device" << std::endl;
      _device_ptr = typename PointerOwnershipPolicy::devicePointerType( util::deviceMalloc<deviceDataType>(_size) );
      _memory_space = HOSTDEVICE;
   }
}

template < typename T, typename PointerOwnershipPolicy >
void DataObject<T, PointerOwnershipPolicy>::resize(size_t new_size)
{
   if( _memory_space == DEVICEONLY )
   {
      std::cerr << "ERROR: Cannot resize device only DataObject" << std::endl;
      return;
   }

   // Get current pointer
   hostDataType* curr = _host_ptr.get();

   // Allocate larger size, then copy over old contents
   hostDataType* new_ptr = util::hostMalloc<hostDataType>(new_size);
   std::memcpy(new_ptr, curr, _size * sizeof(hostDataType));

   // Delete old content, then give _host_ptr ownership of new_ptr
   _host_ptr.reset(nullptr);
   _host_ptr = typename PointerOwnershipPolicy::hostPointerType(new_ptr);
   _size = new_size;

   if( _memory_space != HOSTONLY )
   {
      _device_ptr.reset(nullptr);
      _device_ptr = typename PointerOwnershipPolicy::devicePointerType( util::deviceMalloc<deviceDataType>(_size) );
   }
}

template < typename T, typename PointerOwnershipPolicy >
void DataObject<T, PointerOwnershipPolicy>::extend(size_t amt)
{
   resize(_size + amt);
}

template < typename T, typename PointerOwnershipPolicy >
void DataObject<T, PointerOwnershipPolicy>::updateDevice()
{
   if( _memory_space == HOSTONLY )
   {
      std::cerr <<
         "ERROR: "
         "DataObject is host only; "
         "use mirror() to allocate device memory"
         << std::endl;
      return;
   }

   util::memcpyHostToDevice( _device_ptr.get(), _host_ptr.get(), _size );
}

template < typename T, typename PointerOwnershipPolicy >
void DataObject<T, PointerOwnershipPolicy>::updateHost()
{
   if( _memory_space == DEVICEONLY )
   {
      std::cerr <<
         "ERROR: "
         "DataObject is device only; "
         "use mirror() to allocate host memory"
         << std::endl;
      return;
   }

   util::memcpyDeviceToHost( _host_ptr.get(), _device_ptr.get(), _size );
}

template < typename T, typename PointerOwnershipPolicy >
typename DataObjectType<T>::deviceDataType* DataObject<T, PointerOwnershipPolicy>::copyHostToDevice()
{
   if( _memory_space == HOSTONLY )
   {
      std::cerr <<
         "ERROR: "
         "DataObject is host only; "
         "use mirror() to allocate device memory"
         << std::endl;
      return nullptr;
   }

   return util::memcpyHostToDevice(
      _device_ptr.get(),
      _host_ptr.get(),
      _size
   );
}

template < typename T, typename PointerOwnershipPolicy >
typename DataObjectType<T>::hostDataType* DataObject<T, PointerOwnershipPolicy>::copyDeviceToHost()
{
   if( _memory_space == DEVICEONLY )
   {
      std::cerr <<
         "ERROR: "
         "DataObject is device only; "
         "use mirror() to allocate host memory"
         << std::endl;
      return nullptr;
   }

   return util::memcpyDeviceToHost(
      _host_ptr.get(),
      _device_ptr.get(),
      _size
   );
}

// Operators

// Move assignment
template < typename T, typename PointerOwnershipPolicy >
DataObject<T, PointerOwnershipPolicy>& DataObject<T, PointerOwnershipPolicy>::operator=(
   DataObject<T, PointerOwnershipPolicy>&& mo
)
{
   _size = mo._size;
   _host_ptr = std::move(mo._host_ptr);
   _device_ptr = std::move(mo._device_ptr);
   return *this;
}

template < typename T, typename PointerOwnershipPolicy >
typename PointerOwnershipPolicy::hostDataType* DataObject<T, PointerOwnershipPolicy>::operator*() const
{
   return _host_ptr.get();
}

/* End DataObject */

/* Functors */
struct getHostPointer_functor
{
   template <typename T>
   void operator() (DataObject<T>& obj, T** nt)
   {
      *nt = obj.getHostPointer();
   }

   template <typename T>
   void operator() (T& obj, T* nt)
   {
      *nt = obj;
   }

};

struct getDevicePointer_functor
{
   template <typename T>
   void operator() (DataObject<T>& obj, T** nt)
   {
      *nt = obj.getDevicePointer();
   }

   template <typename T>
   void operator() (T& obj, T* nt)
   {
      *nt = obj;
   }

};

struct memcpyHostToDevice_functor
{
   template <typename T>
   void operator() (DataObject<T>& obj)
   {
      obj.updateDevice();
   }

   template <typename T>
   void operator() (T& obj)
   {
      //std::cout << obj << std::endl;
   }

};

struct memcpyDeviceToHost_functor
{
   template <typename T>
   void operator() (DataObject<T>& obj)
   {
      obj.updateHost();
   }

   template <typename T>
   void operator() (T& obj)
   {
      //std::cout << obj << std::endl;
   }

};

}
