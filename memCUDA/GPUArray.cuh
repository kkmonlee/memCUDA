#ifndef GPUARRAYCUH
#define GPUARRAYCUH

#include <assert.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include "GPUUtils.cuh"

template<typename T>
class DeviceArray
{
protected:
	std::size_t m_size;
	std::vector<T*> md_ptr;

private:
	typedef DeviceArray<T>& reference;
	typedef const DeviceArray<T>& const_reference;

public:
	DeviceArray(int ngpu = 1) : m_size(0)
	{
		int ndev;
		checkCudaErrors(cudaGetDeviceCount(&ndev));
		if (ndev > ngpu)
		{
			std::stringstream msg;
			msg << "Error!" << ndev << " devices available, " << ngpu << " is needed." << std::endl;
			throw std::runtime_error(msg.str());
		}
		md_ptr.resize(ngpu, 0);
	}

	DeviceArray(const_reference src)
	{
		md_ptr.resize(src.ngpu(), 0);
		resize(src.m_size());
		copy(src);
	}

	reference operator=(const_reference rhs)
	{
		if (this != &rhs) {
			if (ngpu() != rhs.ngpu()) {
				md_ptr.resize(rhs.ngpu(), 0);
				resize(rhs.m_size());
			}
			else if (size() != rhs.m_size())
				resize(rhs.m_size());
			copy(rhs)
		}
		return *this;
	}

	void copy(const_reference src)
	{
		if (this != &src) {
			if (bytes() != src.bytes())
				throw std::out_of_range("The arrays are not of the same size.");
			for (int i = 0; i < ngpu() && i < src.ngpu(); i++)
				checkCudaErrors(cudaMemcpy(md_ptr[i], src.DevPtr(i), bytes(), cudaMemcpyDefault));
		}

	}

	// change  array size
	virtual void resize(std::size_t size)
	{
		m_size = size;
		for (int i = ngpu(); i--;)
		{
			checkCudaErrors(cudaSetDevice(i));
			if (md_ptr[i]) checkCudaErrors(cudaFree(md_ptr[i]));
			checkCudaErrors(cudaMalloc((void**)&md_ptr[i], bytes()));
			checkCudaErrors(cudaMemset(md_ptr[i], 0, bytes()));
		}
	}

	// set error elements to 0
	virtual void SetToZero()
	{
		for (int i = md_ptr.size(); i--;) {
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaMemset(md_ptr[i], 0, bytes()));
		}
	}

	// backdoor for arraysize to elements
	virtual std::size_t size() const { return m_size; }

	// backdoor for arraysize to bytes
	virtual std::size_t ngpu()  const { return md_ptr.size(); }

	// return point to gpu mem
	virtual T* DevPtr(int gpu_id = 0) const {
		assert(gpu_id < ngpu());
		return md_ptr[gpu_id];
	}

	virtual void UpdateSecondaryDevices()
	{
		for (int i = 1; i < ngpu(); i++)
			checkCudaErrors(cudaMemcpy(md_ptr[i], md_ptr[0], bytes(), cudaMemcpyDefault));

	}

	// delete and deallocate
	virtual void clear()
	{
		for (int i = ngpu(); i--;)
		{
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaFree(md_ptr[i]));
		}
		m_size = 0;
		md_ptr.clear();
	}

	// call it
	virtual ~DeviceArray() { clear(); }
};
template<typename T>
class DeviceHostArray : public DeviceArray < T >
{
protected:
	std::vector<T> mh_data;

private:
	typedef DeviceHostArray<T>& reference;
	typedef const DeviceHostArray<T>& const_reference;
	typedef DeviceArray<T> Base;

public:
	DeviceHostArray(int ngpu = 1) : Base(ngpu) {};
	DeviceHostArray(const_reference src) : Base(src) {
		mh_data.resize(src.size(), src.VecRef());
	}

	reference operator=(const_reference rhs)
	{
		if (this != &rhs) {
			if (this->ngpu() != rhs.ngpu()) {
				this->md_ptr.resize(rhs.ngpu(), 0);
				resize(rhs.size(), rhs.VecRef());
			}
			else if (this->size() != rhs.size())
				resize(rhs.size(), rhs.VecRef());
			this->copy(rhs);
		}
		return *this;
	}

	void resize(std::size_t size)
	{
		Base::resize(size);
		mh_data.resize(size);
	}

	void resize(std::size_t size, T value)
	{
		Base::resize(size);
		mh_data.resize(size, value);
	}

	void resize(std::size_t size, std::vector<T> &values)
	{
		Base::resize(size);
		mh_data.resize(size);
		
		assert(size <= values.size());
		for (int i = 0; i < size; i++)
			mh_data[i] = values[i];
	}

	void copy(const_reference src)
	{
		if (this != &src) {
			assert(this->size() == src.size());
			for (int i = 0; i < this->size(); i++)
				mh_data[i] = src[i];
		}
	}

	void copyUpdate(const_reference src)
	{
		if (this != &src) {
			assert(Base::size() == src.size());
			copy(src);
			UpdateDevice();
		}
	}

	void copy(DeviceArray<T> src) {
		Base::copy(src);
	}

	T& operator[](const std::size_t idx) { return mh_data.at(idx); }
	T& operator[](const std::size_t idx) const { return mh_data.at(idx); }

	
	std::vector<T>& VecRef() { return mh_data; }
	std::vector<T>& VecRef() const { return mh_data; }

	
	void UpdateDevice()
	{
		for (int i = this->ngpu(); i--;) {
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaMemcpy(Base::DevPtr(i), mh_data.data(), Base::bytes(), cudaMemcpyDefault));
		}
	}

	
	void UpdateHost() { checkCudaErrors(cudaMemcpy(mh_data.data(), Base::DevPtr(), Base::bytes(), cudaMemcpyDefault)); }

	
	void clear()
	{
		Base::clear();
		mh_data.clear();
	}

	 
	~DeviceHostArray() { clear(); }
};

#endif