/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <ctime>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include "particles_kernel_impl.cuh"
#include "constant.h"

extern "C"
{

void cudaInit(int argc, char **argv)
{
	int devID;

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	devID = findCudaDevice(argc, (const char **)argv);

	if (devID < 0)
	{
		printf("No CUDA Capable devices found, exiting...\n");
		exit(EXIT_SUCCESS);
	}
}

void cudaGLInit(int argc, char **argv)
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	findCudaGLDevice(argc, (const char **)argv);
}

void allocateArray(void **devPtr, size_t size)
{
	checkCudaErrors(cudaMalloc(devPtr, size));
}

void freeArray(void *devPtr)
{
	checkCudaErrors(cudaFree(devPtr));
}

void threadSync()
{
	checkCudaErrors(cudaDeviceSynchronize());
}

void copyArrayToDevice(void *device, const void *host, int offset, int size)
{
	checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
}

void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
{
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
			cudaGraphicsMapFlagsNone));
}

void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
}

void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
{
	void *ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
			*cuda_vbo_resource));
	return ptr;
}

void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}

void copyArrayFromDevice(void *host, const void *device,
		struct cudaGraphicsResource **cuda_vbo_resource, int size)
{
	if (*cuda_vbo_resource)
	{
		device = mapGLBufferObject(cuda_vbo_resource);
	}

	checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

	if (*cuda_vbo_resource)
	{
		unmapGLBufferObject(*cuda_vbo_resource);
	}
}

void setParameters(SimParams *hostParams)
{
	// copy parameters to constant memory
	checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
}

//Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = iDivUp(n, numThreads);
}

void integrateSystem(float *pos,
		float *vel,
		float *rad,
		float deltaTime,
		uint numParticles)
{
	thrust::device_ptr<float4> d_pos4((float4 *)pos);
	thrust::device_ptr<float4> d_vel4((float4 *)vel);
	thrust::device_ptr<float>  d_rad(rad);

	thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4, d_rad)),
			thrust::make_zip_iterator(thrust::make_tuple(d_pos4+numParticles, d_vel4+numParticles, d_rad+numParticles)),
			integrate_functor(deltaTime));
}

void calcHash(uint  *gridParticleHash,
		uint  *gridParticleIndex,
		float *pos,
		int    numParticles)
{
	uint numThreads, numBlocks;
	computeGridSize(numParticles, 256, numBlocks, numThreads);

	// execute the kernel
	calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
			gridParticleIndex,
			(float4 *) pos,
			numParticles);

	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");
}

void reorderDataAndFindCellStart(uint  *cellStart,
		uint  *cellEnd,
		float *sortedPos,
		float *sortedVel,
		float *sortedRad,
		uint  *gridParticleHash,
		uint  *gridParticleIndex,
		float *oldPos,
		float *oldVel,
		float *oldRad,
		uint   numParticles,
		uint   numCells)
{
	uint numThreads, numBlocks;
	computeGridSize(numParticles, 256, numBlocks, numThreads);

	// set all cells to empty
	checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));



	uint smemSize = sizeof(uint)*(numThreads+1);
	reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
			cellStart,
			cellEnd,
			(float4 *) sortedPos,
			(float4 *) sortedVel,
			sortedRad,
			gridParticleHash,
			gridParticleIndex,
			(float4 *) oldPos,
			(float4 *) oldVel,
			oldRad,
			numParticles);
	getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");


}

void collide(float *newVel,
		float *sortedPos,
		float *sortedVel,
		float *sortedRad,
		uint  *gridParticleIndex,
		uint  *cellStart,
		uint  *cellEnd,
		uint   numParticles,
		uint   numCells)
{


	// thread per particle
	uint numThreads, numBlocks;
	computeGridSize(numParticles, 64, numBlocks, numThreads);

	// execute the kernel
	collideD<<< numBlocks, numThreads >>>((float4 *)newVel,
			(float4 *)sortedPos,
			(float4 *)sortedVel,
			sortedRad,
			gridParticleIndex,
			cellStart,
			cellEnd,
			numParticles);

	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");


}


void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
{
	thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
			thrust::device_ptr<uint>(dGridParticleHash + numParticles),
			thrust::device_ptr<uint>(dGridParticleIndex));
}



void rnd_init(curandState* devStates)
{
	allocateArray((void **) &devStates, MAX_NUM_PARTICLES * sizeof( curandState ));
	uint numThreads, numBlocks;
	computeGridSize(MAX_NUM_PARTICLES, 256, numBlocks, numThreads);
	// setup seeds
	unsigned long seed = (unsigned long) std::time(NULL);
	setup_kernel <<< numBlocks, numThreads >>> ( devStates, seed );
}

void changeRadius(float *radius, uint numParticles, curandState *devStates)
{
	uint numThreads, numBlocks;

	computeGridSize(numParticles, 64, numBlocks, numThreads);
	changeRadiusD<<<numBlocks, numThreads>>>(radius, numParticles, devStates);
}




}   // extern "C"
