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


#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"


#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>
#include <cmath>

#include "constant.h"
#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize) :
m_bInitialized(false),
m_numParticles(numParticles),
m_hPos(0),
m_hVel(0),
m_dPos(0),
m_dVel(0),
m_gridSize(gridSize),
m_timer(NULL)
{
	m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
	float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

	m_gridSortBits = 18;    // increase this for larger grids

	// set simulation parameters
	m_params.gridSize = m_gridSize;
	m_params.numCells = m_numGridCells;
	m_params.numBodies = m_numParticles;

	m_params.particleRadius = INIT_RADIUS;
	m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
	m_params.colliderRadius = 0.2f;

	m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
	m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

	m_params.spring = 0.5f;
	m_params.damping = 0.02f;
	m_params.shear = 0.1f;
	m_params.attraction = 0.0f;
	m_params.boundaryDamping = -0.5f;

	m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
	m_params.globalDamping = 1.0f;

	m_params.numParticles = m_numParticles;

	_initialize(numParticles);
}

ParticleSystem::~ParticleSystem()
{
	_finalize();
	m_numParticles = 0;
}

uint
ParticleSystem::createVBO(uint size)
{
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return vbo;
}

inline float lerp(float a, float b, float t)
{
	return a + t*(b-a);
}

// create a color ramp
void colorRamp(float t, float *r)
{
	const int ncolors = 7;
	float c[ncolors][3] =
	{
			{ 1.0, 0.0, 0.0, },
			{ 1.0, 0.5, 0.0, },
			{ 1.0, 1.0, 0.0, },
			{ 0.0, 1.0, 0.0, },
			{ 0.0, 1.0, 1.0, },
			{ 0.0, 0.0, 1.0, },
			{ 1.0, 0.0, 1.0, },
	};
	t = t * (ncolors-1);
	int i = (int) t;
	float u = t - floor(t);
	r[0] = lerp(c[i][0], c[i+1][0], u);
	r[1] = lerp(c[i][1], c[i+1][1], u);
	r[2] = lerp(c[i][2], c[i+1][2], u);
}

void
ParticleSystem::_initialize(int numParticles)
{
	assert(!m_bInitialized);

	m_numParticles = numParticles;

	// allocate host storage
	m_hPos = new float[MAX_NUM_PARTICLES*4];
	m_hVel = new float[MAX_NUM_PARTICLES*4];
	m_hRad = new float[MAX_NUM_PARTICLES*4];
	memset(m_hPos, 0, MAX_NUM_PARTICLES*4*sizeof(float));
	memset(m_hVel, 0, MAX_NUM_PARTICLES*4*sizeof(float));
	memset(m_hRad, m_params.particleRadius, MAX_NUM_PARTICLES*4*sizeof(float));

	m_hCellStart = new uint[m_numGridCells];
	memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

	m_hCellEnd = new uint[m_numGridCells];
	memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));

	// allocate GPU data
	unsigned int memSize = sizeof(float) * 4 * MAX_NUM_PARTICLES;

	// set up random number generator
	allocateArray((void **) &m_devStates, MAX_NUM_PARTICLES * sizeof(curandState));
	rnd_init(m_devStates);

	m_posVbo = createVBO(memSize);
	registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);

	m_radiusVBO = createVBO(sizeof(float) * MAX_NUM_PARTICLES);
	registerGLBufferObject(m_radiusVBO, &m_cuda_radiusvbo_resource);

	allocateArray((void **)&m_dVel, memSize);

	allocateArray((void **)&m_dSortedPos, memSize);
	allocateArray((void **)&m_dSortedVel, memSize);
	allocateArray((void **)&m_dSortedRad, sizeof(float) * MAX_NUM_PARTICLES);

	allocateArray((void **)&m_dGridParticleHash, MAX_NUM_PARTICLES*sizeof(uint));
	allocateArray((void **)&m_dGridParticleIndex, MAX_NUM_PARTICLES*sizeof(uint));

	allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
	allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));


	m_colorVBO = createVBO(memSize);
	registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

	// fill color buffer
	glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
	float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float *ptr = data;

	for (uint i=0; i<MAX_NUM_PARTICLES; i++)
	{
		float t = i / (float) MAX_NUM_PARTICLES;


		colorRamp(t, ptr);
		ptr+=3;

		*ptr++ = 1.0f;
	}

	glUnmapBufferARB(GL_ARRAY_BUFFER);

	glBindBufferARB(GL_ARRAY_BUFFER, m_radiusVBO);
	data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	ptr = data;

	for(uint i = 0; i < m_numParticles; i++) {
		*ptr++ = m_params.particleRadius;
	}

	glUnmapBufferARB(GL_ARRAY_BUFFER);

	sdkCreateTimer(&m_timer);

	setParameters(&m_params);

	// set up min and max radius
	m_minRadius = 0.5 * m_params.particleRadius;
	m_maxRadius = 2 * m_params.particleRadius;



	m_bInitialized = true;
}

void
ParticleSystem::_finalize()
{
	assert(m_bInitialized);

	delete [] m_hPos;
	delete [] m_hVel;
	delete [] m_hCellStart;
	delete [] m_hCellEnd;

	freeArray(m_dVel);
	freeArray(m_dSortedPos);
	freeArray(m_dSortedVel);

	freeArray(m_dGridParticleHash);
	freeArray(m_dGridParticleIndex);
	freeArray(m_dCellStart);
	freeArray(m_dCellEnd);
	// random number
	freeArray(m_devStates);
	//freeArray(m_rndNum);

	unregisterGLBufferObject(m_cuda_posvbo_resource);
	unregisterGLBufferObject(m_cuda_radiusvbo_resource);
	unregisterGLBufferObject(m_cuda_colorvbo_resource);
	glDeleteBuffers(1, (const GLuint *)&m_posVbo);
	glDeleteBuffers(1, (const GLuint *)&m_colorVBO);
	glDeleteBuffers(1, (const GLuint *)&m_radiusVBO);


}

// step the simulation
void
ParticleSystem::update(float deltaTime)
{
	assert(m_bInitialized);


	// update constants
	setParameters(&m_params);

	// check radius

	float *hPos = getArray(POSITION);
	float *hVel = getArray(VELOCITY);
	float *hRad = getArray(RADIUS);

	uint numParticles = checkRadius(
			hPos,
			hVel,
			hRad,
			m_numParticles,
			m_minRadius,
			m_maxRadius);
	setArray(POSITION, hPos, 0, MAX_NUM_PARTICLES);
	setArray(VELOCITY, hVel, 0, MAX_NUM_PARTICLES);
	setArray(RADIUS, hRad, 0, MAX_NUM_PARTICLES);

	m_numParticles = numParticles;

	float *dPos, *dRad;

	dPos = (float *) mapGLBufferObject(&m_cuda_posvbo_resource);
	dRad = (float *) mapGLBufferObject(&m_cuda_radiusvbo_resource);

	// integrate
	integrateSystem(
			dPos,
			m_dVel,
			dRad,
			deltaTime,
			m_numParticles);

	// calculate grid hash
	calcHash(
			m_dGridParticleHash,
			m_dGridParticleIndex,
			dPos,
			m_numParticles);

	// sort particles based on hash
	sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

	// reorder particle arrays into sorted order and
	// find start and end of each cell
	reorderDataAndFindCellStart(
			m_dCellStart,
			m_dCellEnd,
			m_dSortedPos,
			m_dSortedVel,
			m_dSortedRad,
			m_dGridParticleHash,
			m_dGridParticleIndex,
			dPos,
			m_dVel,
			dRad,
			m_numParticles,
			m_numGridCells);

	// process collisions
	collide(
			m_dVel,
			m_dSortedPos,
			m_dSortedVel,
			m_dSortedRad,
			m_dGridParticleIndex,
			m_dCellStart,
			m_dCellEnd,
			m_numParticles,
			m_numGridCells);

	changeRadius(dRad, m_numParticles, m_devStates);


	// note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
	unmapGLBufferObject(m_cuda_posvbo_resource);
	unmapGLBufferObject(m_cuda_radiusvbo_resource);
}

void
ParticleSystem::dumpGrid()
{
	// dump grid information
	copyArrayFromDevice(m_hCellStart, m_dCellStart, 0, sizeof(uint)*m_numGridCells);
	copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0, sizeof(uint)*m_numGridCells);
	uint maxCellSize = 0;

	for (uint i=0; i<m_numGridCells; i++)
	{
		if (m_hCellStart[i] != 0xffffffff)
		{
			uint cellSize = m_hCellEnd[i] - m_hCellStart[i];

			//            printf("cell: %d, %d particles\n", i, cellSize);
			if (cellSize > maxCellSize)
			{
				maxCellSize = cellSize;
			}
		}
	}

	printf("maximum particles per cell = %d\n", maxCellSize);
}

void
ParticleSystem::dumpParticles(uint start, uint count)
{
	// debug
	copyArrayFromDevice(m_hPos, 0, &m_cuda_posvbo_resource, sizeof(float)*4*count);
	copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(float)*4*count);

	for (uint i=start; i<start+count; i++)
	{
		//        printf("%d: ", i);
		printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
		printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i*4+0], m_hVel[i*4+1], m_hVel[i*4+2], m_hVel[i*4+3]);
	}
}

float *
ParticleSystem::getArray(ParticleArray array)
{
	assert(m_bInitialized);

	float *hdata = 0;
	float *ddata = 0;
	struct cudaGraphicsResource *cuda_vbo_resource = 0;

	switch (array)
	{
	default:
	case POSITION:
		hdata = m_hPos;
		ddata = m_dPos;
		cuda_vbo_resource = m_cuda_posvbo_resource;
		copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, MAX_NUM_PARTICLES*4*sizeof(float));
		break;

	case VELOCITY:
		hdata = m_hVel;
		ddata = m_dVel;
		copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, MAX_NUM_PARTICLES*4*sizeof(float));
		break;

	case RADIUS:
		hdata = m_hRad;
		ddata = m_dRad;
		cuda_vbo_resource = m_cuda_radiusvbo_resource;
		copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, MAX_NUM_PARTICLES*sizeof(float));
		break;
	}


	return hdata;
}

void
ParticleSystem::setArray(ParticleArray array, const float *data, int start, int count)
{
	assert(m_bInitialized);

	switch (array)
	{
	default:
	case POSITION:
	{

		unregisterGLBufferObject(m_cuda_posvbo_resource);
		glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
		glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);

	}
	break;

	case VELOCITY:
		copyArrayToDevice(m_dVel, data, start*4*sizeof(float), count*4*sizeof(float));
		break;


	case RADIUS:
		unregisterGLBufferObject(m_cuda_radiusvbo_resource);
		glBindBuffer(GL_ARRAY_BUFFER, m_radiusVBO);
		glBufferSubData(GL_ARRAY_BUFFER, start*sizeof(float), count*sizeof(float), data);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		registerGLBufferObject(m_radiusVBO, &m_cuda_radiusvbo_resource);
		break;
	}
}

inline float frand()
{
	return rand() / (float) RAND_MAX;
}

void
ParticleSystem::initGrid(uint *size, float spacing, float jitter, uint numParticles)
{
	srand(1973);

	for (uint z=0; z<size[2]; z++)
	{
		for (uint y=0; y<size[1]; y++)
		{
			for (uint x=0; x<size[0]; x++)
			{
				uint i = (z*size[1]*size[0]) + (y*size[0]) + x;

				if (i < numParticles)
				{
					m_hPos[i*4] = (spacing * x) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
					m_hPos[i*4+1] = (spacing * y) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
					m_hPos[i*4+2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
					m_hPos[i*4+3] = 1.0f;

					m_hVel[i*4] = 0.0f;
					m_hVel[i*4+1] = 0.0f;
					m_hVel[i*4+2] = 0.0f;
					m_hVel[i*4+3] = 0.0f;

					m_hRad[i] = m_params.particleRadius;
				}
			}
		}
	}
}

void
ParticleSystem::reset(ParticleConfig config)
{
	m_numParticles = m_params.numParticles;

	switch (config)
	{
	default:
	case CONFIG_RANDOM:
	{
		int p = 0, v = 0;
		for (uint i=0; i < m_numParticles; i++)
		{
			float point[3];
			point[0] = frand();
			point[1] = frand();
			point[2] = frand();
			m_hPos[p++] = 2 * (point[0] - 0.5f);
			m_hPos[p++] = 2 * (point[1] - 0.5f);
			m_hPos[p++] = 2 * (point[2] - 0.5f);
			m_hPos[p++] = 1.0f; // radius
			m_hVel[v++] = 0.0f;
			m_hVel[v++] = 0.0f;
			m_hVel[v++] = 0.0f;
			m_hVel[v++] = 0.0f;
			m_hRad[i] = m_params.particleRadius;
		}
	}
	break;

	case CONFIG_GRID:
	{
		float jitter = m_params.particleRadius*0.01f;
		uint s = (int) ceilf(powf((float) m_numParticles, 1.0f / 3.0f));
		uint gridSize[3];
		gridSize[0] = gridSize[1] = gridSize[2] = s;
		initGrid(gridSize, m_params.particleRadius*2.0f, jitter, m_numParticles);
	}
	break;
	}

	setArray(POSITION, m_hPos, 0, m_numParticles);
	setArray(VELOCITY, m_hVel, 0, m_numParticles);
	setArray(RADIUS, m_hRad, 0, m_numParticles);
}

void
ParticleSystem::addSphere(int start, float *pos, float *vel, int r, float spacing)
{
	uint index = start;

	for (int z=-r; z<=r; z++)
	{
		for (int y=-r; y<=r; y++)
		{
			for (int x=-r; x<=r; x++)
			{
				float dx = x*spacing;
				float dy = y*spacing;
				float dz = z*spacing;
				float l = sqrtf(dx*dx + dy*dy + dz*dz);
				float jitter = m_params.particleRadius*0.01f;

				if ((l <= m_params.particleRadius*2.0f*r) && (index < m_numParticles))
				{
					m_hPos[index*4]   = pos[0] + dx + (frand()*2.0f-1.0f)*jitter;
					m_hPos[index*4+1] = pos[1] + dy + (frand()*2.0f-1.0f)*jitter;
					m_hPos[index*4+2] = pos[2] + dz + (frand()*2.0f-1.0f)*jitter;
					m_hPos[index*4+3] = pos[3];

					m_hVel[index*4]   = vel[0];
					m_hVel[index*4+1] = vel[1];
					m_hVel[index*4+2] = vel[2];
					m_hVel[index*4+3] = vel[3];

					m_hRad[index] = m_params.particleRadius;
					index++;
				}
			}
		}
	}

	setArray(POSITION, m_hPos, start, index);
	setArray(VELOCITY, m_hVel, start, index);
	setArray(RADIUS, m_hRad, start, index);
}

uint ParticleSystem::checkRadius(float *position, float *velocity, float *radius, uint numParticles, float minRadius, float maxRadius)
{
	const float divisionRatio = std::pow(2.0f, 1.0f/3.0f); // division ratio that preserve mass and momentum
	//uint oldNumParticles = *numParticles;
	for(int i = numParticles - 1; i >= 0; i--)
	{
		//uint numP = numParticles;
		if(radius[i] > maxRadius)
		{


			if(numParticles < MAX_NUM_PARTICLES) { // one particle divide into two
				radius[i] /= divisionRatio;

				// randomly generate division direction
				std::srand(std::time(0));
				float phi = 2 * CUDART_PI_F * ((float)std::rand() / (float)RAND_MAX);
				std::srand(std::time(0));
				float theta = 2 * CUDART_PI_F * ((float)std::rand() / (float)RAND_MAX);

				position[4*numParticles] = position[4*i] + radius[i]/2 * std::sin(phi) * std::cos(theta);
				position[4*numParticles+1] = position[4*i+1] + radius[i]/2 * std::sin(phi) * std::sin(theta);
				position[4*numParticles+2] = position[4*i+2] + radius[i]/2 * std::cos(phi);
				position[4*numParticles+3] = position[4*i+3];

				position[4*i] -= radius[i]/2 * std::sin(phi) * std::cos(theta);
				position[4*i+1] -= radius[i]/2 * std::sin(phi) * std::sin(theta);
				position[4*i+2] -= radius[i]/2 * std::cos(phi);

				velocity[4*numParticles] = velocity[4*i];
				velocity[4*numParticles+1] = velocity[4*i+1];
				velocity[4*numParticles+2] = velocity[4*i+2];
				velocity[4*numParticles+3] = velocity[4*i+3];

				radius[numParticles] = radius[i];
				numParticles++;
			}
		}
		else if(radius[i] < minRadius)
		{
			if(numParticles > 0)
			{
				numParticles--;
				position[4*i] = position[4*numParticles];
				position[4*i+1] = position[4*numParticles+1];
				position[4*i+2] = position[4*numParticles+2];
				position[4*i+3] = position[4*numParticles+3];

				velocity[4*i] = velocity[4*numParticles];
				velocity[4*i+1] = velocity[4*numParticles+1];
				velocity[4*i+2] = velocity[4*numParticles+2];
				velocity[4*i+3] = velocity[4*numParticles+3];

				radius[i] = radius[numParticles];
			}
		}
	}
	return numParticles;
}
