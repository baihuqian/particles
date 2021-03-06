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

#include <GL/glew.h>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "render_particles.h"
#include "shaders.h"

#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#endif

ParticleRenderer::ParticleRenderer()
: m_pos(0),
  m_numParticles(0),
  m_pointSize(1.0f),
  m_particleRadius(5.0f * 0.5f),
  m_program(0),
  m_vbo(0),
  m_colorVBO(0),
  m_radiusVBO(0)
{
	_initGL();
}

ParticleRenderer::~ParticleRenderer()
{
	m_pos = 0;
}
/*
void ParticleRenderer::setPositions(float *pos, int numParticles)
{
    m_pos = pos;
    m_numParticles = numParticles;
}
 */



void ParticleRenderer::_drawPoints()
{

	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	if (m_colorVBO)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
		glColorPointer(4, GL_FLOAT, 0, 0);
		glEnableClientState(GL_COLOR_ARRAY);
	}
	if(m_radiusVBO)
	{
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, m_radiusVBO);
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
	}

	glDrawArrays(GL_POINTS, 0, m_numParticles);


	glDisableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
}

void ParticleRenderer::display(DisplayMode mode /* = PARTICLE_POINTS */) // called in particles.cpp -> display()
{
	switch (mode)
	{
	case PARTICLE_POINTS:
		glColor3f(1, 1, 1);
		glPointSize(m_pointSize);
		_drawPoints();
		break;

	default:
	case PARTICLE_SPHERES:
		glEnable(GL_POINT_SPRITE_ARB);
		glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
		glDepthMask(GL_TRUE);
		glEnable(GL_DEPTH_TEST);

		glUseProgram(m_program);
		glUniform1f(glGetUniformLocation(m_program, "pointScale"), m_window_h / tanf(m_fov*0.5f*(float)M_PI/180.0f));
		//glUniform1f(glGetUniformLocation(m_program, "pointRadius"), m_particleRadius);

		glColor3f(1, 1, 1);
		_drawPoints();

		glUseProgram(0);
		glDisable(GL_POINT_SPRITE_ARB);
		break;
	}
}

GLuint
ParticleRenderer::_compileProgram(const char *vsource, const char *fsource)
{
	// create empty shader
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	// link shader source code
	glShaderSource(vertexShader, 1, &vsource, 0);
	glShaderSource(fragmentShader, 1, &fsource, 0);

	// compile the source code
	glCompileShader(vertexShader);
	glCompileShader(fragmentShader);

	GLuint program = glCreateProgram();

	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);

	glBindAttribLocationARB(program, 1, "pointRadius");

	glLinkProgram(program);

	// check if program linked
	GLint success = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &success); // get linked status of program and store it in success

	if (!success)
	{
		char temp[256];
		glGetProgramInfoLog(program, 256, 0, temp);
		printf("Failed to link program:\n%s\n", temp);
		glDeleteProgram(program);
		program = 0;
	}

	return program;
}


void ParticleRenderer::_initGL()
{
	m_program = _compileProgram(vertexShader, spherePixelShader);

#if !defined(__APPLE__) && !defined(MACOSX)
	glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
	glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
#endif
}
