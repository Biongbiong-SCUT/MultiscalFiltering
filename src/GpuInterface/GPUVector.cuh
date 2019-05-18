#ifndef _GPUVECTOR_CUH
#define _GPUVECTOR_CUH

#include "math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

struct Vector3d
{
	double data[3];
	__host__ __device__ Vector3d() {
		data[0] = data[1] = data[2] = 0.0;
	}

	__host__ __device__ Vector3d(double x, double y, double z) {
		data[0] = x;
		data[1] = y;
		data[2] = z;
	}

	__host__ __device__ Vector3d(const Vector3d &p) {
		data[0] = p.data[0];
		data[1] = p.data[1];
		data[2] = p.data[2];
	}

	__host__ __device__ Vector3d(const double d[]) {

		data[0] = d[0];
		data[1] = d[1];
		data[2] = d[2];
	}

	__host__ __device__ const Vector3d operator + (const Vector3d & p) const {

		return Vector3d(data[0] + p.data[0],
			data[1] + p.data[1],
			data[2] + p.data[2]);
	}

	__host__ __device__ const Vector3d operator += (const Vector3d &p) {
		data[0] += p.data[0];
		data[1] += p.data[1];
		data[2] += p.data[2];

		return *this;
	}

	__host__ __device__ const Vector3d operator - (const Vector3d &p)const {
		return Vector3d(data[0] - p.data[0],
			data[1] - p.data[1],
			data[2] - p.data[2]);
	}

	__host__ __device__ const Vector3d operator -= (const Vector3d &p) {
		data[0] -= p.data[0];
		data[1] -= p.data[1];
		data[2] -= p.data[2];
		return *this;
	}

	__host__ __device__ const Vector3d operator / (double t)const {
		return Vector3d(data[0] / t, data[1] / t, data[2] / t);
	}

	__host__ __device__ const Vector3d operator /= (double t) {
		data[0] /= t;
		data[1] /= t;
		data[2] /= t;
		return *this;
	}

	__host__ __device__ const Vector3d operator * (double scale) {
		return Vector3d(data[0] * scale,
			data[1] * scale,
			data[2] * scale);
	}

	__host__ __device__ const Vector3d operator *= (double scale) {
		data[0] *= scale;
		data[1] *= scale;
		data[2] *= scale;
		return *this;
	}
	__host__ __device__ double operator | (const Vector3d &p) {
		return data[0] * p.data[0] +
			data[1] * p.data[1] +
			data[2] * p.data[2];
	}

	__host__ __device__ const Vector3d operator % (const Vector3d& p)const {

		return Vector3d(
			data[1] * p.data[2] - data[2] * p.data[1],
			data[2] * p.data[0] - data[0] * p.data[2],
			data[0] * p.data[1] - data[1] * p.data[0]
		);

	}

	__host__ __device__ void normalize() {
		double len = norm();
		data[0] /= len;
		data[1] /= len;
		data[2] /= len;
	}
	__host__ __device__ double norm()const {
		return sqrt((data[0] * data[0] + data[1] * data[1] + data[2] * data[2]));

	}

	__host__ __device__ double len()const {
		return sqrt((data[0] * data[0] + data[1] * data[1] + data[2] * data[2]));
	}
	__host__ __device__ double &operator[] (int idx) {
		return data[idx];
	}
	__host__ __device__ double operator[] (int idx)const {
		return data[idx];
	}
};

#endif