#pragma once

#include "cuda_runtime.h"
#include "primitives.hpp"

__host__ __device__
float Min( float a, float b )
{
	if ( a < b )
	{
		return a;
	}

	return b;
}

__host__ __device__ 
float Max( float a, float b )
{
	if ( a > b )
	{
		return a;
	}

	return b;
}

__host__ __device__
bool IsRaySphereIntersect( const Sphere& sph, const Ray& ray, float& nearestIntersectionRoot )
{
	Vec3 L = ray.position - sph.position;
	float a = Vec3::dot( ray.direction, ray.direction );
	float b = 2 * Vec3::dot( ray.direction, L );
	float c = Vec3::dot( L, L ) - sph.radius * sph.radius;

	float discr = b * b - 4 * a*c;
	float t0, t1;

	if ( discr < 0.f )
	{
		return false;
	}

	if ( discr == 0.f )
	{
		t0 = t1 = -0.5f * b / a;
	}

	if ( discr > 0.f )
	{
		float q = ( b > 0 ) ?
			-0.5f * ( b + sqrtf( discr ) ) :
			-0.5f * ( b - sqrtf( discr ) );

		t0 = q / a;
		t1 = c / q;
	}

	if ( t0 > t1 )
	{
		float tmp = t0;
		t0 = t1;
		t1 = tmp;
	}

	if ( t0 < 0.f )
	{
		t0 = t1;

		if ( t0 < 0.f )
		{
			return false;
		}
	}

	nearestIntersectionRoot = t0;
	return true;
}

struct HitRecord
{
	float	t;		// Ray::pointAt() t parameter
	Vec3	point;	// surface hitpoint
	Vec3	normal;	// surface hitpoint normal 
	Sphere*	sphere;	// the thing that got hit
};

__host__ __device__
bool IsSphereHit( const Sphere& sph, const Ray& r, const float t_min, const float t_max, HitRecord& rec )
{
	float t = 0;
	if ( IsRaySphereIntersect( sph, r, t ) )
	{
		if ( t < t_max && t > t_min )
		{
			rec.t = t;
			rec.point = r.pointAt( t );
			rec.normal = ( rec.point - sph.position ) / sph.radius;
			rec.sphere = const_cast<Sphere*>( &sph );
			return true;
		}
	}

	return false;
}

__host__ __device__
bool FindClosestHitObject( Sphere* spheres, size_t nSpheres, const Ray& r,
	const float t_min, const float t_max, HitRecord& rec )
{
	HitRecord tempRecord;
	bool hitAnything = false;
	float closestHit = t_max;

	for( int i = 0; i < nSpheres; ++i )
	{
		if ( IsSphereHit( spheres[i], r, t_min, closestHit, tempRecord ) )
		{
			hitAnything = true;
			closestHit = tempRecord.t;
			rec = tempRecord;
		}
	}

	return hitAnything;
}

__host__ __device__
Vec3 Reflect( const Vec3& inbound, const Vec3& surfaceNormal )
{
	return inbound - 2.f * Vec3::dot( inbound, surfaceNormal ) * surfaceNormal;
}

__host__ __device__
Vec3 Refract( const Vec3& inbound, const Vec3& surfaceNormal,
	const float eta_t, const float eta_i = 1.f)
{
	float cosine = -1 * Max( -1.f, Min( 1.f, Vec3::dot( inbound, surfaceNormal ) ) );

	if ( cosine < 0 )
	{
		return Refract( inbound, -1 * surfaceNormal, eta_i, eta_t );
	}

	float eta = eta_i / eta_t;
	float k = 1 - eta * eta * ( 1 - cosine * cosine );

	if ( k < 0 )
	{
		return Vec3( 1.f, 0.f, 0.f );
	}
	else
	{
		return inbound * eta + surfaceNormal * ( eta * cosine - sqrtf( k ) );
	}
}