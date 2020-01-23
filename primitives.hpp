#pragma once

#include "cuda_runtime.h"
#include <algorithm>

struct Vec3
{
	float x, y, z;
	
	__host__ __device__
	Vec3() {}

	__host__ __device__
	Vec3( float x, float y, float z ) : x( x ), y( y ), z( z ) {}

	__host__ __device__
	friend Vec3 operator+( const Vec3& lhs, const Vec3& rhs )
	{
		return Vec3( lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z );
	}

	__host__ __device__
	friend Vec3 operator-( const Vec3& lhs, const Vec3& rhs )
	{
		return Vec3( lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z );
	}

	__host__ __device__
	friend Vec3 operator*( const Vec3& v, float n )
	{
		return Vec3( v.x * n, v.y * n, v.z * n );
	}

	__host__ __device__
	friend Vec3 operator*( float n, const Vec3& v )
	{
		return Vec3( v.x * n, v.y * n, v.z * n );
	}

	__host__ __device__
	friend Vec3 operator*( const Vec3& v0, const Vec3& v1 )
	{
		return Vec3( v0.x * v1.x, v0.y * v1.y, v0.z * v1.z );
	}

	__host__ __device__
	friend Vec3 operator/( const Vec3& v, float n )
	{
		return Vec3( v.x / n, v.y / n, v.z / n );
	}

	__host__ __device__
	void operator +=( const Vec3& rhs )
	{
		x += rhs.x; y += rhs.y; z += rhs.z;
	}

	__host__ __device__
	void operator /=( float f )
	{
		x /= f; y /= f; z /= f;
	}

	__host__ __device__
	static float dot( const Vec3& v0, const Vec3& v1 )
	{
		return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
	}

	__host__ __device__
	static Vec3 cross( const Vec3& v0, const Vec3& v1 )
	{
		return Vec3( v0.y * v1.z - v0.z * v1.y,
			-( v0.x * v1.z - v0.z * v1.x ),
			v0.x * v1.y - v0.y * v1.x
		);
	}

	__host__ __device__
	float len() const
	{
		return sqrtf( x*x + y * y + z * z );
	}

	__host__ __device__
	static Vec3 unit( const Vec3& v )
	{
		return v * ( 1.f / v.len() );
	}
};

struct Material
{
	float diffuseAttenuation;
	float specularAttenuation;
	float reflectionAttenuation;
	float refractionAttenuation;

	Vec3 diffuseColor;
	float specularExponent;
	float refractionIndex;

	__host__ __device__ 
	Material() {}
};

struct Sphere
{
public:
	Vec3 position;
	float radius;
	Material material;

	__host__ __device__
	Sphere()
	{}

	__host__ __device__
	Sphere( Vec3 position, float radius, Material material )
		: position( position ), radius( radius ), material( material )
	{}
};


struct Ray
{
	Vec3 position;
	Vec3 direction;

	__host__ __device__
	Vec3 pointAt( float t ) const
	{
		return position + t * direction;
	}

	__host__ __device__
	Ray()
	{}

	__host__ __device__
	Ray( Vec3 position, Vec3 direction ) : position( position ),
		direction( direction )
	{}
};

struct Light
{
	Vec3 position;
	float intensity;

	__host__ __device__
	Light() {};

	__host__ __device__
	Light( Vec3 position, float intensity ) :
		position( position ), intensity( intensity )
	{}
};

struct Camera
{
	Vec3 position;
	Vec3 horizontal;
	Vec3 vertial;
	Vec3 upperRightCorner;

	__host__ __device__
	Camera()
	{};

	__host__ __device__
	Camera( Vec3 position, Vec3 lookAt, Vec3 up,
		float fieldOfView, float aspectRatio )
	{
		Vec3 u, v, w;
		float theta = fieldOfView * 3.14159f / 180.f;
		float half_height = tanf( theta / 2 );
		float half_width = aspectRatio * half_height;

		this->position = position;

		w = Vec3::unit( position - lookAt );
		u = Vec3::unit( Vec3::cross( up, w ) );
		v = Vec3::cross( w, u );

		upperRightCorner = position + u * half_width + v * half_height - w;
		horizontal = 2.f * half_width * u;
		vertial = 2.f * half_height * v;
	};

	__host__ __device__
	Ray emitRay( float x, float y ) const
	{
		Ray r;

		r.position = position;
		r.direction = Vec3::unit( upperRightCorner
			- x * horizontal - y * vertial - position );

		return r;
	}
};

struct Color
{
	uint8_t b, g, r;

	__host__ __device__
	Color() {}

	__host__ __device__
	Color( uint8_t r, uint8_t g, uint8_t b ) : r( r ), g( g ), b( b ) {}

	__host__ __device__
	Color( const Vec3& v ) :
		r( uint8_t( 255.9f * v.x ) ),
		g( uint8_t( 255.9f * v.y ) ),
		b( uint8_t( 255.9f * v.z ) )
	{}
};