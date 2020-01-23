#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "logic.hpp"
#include "primitives.hpp"
#include "bmp.hpp"

#include <vector>
#include <chrono>
#include <iostream>

#define NOW std::chrono::high_resolution_clock::now()

#define WIDTH			1440
#define HEIGHT			900
#define FWIDTH			float(WIDTH)
#define FHEIGHT			float(HEIGHT)
#define ASPECT			FWIDTH / FHEIGHT
#define FOV				90.f
#define RAY_TTL			4
#define AA_RAYS			0

#define NSPHERES 		4
#define NLIGHTS			1

#define BLOCKDIMSIZE	24

Sphere spheres[NSPHERES];
Light lights[NLIGHTS];
Camera camera;

__device__ Sphere d_spheres[NSPHERES];
__device__ Light d_lights[NLIGHTS];
__device__ Camera d_camera;
__device__ Color d_colorBuffer[WIDTH * HEIGHT];

uint64_t runtimeCPU	= 0;
uint64_t runtimeCUDA = 0;
uint64_t runtimeCUDAFull = 0;

void SetupWorld()
{
	Material mirror;
	mirror.refractionIndex = 1.f;
	mirror.diffuseAttenuation = 0.5f;
	mirror.specularAttenuation = 10.0f;
	mirror.reflectionAttenuation = 0.8f;
	mirror.refractionAttenuation = 0.0;
	mirror.diffuseColor = Vec3( 1.f, 1.0f, 1.0f );
	mirror.specularExponent = 0.f;

	spheres[0] = Sphere( Vec3( 1.f, -1.f, -1.f ), 0.5f, mirror );
	spheres[1] = Sphere( Vec3( 1.f, 1.f, -1.f ), 0.5f, mirror );
	spheres[2] = Sphere( Vec3( -1.f, -1.f, -1.f ), 0.5f, mirror );
	spheres[3] = Sphere( Vec3( -1.f, 1.f, -1.f ), 0.5f, mirror );
	
	lights[0] = Light( Vec3( 0.f, 0.f, 0.f ), 1.f );

	camera = Camera( Vec3( 2.f, 2.f, 1.f ), Vec3( 0.f, 0.f, -1.f ),
		Vec3( 0.f, 1.f, 0.f ), FOV, ASPECT );
}

__device__
Ray CalcReflectedRay( const Ray& inbound, const HitRecord& rec )
{
	Vec3 reflectDir = Reflect( inbound.direction, rec.normal );
	Ray reflected = Ray( rec.point, reflectDir );

	return reflected;
}

__device__
Ray CalcRefractedRay( const Ray& inbound, const HitRecord& rec )
{
	Vec3 refractDir = Refract( inbound.direction, rec.normal,
		rec.sphere->material.refractionIndex );
	Ray refracted = Ray( rec.point, refractDir );

	return refracted;
}

__device__
void CalculateLight( const Ray& inbound, const HitRecord& rec, float& diffuse, float& specular)
{
	for ( int i = 0; i < NLIGHTS; ++i )
	{
		Vec3 surfaceNormal = Vec3::unit( rec.point - rec.sphere->position );
		Vec3 lightDirection = Vec3::unit( d_lights[i].position - rec.point );
		Ray pointToLightRay = Ray( rec.point, Vec3::unit( d_lights[i].position - rec.point ) );

		if ( FindClosestHitObject( d_spheres, NSPHERES,
			pointToLightRay, 0.001f, 1'000.f, {} ) )
		{
			continue;
		}

		diffuse += d_lights[i].intensity * Max( 0.f,
			Vec3::dot( lightDirection, surfaceNormal ) );

		if ( rec.sphere->material.specularExponent >= 1.f )
		{
			specular += std::powf( Max( 0.f,
				Vec3::dot( Reflect( lightDirection, surfaceNormal ), inbound.direction ) ),
				rec.sphere->material.specularExponent ) * d_lights[i].intensity;
		}
	}
}

__device__
Vec3 Cast( const Ray& ray, const int ttl )
{
	Vec3 unitDir = Vec3::unit( ray.direction );
	float t = 0.5*( unitDir.y + 1.f );
	Vec3 res = ( 1.f - t )*Vec3( 1.f, 1.f, 1.f ) + t * Vec3( 0.5f, 0.1f, 7.f );
	//Vec3 res( 0.f, 0.f, 0.f );

	if ( ttl <= 0 )
	{
		return res;
	}

	HitRecord rec;
	if ( !FindClosestHitObject( d_spheres, NSPHERES, ray, 0.001f,
		1000.0f, rec ) )
	{
		return res;
	}

	Vec3 reflectedColor = Cast( CalcReflectedRay( ray, rec ), ttl - 1 );		
	Vec3 refractedColor = Cast( CalcRefractedRay( ray, rec ), ttl - 1 );

	float diffuseIntensity = 0.f;
	float specularIntensity = 0.f;
	CalculateLight( ray, rec, diffuseIntensity, specularIntensity );

	res = rec.sphere->material.diffuseColor * diffuseIntensity * rec.sphere->material.diffuseAttenuation +
		Vec3( 1.f, 1.f, 1.f ) * specularIntensity * rec.sphere->material.specularAttenuation +
		reflectedColor * rec.sphere->material.reflectionAttenuation +
		refractedColor * rec.sphere->material.refractionAttenuation;

	return res;
}

__device__ 
float dev_random()
{
	return 0.f;
}

__global__ 
void TracerKernel()
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if ( idx >= WIDTH || idy >= HEIGHT )
	{
		return;
	}

	Vec3 colorBuffer = Vec3( 0.f, 0.f, 0.f );
	for ( size_t aa = 0; aa < AA_RAYS + 1; aa++ )
	{
		float u = ((float)idx + dev_random() )/ FWIDTH;
		float v = ((float)idy + dev_random() )/ FHEIGHT;

		Vec3 color = Cast( d_camera.emitRay( u, v ), RAY_TTL );

		// color correction
		float max = Max( color.x, Max( color.y, color.z ) );
		if ( max > 1.f )
		{
			color = color * ( 1.f / max );
		}

		colorBuffer += color;
	}
	
	Vec3 final = colorBuffer / ( AA_RAYS + 1 );
	final = Vec3( sqrtf( final.x ), sqrtf( final.y ), sqrtf( final.z ) );

	d_colorBuffer[idy * WIDTH + idx] = Color( final );
}

void RayTraceGPU()
{
	cudaError_t err;

	SetupWorld();

	auto start = NOW;

	err = cudaMemcpyToSymbol( d_spheres, spheres, NSPHERES * sizeof( Sphere ) );
	err = cudaMemcpyToSymbol( d_lights, lights, NLIGHTS * sizeof( Light ) );
	err = cudaMemcpyToSymbol( d_camera, &camera, sizeof( Camera ) );

	dim3 blockSize( BLOCKDIMSIZE, BLOCKDIMSIZE );
	dim3 nBlocks( ( WIDTH - 1 ) / BLOCKDIMSIZE + 1, ( HEIGHT - 1 ) / BLOCKDIMSIZE + 1 );

	auto kernelStart = NOW;

	TracerKernel << <nBlocks, blockSize >> > ( );
	cudaDeviceSynchronize();

	auto kernelStop = NOW;

	Color* colorBuffer = new Color[WIDTH * HEIGHT];

	err = cudaMemcpyFromSymbol( colorBuffer, d_colorBuffer, WIDTH * HEIGHT * sizeof( Color ) );

	auto fullStop = NOW;

	std::string errName = cudaGetErrorName( err );
	std::string errMsg = cudaGetErrorString( err );

	runtimeCUDA = std::chrono::duration_cast<std::chrono::microseconds>(kernelStop - kernelStart).count();
	runtimeCUDAFull = std::chrono::duration_cast<std::chrono::microseconds>( fullStop - start ).count();
	
	SaveBMP( "D:\\raytracer\\raytraceGPU.bmp", colorBuffer, WIDTH, HEIGHT );

	delete colorBuffer;
}

float random()
{
	return 2.f * ( (float)rand() / (float)RAND_MAX ) - 1.f;
}

Vec3 CastCPU( const Ray& ray, const int ttl )
{
	Vec3 unitDir = Vec3::unit( ray.direction );
	float t = 0.5*( unitDir.y + 1.f );
	Vec3 res = ( 1.f - t )*Vec3( 1.f, 1.f, 1.f ) + t * Vec3( 0.5f, 0.1f, 7.f );
	//Vec3 res( 0.f, 0.f, 0.f );

	if ( ttl <= 0 )
	{
		return res;
	}

	HitRecord rec;
	bool hitAnything = FindClosestHitObject( spheres, NSPHERES, ray, 0.001f,
		std::numeric_limits<float>::max(), rec );

	if ( !hitAnything )
	{
		return res;
	}

	Vec3 reflectDir = Reflect( ray.direction, rec.normal );
	Ray reflected = Ray( rec.point, reflectDir );
	Vec3 reflectedColor = CastCPU( reflected, ttl - 1 );

	Vec3 refractDir = Refract( ray.direction, rec.normal,
		rec.sphere->material.refractionIndex );
	Ray refracted = Ray( rec.point, refractDir );
	Vec3 refractedColor = CastCPU( refracted, ttl - 1 );

	Sphere& sphere = *rec.sphere;
	float diffuseIntensity = 0.f;
	float specularIntensity = 0.f;

	for ( size_t i = 0; i < NLIGHTS; i++ )
	{
		Vec3 intersectionPoint = rec.point;
		Vec3 surfaceNormal = Vec3::unit( intersectionPoint - sphere.position );
		Vec3 lightDirection = Vec3::unit( lights[i].position - intersectionPoint );

		bool isLightSourceObstructed;
		Ray pointToLightRay = Ray( rec.point, Vec3::unit( lights[i].position - rec.point ) );
		HitRecord temp;
		isLightSourceObstructed = FindClosestHitObject( spheres, NSPHERES,
			pointToLightRay, 0.001f, std::numeric_limits<float>::max(), temp );

		if ( isLightSourceObstructed )
		{
			continue;
		}

		diffuseIntensity += lights[i].intensity * std::max( 0.f,
			Vec3::dot( lightDirection, surfaceNormal ) );

		if ( sphere.material.specularExponent >= 1.f )
		{
			specularIntensity += std::powf( std::max( 0.f,
				Vec3::dot( Reflect( lightDirection, surfaceNormal ), ray.direction ) ),
				sphere.material.specularExponent ) * lights[i].intensity;
		}
	}

	res = sphere.material.diffuseColor * diffuseIntensity * sphere.material.diffuseAttenuation +
		Vec3( 1.f, 1.f, 1.f ) * specularIntensity * sphere.material.specularAttenuation +
		reflectedColor * sphere.material.reflectionAttenuation +
		refractedColor * sphere.material.refractionAttenuation;

	return res;
}

void RayTraceCPU()
{
	SetupWorld();

	std::vector<Vec3> frameBuffer;
	frameBuffer.resize( WIDTH * HEIGHT );

	auto start = NOW;

	for ( size_t y = 0; y < HEIGHT; y++ )
	{
		for ( size_t x = 0; x < WIDTH; x++ )
		{
			Vec3 color = Vec3( 0.f, 0.f, 0.f );
			for ( size_t aa = 0; aa < AA_RAYS + 1; aa++ )
			{
				float u = ( (float)x + random() ) / FWIDTH;
				float v = ( (float)y + random() ) / FHEIGHT;

				Vec3 pixelColor = CastCPU( camera.emitRay( u, v ), RAY_TTL );

				float max = std::max( pixelColor.x, std::max( pixelColor.y, pixelColor.z ) );
				if ( max > 1.f )
				{
					pixelColor = pixelColor * ( 1.f / max );
				}

				color += pixelColor;
			}

			Vec3 final = color / ( AA_RAYS + 1 );
			final = Vec3( sqrtf( final.x ), sqrtf( final.y ), sqrtf( final.z ) );

			frameBuffer[y * WIDTH + x] = final;
		}
	}

	auto delta = NOW - start;
	
	runtimeCPU = std::chrono::duration_cast<std::chrono::microseconds>( delta ).count();

	std::vector<Color> colorBuffer;
	for ( const auto& pixel : frameBuffer )
	{
		colorBuffer.emplace_back( pixel );
	}

	SaveBMP( "D:\\raytracer\\raytraceCPU.bmp", colorBuffer.data(), WIDTH, HEIGHT );


}

int main()
{	
	RayTraceGPU();

	RayTraceCPU();
	
	std::printf( "CPU time:\t%d\nKernel time:\t%d\nFull CUDA time:\t%d",
		runtimeCPU, runtimeCUDA, runtimeCUDAFull );

	std::cin.get();
	return 0;
}