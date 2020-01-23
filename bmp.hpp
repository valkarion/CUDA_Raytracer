#pragma once

#include <cstdint>
#include <string>
#include <fstream>

#pragma pack(push, 1)
struct BMPHeader
{
	/* BMP HEADER */
	uint16_t type = 0x4D42; // "BM"
	uint32_t size;
	uint16_t reserved1 = 0;
	uint16_t reserved2 = 0;
	uint32_t offsetBytes = sizeof( BMPHeader );

	/* DIB HEADER */
	uint32_t dibHeaderSize = 40;
	uint32_t width;
	uint32_t height;
	uint16_t planes = 1;
	uint16_t bitDepth = 24;
	uint32_t compression = 0; // BI_RGB
	uint32_t imageSize = 0; // ^ can be 0 because of this
	uint32_t horizontalRes = 0;
	uint32_t verticalRes = 0;
	uint32_t nColors = 0;
	uint32_t impColors = 0;
};
#pragma pack(pop) 

BMPHeader CreateBMPHeader( uint32_t width, uint32_t height )
{
	BMPHeader header = {};

	header.size = sizeof( BMPHeader ) + width * height * 3;

	header.width = width;
	header.height = height;

	return header;
}

void SaveBMP( const std::string& path, void* buffer,
	uint32_t width, uint32_t height )
{
	std::ofstream out( path, std::ios::binary );

	BMPHeader header = CreateBMPHeader( width, height );
	char headerBuffer[sizeof( BMPHeader )];
	std::memcpy( headerBuffer, &header, sizeof( BMPHeader ) );
	out.write( headerBuffer, sizeof( BMPHeader ) );

	// write the buffer upside down, good job BMP.
	char* start = (char*)buffer;
	size_t stride = width * 3;
	char* lastRow = start + ( ( height - 1 ) * stride );
	while ( lastRow >= start )
	{
		out.write( lastRow, stride );
		lastRow -= stride;
	}

	out.close();
}