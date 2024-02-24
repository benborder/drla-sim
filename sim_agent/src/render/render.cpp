#include "render.h"

#include <opencv2/opencv.hpp>

using namespace sim;

void sim::draw_grid(cv::Mat& img, const cv::Size2i& num_tiles, const cv::Scalar& colour, int line_thickness)
{
	// Assume tiles match grid evenly
	cv::Size2i tile_size(img.cols / num_tiles.width, img.rows / num_tiles.height);
	for (int i = 0; i <= num_tiles.width; ++i)
	{
		int x = tile_size.width * i;
		cv::line(img, cv::Point(x, 0), cv::Point(x, img.rows - 1), colour, line_thickness);
	}
	for (int j = 0; j <= num_tiles.height; ++j)
	{
		int y = tile_size.height * j;
		cv::line(img, cv::Point(0, y), cv::Point(img.cols - 1, y), colour, line_thickness);
	}
}

void sim::draw_shape(
	cv::Mat& img, Shape shape, const cv::Point& position, const cv::Size2i& size, const RenderShapeOptions& options)
{
	cv::Point p1 = position;
	cv::Point p2 = position + cv::Point(size);
	cv::Point p_mid{position.x + size.width / 2, position.y + size.height / 2};

	int line_thickness = options.line_thickness;
	if (line_thickness < 0)
	{
		line_thickness = std::max(1, std::min(size.width, size.height) / 6);
		// Add padding to tile if drawing lines
		if (!options.is_filled)
		{
			p1.x += (line_thickness + 1) / 2;
			p1.y += (line_thickness + 1) / 2;
			p2.x -= (line_thickness + 1) / 2;
			p2.y -= (line_thickness + 1) / 2;
		}
	}

	switch (shape)
	{
		case Shape::kCircle:
		{
			int radius = (std::min(size.width, size.height) - (options.is_filled ? 0 : (line_thickness + 1))) / 2;
			cv::circle(img, p_mid, radius, options.colour, options.is_filled ? cv::FILLED : line_thickness);
			break;
		}
		case Shape::kTriangleUp:
		{
			std::array<cv::Point, 3> vertices = {cv::Point(p_mid.x, p1.y), p2, cv::Point(p1.x, p2.y)};
			if (options.is_filled)
			{
				cv::fillPoly(img, vertices, options.colour);
			}
			else
			{
				cv::polylines(img, vertices, true, options.colour, line_thickness);
			}
			break;
		}
		case Shape::kTriangleDown:
		{
			std::array<cv::Point, 3> vertices = {p1, cv::Point(p2.x, p1.y), cv::Point(p_mid.x, p2.y)};
			if (options.is_filled)
			{
				cv::fillPoly(img, vertices, options.colour);
			}
			else
			{
				cv::polylines(img, vertices, true, options.colour, line_thickness);
			}
			break;
		}
		case Shape::kTriangleLeft:
		{
			std::array<cv::Point, 3> vertices = {cv::Point(p2.x, p1.y), p2, cv::Point(p1.x, p_mid.y)};
			if (options.is_filled)
			{
				cv::fillPoly(img, vertices, options.colour);
			}
			else
			{
				cv::polylines(img, vertices, true, options.colour, line_thickness);
			}
			break;
		}
		case Shape::kTriangleRight:
		{
			std::array<cv::Point, 3> vertices = {p1, cv::Point(p2.x, p_mid.y), cv::Point(p1.x, p2.y)};
			if (options.is_filled)
			{
				cv::fillPoly(img, vertices, options.colour);
			}
			else
			{
				cv::polylines(img, vertices, true, options.colour, line_thickness);
			}
			break;
		}
		case Shape::kRectangle:
		{
			cv::rectangle(img, p1, p2, options.colour, options.is_filled ? cv::FILLED : line_thickness);
			break;
		}
		case Shape::kCross:
		{
			cv::line(img, p1, p2, options.colour, line_thickness);
			cv::line(img, cv::Point(p2.x, p1.y), cv::Point(p1.x, p2.y), options.colour, line_thickness);
			break;
		}
		case Shape::kLineHorizontal:
		{
			cv::line(img, cv::Point(p1.x, p_mid.y), cv::Point(p2.x, p_mid.y), options.colour, line_thickness);
			break;
		}
		case Shape::kLineVertical:
		{
			cv::line(img, cv::Point(p_mid.x, p1.y), cv::Point(p_mid.x, p2.y), options.colour, line_thickness);
			break;
		}
	}
}
