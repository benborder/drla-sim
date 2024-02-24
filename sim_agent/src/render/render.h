#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace sim
{

enum class Shape
{
	kCircle,
	kRectangle,
	kTriangleUp,
	kTriangleDown,
	kTriangleLeft,
	kTriangleRight,
	kCross,
	kLineHorizontal,
	kLineVertical,
};

struct RenderShapeOptions
{
	// The RGB colour of the object
	cv::Scalar colour;
	// The line thickness in pixels. If a value <= 0, the thickness is automatically determined
	int line_thickness = -1;
	// If true the shape be filled, otherwise a line using line_thickness is drawn
	bool is_filled = false;
};

void draw_grid(cv::Mat& img, const cv::Size2i& num_tiles, const cv::Scalar& colour, int line_thickness);

void draw_shape(
	cv::Mat& img, Shape shape, const cv::Point& position, const cv::Size2i& size, const RenderShapeOptions& options);

}; // namespace sim
