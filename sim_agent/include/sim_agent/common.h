#pragma once

#include <stdint.h>

#include <algorithm>

namespace sim
{

struct Position
{
	int32_t x;
	int32_t y;
};

inline size_t encode_position(const Position& pos, uint32_t width)
{
	return pos.y * width + pos.x;
}

inline Position decode_position(size_t index, uint32_t width)
{
	return {
		static_cast<int32_t>(index) % static_cast<int32_t>(width),
		static_cast<int32_t>(index) / static_cast<int32_t>(width)};
}

inline Position operator+(const Position& a, const Position& b)
{
	return {a.x + b.x, a.y + b.y};
}

inline Position operator-(const Position& a, const Position& b)
{
	return {a.x - b.x, a.y - b.y};
}

inline bool operator==(const Position& a, const Position& b)
{
	return (a.x == b.x) && (a.y == b.y);
}

inline bool operator!=(const Position& a, const Position& b)
{
	return !(a == b);
}

template <template <typename> class C, typename T>
inline bool contains(const C<T>& container, const T& value)
{
	return std::find(container.begin(), container.end(), value) != container.end();
}

} // namespace sim
