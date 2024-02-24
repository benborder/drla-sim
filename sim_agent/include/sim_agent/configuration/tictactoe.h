#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace sim
{

namespace Config
{

struct TictactoeEnv
{
	// The size in pixels to visualise each board position
	int visualisation_size = 16;
	// Determines how greedy the action selection is for the expert agent. 0 is maximally greedy and the larger the value
	// the more random.
	float expert_temperature = 0.0F;
};

} // namespace Config

struct TictactoeState
{
	// The current player turn
	int player = 1;
	// The state of the board, player 1 is 1, player 2 is -1, 0 is unnocupied
	std::array<int, 9> board = {0, 0, 0, 0, 0, 0, 0, 0, 0};
};

} // namespace sim
