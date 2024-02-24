#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sim
{

namespace Config
{

struct ConnectFourEnv
{
	// The number of connections to win
	int num_connect = 4;
	// The number of rows on the board
	int rows = 6;
	// The number of cols on the board
	int cols = 7;
	// The size in pixels to visualise each board position
	int visualisation_size = 16;
	// Determines how greedy the action selection is for the expert agent. 0 is maximally greedy and the larger the value
	// the more random.
	float expert_temperature = 0.0F;
};

} // namespace Config

struct ConnectFourState
{
	// The current player turn
	int player = 0;
	// The state of the board, player 1 is 1, player 2 is -1, 0 is unnocupied
	std::vector<int> board;
};

} // namespace sim
