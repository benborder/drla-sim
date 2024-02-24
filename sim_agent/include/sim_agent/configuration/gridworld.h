#pragma once

#include "sim_agent/common.h"

#include <stdint.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <variant>
#include <vector>

namespace sim
{

enum class TileType : uint32_t
{
	kEmpty,	 // an invalid area not reachable by the agent
	kFloor,	 // A tile the agent can move on, and can contain objects
	kSafe,	 // A tile the agent can move onto or interacti with, but has positive consequences (gain hp, positive reward,
					 // invulnerable, terminate episode as success etc)
	kHazard, // A tile the agent can move onto or interact with, but has a negative consequence (lose hp, terminate
					 // episode as failure, negative reward etc)
	kOpenDoor,	 // A passable section in a wall, an agent can interact to close the door
	kClosedDoor, // An impassable section in a wall, the agent can interact to open the door
	kLockedDoor, // An impassable section in a wall, the agent must interact whilst in possession of a key object to
							 // unlock the door
	kWall,			 // A tile which the agent cannot pass through
};

enum class TriggerConditionType
{
	kPlayer1,			// Must be player1
	kPlayer2,			// Must be player2
	kOpponent,		// Must be the current players opponent
	kOnEnterTile, // When the player moves onto the tile
	kOnLeaveTile, // When the player moves off the tile
	kFloor,				// Must be a floor tile
	kWall,				// Must be a wall tile
	kOpenDoor,		// Must be an open door tile
	kClosedDoor,	// Must be an closed door tile
	kLockedDoor,	// Must be an locked door tile
	kHazard,			// Must be a hazard tile
	kSafe,				// Must be a safe tile
};

enum class TriggerEffectType
{
	kReward,			 // Gives a +ve or -ve reward
	kTerminal,		 // terminates the episode
	kHP,					 // Gain or lose hp,
	kMaskAction,	 // Prevents the agent from taking a specific action
	kUnmaskAction, // Removes an action from being masked
};

enum class Orientation
{
	k0Deg,
	k90Deg,
	k180Deg,
	k270Deg,
};

enum class Actions : int
{
	kInvalid = -1,
	kNone,
	kMoveLeft,
	kMoveRight,
	kMoveBackward,
	kMoveForward,
	kMoveLeftBackward,
	kMoveLeftForward,
	kMoveRightBackward,
	kMoveRightForward,
	kRotateClockwise,
	kRotateCounterClockwise,
	kInteract, // Opens/unlocks/closes door
	kPickupObject,
	kActionCount,
};

using TriggerValue = std::variant<float, Actions>;

/// @brief If the conditions are met, the defined effect is triggered.
struct Trigger
{
	// The list of conditions which must be met to trigger the effect
	std::vector<TriggerConditionType> conditions;
	// The list of effects for this trigger
	TriggerEffectType effect;
	// The value of an effect (i.e. reward, action or change in state)
	TriggerValue value = {};
};

struct TileState
{
	// The type of this tile
	TileType type;
	// The players occupying this tile.
	std::vector<uint32_t> players = {};
	// The objects occupying this tile.
	std::vector<uint32_t> objects = {};
	// Triggers an effect if the condition is met when an agent moves onto or interacts with this tile
	std::vector<Trigger> triggers = {};
};

struct PlayerState
{
	// The position of the player in the gridworld
	Position position;
	// The orientation of the player in the gridworld
	Orientation orientation;
	// The players HP, when reaches <= 0, the episode terminates with a negative reward
	float hp = 1.0F;
	// The players inventory by object id
	std::vector<uint32_t> inventory = {};
	// Actions the player is prevented from taking
	std::vector<Actions> masked_actions = {};
};

namespace Config
{

struct TileGenerator
{
	TileType type;
	// Triggers an effect if the condition is met when an agent moves onto or interacts with this tile
	std::vector<Trigger> triggers = {};
};

struct GridGenerator
{
	// A list of tiles to randomly place. Tiles have rules as to where they can be placed
	std::vector<TileGenerator> tile_generators;
	// The objects to randomly place
	std::vector<uint32_t> objects = {};
	// The players to randomly generate start positions for
	std::vector<uint32_t> players = {};
};

struct GridWorldEnv
{
	// The number of tiles in the vertical direction
	uint32_t height = 8;
	// The number of tiles in the horizontal direction
	uint32_t width = 8;
	// Observations are relative to the agents perspective if true. If false they are absolute to the env.
	bool relative = false;
	// The observation window size (a square, so 4 means a 4 tile radius, which is a window of 4*2+1 = 9, i.e. 9x9
	// window). The center tile is the agent.
	uint32_t window = 4;
	// The number of players (only supports up to 2 right now)
	int players = 1;
	// The maximum number of players allowed on a tile
	uint32_t max_players_per_tile = 1;
	// The maximum number of objects allowed on a tile
	uint32_t max_objects_per_tile = 1;
	// The actions available to agents
	std::vector<Actions> action_set;
	// Determines how greedy the action selection is for the expert agent. 0 is maximally greedy and the larger the value
	// the more random.
	float expert_temperature = 0.0F;
	// The size of each tile in pixels to visualise
	uint32_t visualisation_tile_size = 8;
	// Generates tiles in addition to the default loaded grid.
	GridGenerator grid_generator;
};

} // namespace Config

struct GridWorldState
{
	// The current player taking an action
	int player = 0;
	// The state of the grid world
	std::vector<TileState> grid;
	// The state of each player
	std::vector<PlayerState> player_state;
	// Global triggers are checked in the current players tile for each game step
	std::vector<Trigger> global_triggers = {};
};

} // namespace sim
