#pragma once

#include "configuration.h"

#include <drla/auxiliary/serialise_json.h>
#include <drla/configuration.h>
#include <drla/types.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <iostream>

namespace nlohmann
{
template <>
struct adl_serializer<sim::TriggerValue>
{
	static void to_json(json& j, const sim::TriggerValue& data)
	{
		std::visit([&j](const auto& v) { j = v; }, data);
	}

	static void from_json(const json& j, sim::TriggerValue& data)
	{
		auto value = j.get<sim::Actions>();
		if (value != sim::Actions::kInvalid)
		{
			// Invalid action, so this is probably not an Actions type
			data = std::move(value);
			return;
		}
		else
		{
			data = j.get<float>();
		}
	}
};
} // namespace nlohmann

namespace sim
{

NLOHMANN_JSON_SERIALIZE_ENUM(
	SimEnvType,
	{
		{SimEnvType::kInvalid, "Invalid"},
		{SimEnvType::kCartPole, "CartPole"},
		{SimEnvType::kTictactoe, "Tictactoe"},
		{SimEnvType::kConnectFour, "ConnectFour"},
		{SimEnvType::kGridWorld, "GridWorld"},
	})

NLOHMANN_JSON_SERIALIZE_ENUM(
	Actions,
	{
		{Actions::kInvalid, "Invalid"},
		{Actions::kNone, "None"},
		{Actions::kMoveLeft, "MoveLeft"},
		{Actions::kMoveRight, "MoveRight"},
		{Actions::kMoveBackward, "MoveBackward"},
		{Actions::kMoveForward, "MoveForward"},
		{Actions::kMoveLeftBackward, "MoveLeftBackward"},
		{Actions::kMoveLeftForward, "MoveLeftForward"},
		{Actions::kMoveRightBackward, "MoveRightBackward"},
		{Actions::kMoveRightForward, "MoveRightForward"},
		{Actions::kRotateClockwise, "RotateClockwise"},
		{Actions::kRotateCounterClockwise, "RotateCounterClockwise"},
		{Actions::kInteract, "Interact"},
		{Actions::kPickupObject, "PickupObject"},
		{Actions::kActionCount, "ActionCount"},
	})

NLOHMANN_JSON_SERIALIZE_ENUM(
	TileType,
	{
		{TileType::kEmpty, "Empty"},
		{TileType::kFloor, "Floor"},
		{TileType::kWall, "Wall"},
		{TileType::kOpenDoor, "OpenDoor"},
		{TileType::kClosedDoor, "ClosedDoor"},
		{TileType::kLockedDoor, "LockedDoor"},
		{TileType::kHazard, "Hazard"},
		{TileType::kSafe, "Safe"},
	})

NLOHMANN_JSON_SERIALIZE_ENUM(
	TriggerConditionType,
	{
		{TriggerConditionType::kPlayer1, "Player1"},
		{TriggerConditionType::kPlayer2, "Player2"},
		{TriggerConditionType::kOpponent, "Opponent"},
		{TriggerConditionType::kFloor, "Floor"},
		{TriggerConditionType::kWall, "Wall"},
		{TriggerConditionType::kOpenDoor, "OpenDoor"},
		{TriggerConditionType::kClosedDoor, "ClosedDoor"},
		{TriggerConditionType::kLockedDoor, "LockedDoor"},
		{TriggerConditionType::kHazard, "Hazard"},
		{TriggerConditionType::kSafe, "Safe"},
	})

NLOHMANN_JSON_SERIALIZE_ENUM(
	TriggerEffectType,
	{
		{TriggerEffectType::kReward, "Reward"},
		{TriggerEffectType::kTerminal, "Terminal"},
		{TriggerEffectType::kHP, "HP"},
		{TriggerEffectType::kMaskAction, "MaskAction"},
		{TriggerEffectType::kUnmaskAction, "UnmaskAction"},
	})

NLOHMANN_JSON_SERIALIZE_ENUM(
	Orientation,
	{
		{Orientation::k0Deg, "0Deg"},
		{Orientation::k90Deg, "90Deg"},
		{Orientation::k180Deg, "180Deg"},
		{Orientation::k270Deg, "270Deg"},
	})

static inline void from_json(const nlohmann::json& json, Trigger& trigger)
{
	trigger.conditions << required_input{json, "conditions"};
	trigger.effect << required_input{json, "effect"};
	trigger.value << optional_input{json, "value"};
}

static inline void to_json(nlohmann::json& json, const Trigger& trigger)
{
	json["conditions"] = trigger.conditions;
	json["effect"] = trigger.effect;
	json["value"] = trigger.value;
}

namespace Config
{

static inline void from_json(const nlohmann::json& json, Config::CartPoleEnv& env)
{
	env.gravity << optional_input{json, "gravity"};
	env.masscart << optional_input{json, "masscart"};
	env.masspole << optional_input{json, "masspole"};
	env.length << optional_input{json, "length"};
	env.force_mag << optional_input{json, "force_mag"};
	env.tau << optional_input{json, "tau"};
	env.theta_threshold_radians << optional_input{json, "theta_threshold_radians"};
	env.x_threshold << optional_input{json, "x_threshold"};
	env.initial_range << optional_input{json, "initial_range"};
	env.kinematics_integrator_euler << optional_input{json, "kinematics_integrator_euler"};
	env.resolution << optional_input{json, "resolution"};
}

static inline void to_json(nlohmann::json& json, const Config::CartPoleEnv& env)
{
	json["gravity"] = env.gravity;
	json["masscart"] = env.masscart;
	json["masspole"] = env.masspole;
	json["length"] = env.length;
	json["force_mag"] = env.force_mag;
	json["tau"] = env.tau;
	json["theta_threshold_radians"] = env.theta_threshold_radians;
	json["x_threshold"] = env.x_threshold;
	json["initial_range"] = env.initial_range;
	json["kinematics_integrator_euler"] = env.kinematics_integrator_euler;
	json["resolution"] = env.resolution;
}

static inline void from_json(const nlohmann::json& json, Config::TictactoeEnv& env)
{
	env.visualisation_size << optional_input{json, "visualisation_size"};
	env.expert_temperature << optional_input{json, "expert_temperature"};
}

static inline void to_json(nlohmann::json& json, const Config::TictactoeEnv& env)
{
	json["visualisation_size"] = env.visualisation_size;
	json["expert_temperature"] = env.expert_temperature;
}

static inline void from_json(const nlohmann::json& json, Config::ConnectFourEnv& env)
{
	env.num_connect << optional_input{json, "num_connect"};
	env.rows << optional_input{json, "rows"};
	env.cols << optional_input{json, "cols"};
	env.visualisation_size << optional_input{json, "visualisation_size"};
	env.expert_temperature << optional_input{json, "expert_temperature"};
}

static inline void to_json(nlohmann::json& json, const Config::ConnectFourEnv& env)
{
	json["num_connect"] = env.num_connect;
	json["rows"] = env.rows;
	json["cols"] = env.cols;
	json["visualisation_size"] = env.visualisation_size;
	json["expert_temperature"] = env.expert_temperature;
}

static inline void from_json(const nlohmann::json& json, Config::TileGenerator& gen)
{
	gen.type << required_input{json, "type"};
	gen.triggers << optional_input{json, "triggers"};
}

static inline void to_json(nlohmann::json& json, const Config::TileGenerator& gen)
{
	json["type"] = gen.type;
	json["triggers"] = gen.triggers;
}

static inline void from_json(const nlohmann::json& json, Config::GridGenerator& gen)
{
	gen.tile_generators << optional_input{json, "tile_generators"};
	gen.objects << optional_input{json, "objects"};
	gen.players << optional_input{json, "players"};
}

static inline void to_json(nlohmann::json& json, const Config::GridGenerator& gen)
{
	json["tile_generators"] = gen.tile_generators;
	json["objects"] = gen.objects;
	json["players"] = gen.players;
}

static inline void from_json(const nlohmann::json& json, Config::GridWorldEnv& env)
{
	env.height << required_input{json, "height"};
	env.width << required_input{json, "width"};
	env.window << required_input{json, "window"};
	env.relative << required_input{json, "relative"};
	env.players << optional_input{json, "players"};
	env.action_set << required_input{json, "action_set"};
	env.expert_temperature << optional_input{json, "expert_temperature"};
	env.visualisation_tile_size << optional_input{json, "visualisation_tile_size"};
	env.max_players_per_tile << optional_input{json, "max_players_per_tile"};
	env.max_objects_per_tile << optional_input{json, "max_objects_per_tile"};
	env.grid_generator << optional_input{json, "grid_generator"};
}

static inline void to_json(nlohmann::json& json, const Config::GridWorldEnv& env)
{
	json["height"] = env.height;
	json["width"] = env.width;
	json["window"] = env.window;
	json["relative"] = env.relative;
	json["players"] = env.players;
	json["action_set"] = env.action_set;
	json["expert_temperature"] = env.expert_temperature;
	json["visualisation_tile_size"] = env.visualisation_tile_size;
	json["max_players_per_tile"] = env.max_players_per_tile;
	json["max_objects_per_tile"] = env.max_objects_per_tile;
	json["grid_generator"] = env.grid_generator;
}

static inline void to_json(nlohmann::json& json, const Config::SimEnv& env)
{
	std::visit([&](const auto& sim_env) { to_json(json, sim_env); }, env);
}

} // namespace Config

static inline void from_json(const nlohmann::json& json, ConfigData& config)
{
	config.type << required_input{json, "type"};
	switch (config.type)
	{
		case SimEnvType::kInvalid:
		{
			spdlog::error("Invalid Env type");
			break;
		}
		case SimEnvType::kCartPole:
		{
			Config::CartPoleEnv env;
			env << required_input{json, "environment"};
			config.env = std::move(env);
			break;
		}
		case SimEnvType::kTictactoe:
		{
			Config::TictactoeEnv env;
			env << required_input{json, "environment"};
			config.env = std::move(env);
			break;
		}
		case SimEnvType::kConnectFour:
		{
			Config::ConnectFourEnv env;
			env << required_input{json, "environment"};
			config.env = std::move(env);
			break;
		}
		case SimEnvType::kGridWorld:
		{
			Config::GridWorldEnv env;
			env << required_input{json, "environment"};
			config.env = std::move(env);
			break;
		}
	}
	config.initial_state_path << optional_input{json, "initial_state_path"};
	config.agent << required_input{json, "agent"};
	config.observation_save_period << optional_input{json, "observation_save_period"};
	config.observation_gif_save_period << optional_input{json, "observation_gif_save_period"};
	config.metric_image_log_period << optional_input{json, "metric_image_log_period"};
	config.gif_playback_speed << optional_input{json, "gif_playback_speed"};
}

static inline void to_json(nlohmann::json& json, const ConfigData& config)
{
	json["type"] = config.type;
	json["initial_state_path"] = config.initial_state_path;
	json["environment"] = config.env;
	json["agent"] = config.agent;
	json["observation_save_period"] = config.observation_save_period;
	json["observation_gif_save_period"] = config.observation_gif_save_period;
	json["metric_image_log_period"] = config.metric_image_log_period;
	json["gif_playback_speed"] = config.gif_playback_speed;
}

static inline void from_json(const nlohmann::json& json, CartPoleState& state)
{
}

static inline void to_json(nlohmann::json& json, const CartPoleState& state)
{
}

static inline void from_json(const nlohmann::json& json, TictactoeState& state)
{
	state.player << required_input{json, "player"};
	state.board << required_input{json, "board"};
}

static inline void to_json(nlohmann::json& json, const TictactoeState& state)
{
	json["player"] = state.player;
	json["board"] = state.board;
}

static inline void from_json(const nlohmann::json& json, ConnectFourState& state)
{
	state.player << required_input{json, "player"};
	state.board << required_input{json, "board"};
}

static inline void to_json(nlohmann::json& json, const ConnectFourState& state)
{
	json["player"] = state.player;
	json["board"] = state.board;
}

static inline void from_json(const nlohmann::json& json, Position& position)
{
	position.x << required_input{json, "x"};
	position.y << required_input{json, "y"};
}

static inline void to_json(nlohmann::json& json, const Position& position)
{
	json["x"] = position.x;
	json["y"] = position.y;
}

static inline void from_json(const nlohmann::json& json, TileState& tile_state)
{
	tile_state.type << required_input{json, "type"};
	tile_state.players << optional_input{json, "players"};
	tile_state.objects << optional_input{json, "objects"};
	tile_state.triggers << optional_input{json, "triggers"};
}

static inline void to_json(nlohmann::json& json, const TileState& tile_state)
{
	json["type"] = tile_state.type;
	json["players"] = tile_state.players;
	json["objects"] = tile_state.objects;
	json["triggers"] = tile_state.triggers;
}

static inline void from_json(const nlohmann::json& json, PlayerState& player_state)
{
	player_state.position << required_input{json, "position"};
	player_state.orientation << required_input{json, "orientation"};
	player_state.hp << optional_input{json, "hp"};
	player_state.inventory << optional_input{json, "inventory"};
	player_state.masked_actions << optional_input{json, "masked_actions"};
}

static inline void to_json(nlohmann::json& json, const PlayerState& player_state)
{
	json["position"] = player_state.position;
	json["orientation"] = player_state.orientation;
	json["hp"] = player_state.hp;
	json["inventory"] = player_state.inventory;
	json["masked_actions"] = player_state.masked_actions;
}

static inline void from_json(const nlohmann::json& json, GridWorldState& state)
{
	state.player << required_input{json, "player"};
	state.grid << required_input{json, "grid"};
	state.player_state << required_input{json, "player_state"};
	state.global_triggers << optional_input{json, "global_triggers"};
}

static inline void to_json(nlohmann::json& json, const GridWorldState& state)
{
	json["player"] = state.player;
	json["grid"] = state.grid;
	json["player_state"] = state.player_state;
	json["global_triggers"] = state.global_triggers;
}

} // namespace sim

namespace drla
{

static inline void from_json(const nlohmann::json& json, State& state)
{
	state.episode_end << optional_input{json, "episode_end"};
	state.max_episode_steps << optional_input{json, "max_episode_steps"};
	state.step << optional_input{json, "step"};
	sim::SimEnvType type;
	type << required_input{json, "type"};
	switch (type)
	{
		case sim::SimEnvType::kInvalid:
		{
			spdlog::error("Invalid env type: {}", json.find("type").value());
			throw std::runtime_error("Invalid env type");
		}
		case sim::SimEnvType::kCartPole:
		{
			sim::CartPoleState s;
			s << required_input{json, "env_state"};
			state.env_state = std::make_any<sim::CartPoleState>(std::move(s));
			break;
		}
		case sim::SimEnvType::kTictactoe:
		{
			sim::TictactoeState s;
			s << required_input{json, "env_state"};
			state.env_state = std::make_any<sim::TictactoeState>(std::move(s));
			break;
		}
		case sim::SimEnvType::kConnectFour:
		{
			sim::ConnectFourState s;
			s << required_input{json, "env_state"};
			state.env_state = std::make_any<sim::ConnectFourState>(std::move(s));
			break;
		}
		case sim::SimEnvType::kGridWorld:
		{
			sim::GridWorldState s;
			s << required_input{json, "env_state"};
			state.env_state = std::make_any<sim::GridWorldState>(std::move(s));
			break;
		}
	}
}

static inline void to_json(nlohmann::json& json, const State& state)
{
	json["episode_end"] = state.episode_end;
	json["max_episode_steps"] = state.max_episode_steps;
	json["step"] = state.step;
	if (state.env_state.type() == typeid(sim::CartPoleState))
	{
		json["type"] = sim::SimEnvType::kCartPole;
		json["env_state"] = std::any_cast<sim::CartPoleState>(state.env_state);
	}
	if (state.env_state.type() == typeid(sim::TictactoeState))
	{
		json["type"] = sim::SimEnvType::kTictactoe;
		json["env_state"] = std::any_cast<sim::TictactoeState>(state.env_state);
	}
	if (state.env_state.type() == typeid(sim::ConnectFourState))
	{
		json["type"] = sim::SimEnvType::kConnectFour;
		json["env_state"] = std::any_cast<sim::ConnectFourState>(state.env_state);
	}
	if (state.env_state.type() == typeid(sim::GridWorldState))
	{
		json["type"] = sim::SimEnvType::kGridWorld;
		json["env_state"] = std::any_cast<sim::GridWorldState>(state.env_state);
	}
}

} // namespace drla
