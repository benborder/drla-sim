#pragma once

#include "configuration/cartpole.h"
#include "configuration/connectfour.h"
#include "configuration/gridworld.h"
#include "configuration/tictactoe.h"

#include <drla/configuration.h>

#include <array>
#include <cstddef>
#include <string>
#include <variant>

namespace sim
{

enum class SimEnvType
{
	kInvalid,
	kCartPole,
	kTictactoe,
	kConnectFour,
	kGridWorld,
};

namespace Config
{

using SimEnv = std::variant<CartPoleEnv, TictactoeEnv, ConnectFourEnv, GridWorldEnv>;

} // namespace Config

struct ConfigData
{
	// The Sim environment type
	SimEnvType type;

	// The path to load initial state from when an env is reset. If a directory, then all files will be loaded and
	// randomly sampled from. Otherwise the specific file will be loaded. If empty, no state file is loaded.
	std::string initial_state_path;

	// The Sim environment configuration
	Config::SimEnv env;

	// Configuration specific to the agent
	drla::Config::Agent agent;

	// Every n train steps save the final frame of the next episode
	int observation_save_period = 1000;

	// Every n train steps save the entire next episode as a gif
	int observation_gif_save_period = 10000;

	// Every n train timesteps log any images from metrics
	int metric_image_log_period = 1000;

	// The frame time gifs are played back at (in ms)
	int gif_playback_speed = 50;
};

} // namespace sim
