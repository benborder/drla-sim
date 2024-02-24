#include "sim_agent.h"

#include "cartpole_env.h"
#include "connectfour_env.h"
#include "gridworld_env.h"
#include "tictactoe_env.h"
#include "utility.h"

#include <iostream>

using namespace sim;

SimAgent::SimAgent(ConfigData&& config, drla::AgentCallbackInterface* callback, const std::filesystem::path& data_path)
		: config_(std::move(config))
		, base_path_(data_path)
		, agent_(drla::make_agent(config_.agent, this, callback, data_path))
		, gen_(std::random_device{}())
{
}

SimAgent::~SimAgent()
{
}

void SimAgent::train()
{
	agent_->train();
}

void SimAgent::stop_train()
{
	agent_->stop_train();
}

void SimAgent::run(int env_count, drla::RunOptions options)
{
	std::vector<drla::State> initial_states;
	initial_states.reserve(env_count);
	for (int i = 0; i < env_count; ++i)
	{
		auto state = get_initial_state();
		state.max_episode_steps = options.max_steps;
		initial_states.push_back(std::move(state));
	}
	agent_->run(initial_states, std::move(options));
}

std::unique_ptr<drla::Environment> SimAgent::make_environment()
{
	switch (config_.type)
	{
		case SimEnvType::kCartPole: return std::make_unique<CartPole>(config_.env);
		case SimEnvType::kTictactoe: return std::make_unique<Tictactoe>(config_.env);
		case SimEnvType::kConnectFour: return std::make_unique<ConnectFour>(config_.env);
		case SimEnvType::kGridWorld: return std::make_unique<GridWorld>(config_.env);
		case SimEnvType::kInvalid: throw std::runtime_error("Invalid env type");
	}
	return nullptr;
}

drla::State SimAgent::get_initial_state()
{
	if (!config_.initial_state_path.empty())
	{
		std::vector<std::filesystem::path> paths;
		std::filesystem::path path(config_.initial_state_path);
		if (path.is_relative())
		{
			path = base_path_ / path;
		}
		if (path.has_filename() && (path.extension() == ".json" || path.extension() == ".jsonc"))
		{
			paths.push_back(path);
		}
		else
		{
			for (const auto& entry : std::filesystem::directory_iterator(path))
			{
				auto state_path = entry.path();
				auto ext = state_path.extension().string();
				if (entry.is_regular_file() && (ext == ".json" || ext == ".jsonc"))
				{
					paths.push_back(state_path);
				}
			}
		}
		// Randomly sample state
		if (!paths.empty())
		{
			std::uniform_int_distribution<size_t> paths_dist(0, paths.size() - 1);
			size_t i = paths_dist(gen_);
			return utility::load_state(paths[i]);
		}
	}

	drla::State state;
	switch (config_.type)
	{
		case SimEnvType::kCartPole: state.env_state = std::make_any<CartPoleState>(); break;
		case SimEnvType::kTictactoe: state.env_state = std::make_any<TictactoeState>(); break;
		case SimEnvType::kConnectFour: state.env_state = std::make_any<ConnectFourState>(); break;
		case SimEnvType::kGridWorld: state.env_state = std::make_any<GridWorldState>(); break;
		case SimEnvType::kInvalid: throw std::runtime_error("Invalid env type");
	}
	return state;
}
