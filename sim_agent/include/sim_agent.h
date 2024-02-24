#pragma once

#include "sim_agent/configuration.h"

#include <drla/agent.h>
#include <drla/auxiliary/env_manager.h>
#include <drla/callback.h>
#include <drla/environment.h>

#include <filesystem>
#include <memory>
#include <random>

namespace sim
{

class SimAgent final : public drla::GenericEnvironmentManager
{
public:
	SimAgent(ConfigData&& config, drla::AgentCallbackInterface* callback, const std::filesystem::path& data_path = "");
	~SimAgent();

	/// @brief Train the agent. Blocks until training finnished or stopped.
	void train();

	/// @brief Stop training the agent.
	void stop_train();

	/// @brief Run the agent, blocking until the max_steps reached or the environment terminates.
	/// @param env_count The number of environments to run
	/// @param options Options which change various behaviours of the agent. See RunOptions for more detail on available
	/// options.
	void run(int env_count, drla::RunOptions options = {});

protected:
	std::unique_ptr<drla::Environment> make_environment() override;
	drla::State get_initial_state() override;

private:
	const ConfigData config_;
	const std::filesystem::path base_path_;
	std::unique_ptr<drla::Agent> agent_;
	std::mt19937 gen_;
};

} // namespace sim
