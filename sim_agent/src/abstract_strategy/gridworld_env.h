#pragma once

#include "configuration.h"
#include "configuration/gridworld.h"

#include <drla/environment.h>
#include <opencv2/core/mat.hpp>

#include <random>
#include <vector>

namespace sim
{

class GridWorld final : public drla::Environment
{
public:
	GridWorld(const Config::SimEnv& config);

	drla::EnvironmentConfiguration get_configuration() const override;

	drla::EnvStepData step(torch::Tensor action) override;
	drla::EnvStepData reset(const drla::State& initial_state) override;
	drla::Observations get_visualisations() override;

	torch::Tensor expert_agent() override;
	std::unique_ptr<drla::Environment> clone() const override;

private:
	void move_current_player(Position new_pos);
	void rotate_current_player(int direction);
	void process_trigger(const Trigger& trigger);
	drla::Observations get_observations() const;
	torch::Tensor get_grid_observation() const;
	drla::ActionSet legal_actions() const;
	std::vector<double> get_heuristic() const;

private:
	const Config::GridWorldEnv& config_;
	drla::ActionSet action_set_;
	std::mt19937 gen_;

	int step_ = 0;
	int max_episode_steps_ = 0;
	bool episode_end_;
	float reward_ = 0.0F;

	GridWorldState state_;
	std::vector<Trigger> trigger_queue_;
	size_t entered_tile_pos_;
	size_t exited_tile_pos_;

	cv::Mat visualisation_;
};

} // namespace sim
