#pragma once

#include "configuration.h"
#include "configuration/tictactoe.h"

#include <drla/environment.h>
#include <opencv2/core/mat.hpp>

#include <random>
#include <vector>

namespace sim
{

class Tictactoe final : public drla::Environment
{
public:
	Tictactoe(const Config::SimEnv& config);

	drla::EnvironmentConfiguration get_configuration() const override;

	drla::EnvStepData step(torch::Tensor action) override;
	drla::EnvStepData reset(const drla::State& initial_state) override;
	drla::Observations get_visualisations() override;

	torch::Tensor expert_agent() override;
	std::unique_ptr<drla::Environment> clone() const override;

private:
	torch::Tensor get_observation() const;
	drla::ActionSet legal_actions() const;
	bool has_winner() const;
	std::vector<double> get_heuristic() const;

private:
	const Config::TictactoeEnv& config_;

	std::mt19937 gen_;

	int step_ = 0;
	int max_episode_steps_ = 0;
	bool episode_end_;

	TictactoeState state_;
	cv::Mat visualisation_;
};

} // namespace sim
