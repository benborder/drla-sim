#pragma once

#include "configuration.h"
#include "configuration/cartpole.h"

#include <drla/environment.h>
#include <opencv2/core/mat.hpp>

#include <random>
#include <vector>

namespace sim
{

enum class CartPoleActions : int
{
	kLeft = 0,
	kRight = 1,
};

class CartPole final : public drla::Environment
{
public:
	CartPole(const Config::SimEnv& config);

	drla::EnvironmentConfiguration get_configuration() const override;

	drla::EnvStepData step(torch::Tensor action) override;
	drla::EnvStepData reset(const drla::State& initial_state) override;
	drla::Observations get_visualisations() override;

	torch::Tensor expert_agent() override;
	std::unique_ptr<drla::Environment> clone() const override;

private:
	torch::Tensor get_observation() const;

private:
	const Config::CartPoleEnv& config_;

	std::random_device rdev_;
	std::default_random_engine rndgen_;

	CartPoleState state_;
	int step_ = 0;
	int max_episode_steps_ = 0;
	bool episode_end_;

	const float polemass_length_ = 0;
	const float total_mass_ = 0;
	float steps_beyond_terminated_ = -1;

	cv::Mat visualisation_;
};

} // namespace sim
