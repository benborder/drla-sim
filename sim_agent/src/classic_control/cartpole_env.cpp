#include "cartpole_env.h"

#include "render/render.h"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>

using namespace sim;

CartPole::CartPole(const Config::SimEnv& config)
		: config_(std::get<Config::CartPoleEnv>(config))
		, rndgen_(rdev_())
		, polemass_length_(config_.masspole * config_.length)
		, total_mass_(config_.masspole + config_.masscart)
{
	visualisation_ = cv::Mat::zeros(
		std::lroundf(2 * config_.resolution), std::lroundf(2 * config_.x_threshold * config_.resolution), CV_8UC3);
}

drla::EnvStepData CartPole::step(torch::Tensor action)
{
	auto a = CartPoleActions(action[0].item<int>());
	torch::Tensor reward = torch::ones(1);

	float force = 0;
	switch (a)
	{
		case CartPoleActions::kLeft:
		{
			force = config_.force_mag;
			break;
		}
		case CartPoleActions::kRight:
		{
			force = -config_.force_mag;
			break;
		}
		default:
		{
			throw std::runtime_error("Invalid action");
		}
	}

	float costheta = std::cos(state_.theta);
	float sintheta = std::sin(state_.theta);

	float temp = (force + polemass_length_ * state_.theta_dot * state_.theta_dot * sintheta) / total_mass_;
	float thetaacc = (config_.gravity * sintheta - costheta * temp) /
									 (config_.length * (4.0F / 3.0F - config_.masspole * costheta * costheta / total_mass_));
	float xacc = temp - polemass_length_ * thetaacc * costheta / total_mass_;

	if (config_.kinematics_integrator_euler)
	{
		state_.x = state_.x + config_.tau * state_.x_dot;
		state_.x_dot = state_.x_dot + config_.tau * xacc;
		state_.theta = state_.theta + config_.tau * state_.theta_dot;
		state_.theta_dot = state_.theta_dot + config_.tau * thetaacc;
	}
	else // semi-implicit euler
	{
		state_.x_dot = state_.x_dot + config_.tau * xacc;
		state_.x = state_.x + config_.tau * state_.x_dot;
		state_.theta_dot = state_.theta_dot + config_.tau * thetaacc;
		state_.theta = state_.theta + config_.tau * state_.theta_dot;
	}

	if (episode_end_)
	{
		reward[0] = 0.0f;
	}

	episode_end_ = state_.x < -config_.x_threshold || state_.x > config_.x_threshold ||
								 state_.theta < -config_.theta_threshold_radians || state_.theta > config_.theta_threshold_radians;

	return {
		{get_observation()},
		reward,
		{std::make_any<CartPoleState>(state_), step_, episode_end_, max_episode_steps_},
		{0, 1}};
}

// This is performed after a step but before the next step
drla::EnvStepData CartPole::reset(const drla::State& initial_state)
{
	step_ = 0;
	episode_end_ = false;
	max_episode_steps_ = initial_state.max_episode_steps;

	std::uniform_real_distribution<float> uniform(-config_.initial_range, config_.initial_range);

	state_.x = uniform(rndgen_);
	state_.x_dot = uniform(rndgen_);
	state_.theta_dot = uniform(rndgen_);
	state_.theta = uniform(rndgen_);

	return {{get_observation()}, torch::zeros(1), {std::make_any<CartPoleState>(state_), step_, episode_end_}, {0, 1}};
}

drla::Observations CartPole::get_visualisations()
{
	visualisation_.setTo(cv::Scalar(0));

	constexpr float kCartWidth = 0.3F;
	constexpr float kCartHeight = 0.2F;

	const float virt_x = config_.x_threshold + state_.x;

	// Draw horizontal line in the middle of the screen
	draw_shape(
		visualisation_,
		Shape::kLineHorizontal,
		cv::Point(0, std::lroundf(2 * kCartHeight * config_.resolution)),
		{visualisation_.cols, visualisation_.rows},
		{cv::Scalar(96, 96, 96), 2});

	cv::Point cart_com(
		std::lroundf(virt_x * config_.resolution),
		std::lroundf(visualisation_.rows / 2.0F + 1.5F * kCartHeight * config_.resolution));
	cv::Point cart_tl(
		std::lroundf((virt_x - kCartWidth / 2.0F) * config_.resolution),
		std::lroundf(visualisation_.rows / 2.0F + kCartHeight * config_.resolution));

	// Draw cart as a box
	cv::Size2i box_size(std::lroundf(kCartWidth * config_.resolution), std::lroundf(kCartHeight * config_.resolution));
	draw_shape(visualisation_, Shape::kRectangle, cart_tl, box_size, RenderShapeOptions{cv::Scalar(0, 192, 0), 2, true});

	// Draw pole
	const float virt_length = config_.length * config_.resolution;
	cv::Point pole_tip(
		cart_com.x + std::sin(state_.theta) * virt_length, cart_com.y - std::cos(state_.theta) * virt_length);
	cv::line(visualisation_, cart_com, pole_tip, cv::Scalar(255, 255, 255), 2);

	return {torch::from_blob(
						visualisation_.data, {visualisation_.rows, visualisation_.cols, visualisation_.channels()}, torch::kByte)
						.clone()};
}

drla::EnvironmentConfiguration CartPole::get_configuration() const
{
	drla::EnvironmentConfiguration config;
	config.observation_shapes.push_back({{4}});
	config.observation_dtypes.push_back(torch::kFloat);
	config.action_space = {drla::ActionSpaceType::kDiscrete, {2}};
	config.action_set = {0, 1};
	config.reward_types = {"score"};
	config.num_actors = 1;
	return config;
}

torch::Tensor CartPole::expert_agent()
{
	return {};
}

std::unique_ptr<drla::Environment> CartPole::clone() const
{
	return nullptr;
}

torch::Tensor CartPole::get_observation() const
{
	torch::Tensor obs = torch::empty({4});
	obs[0] = state_.x / (2 * config_.x_threshold);
	obs[1] = state_.x_dot;
	obs[2] = state_.theta / (2 * config_.theta_threshold_radians);
	obs[3] = state_.theta_dot;
	return obs;
}
