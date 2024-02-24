#include "runner.h"

#include <drla/auxiliary/tensor_media.h>
#include <spdlog/fmt/chrono.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <string>

using namespace sim;
using namespace drla;

SimRunner::SimRunner(sim::ConfigData config, const std::filesystem::path& path)
		: config_(config), data_path_(path), sim_agent_(std::move(config), this, path)
{
}

void SimRunner::run(int env_count, int max_steps, bool save_gif)
{
	current_episodes_.resize(env_count);

	spdlog::info("Running {} environments\n", env_count);

	{
		drla::RunOptions options;
		options.enable_visualisations = save_gif;
		options.max_steps = max_steps;

		sim_agent_.run(env_count, options);
	}

	fmt::print("\n");
	spdlog::info("Complete!", env_count);

	for (auto& episode_result : episode_results_)
	{
		const float episode_reward = episode_result.reward[0].item<float>();
		spdlog::info("Episode length: {}", episode_result.length);
		spdlog::info("Episode Reward: {}", episode_reward);

		spdlog::info("Score: {}", episode_result.score.item<float>());

		if (save_gif && episode_result.step_data.size() > 2)
		{
			auto gif_path = data_path_ / fmt::format(
																		 "capture_{:%Y-%m-%d}_score_{}_ep{}.gif",
																		 fmt::localtime(std::time(nullptr)),
																		 episode_result.score.item<float>(),
																		 episode_result.id);
			std::vector<torch::Tensor> images;
			images.reserve(episode_result.step_data.size());
			for (auto& step_data : episode_result.step_data) { images.push_back(step_data.visualisation.front()); }
			save_gif_animation(gif_path, images, config_.gif_playback_speed);
		}
	}
}

void SimRunner::train_init(const drla::InitData& data)
{
}

drla::AgentResetConfig SimRunner::env_reset(const drla::StepData& data)
{
	std::lock_guard lock(m_step_);
	EpisodeResult& episode_result = current_episodes_[data.env];

	if (episode_result.length == 0)
	{
		// Clear the previous episode result for the env of this step data
		episode_result = {};
		episode_result.id = total_game_count_++;
		episode_result.reward = torch::zeros(data.reward.sizes());
		episode_result.score = torch::zeros(data.env_data.reward.sizes());
	}

	return {};
}

bool SimRunner::env_step(const drla::StepData& data)
{
	std::lock_guard lock(m_step_);
	fmt::print("\rstep: ");
	std::string state_str;
	if (data.env_data.state.env_state.type() == typeid(CartPoleState))
	{
		const auto& state = std::any_cast<const CartPoleState&>(data.env_data.state.env_state);
		state_str = fmt::format(
			"x: {: 0.4f}, x_dot: {: 0.4f}, theta: {: 0.4f}, theta_dot: {: 0.4f}",
			state.x,
			state.x_dot,
			state.theta,
			state.theta_dot);
	}
	for (auto& eps : current_episodes_) { fmt::print("{} {}", eps.length, state_str); }

	EpisodeResult& episode_result = current_episodes_[data.env];

	episode_result.length++;
	episode_result.reward += data.reward;
	episode_result.score += data.env_data.reward;
	episode_result.step_data.push_back(data);

	if (data.env_data.state.episode_end)
	{
		episode_result.env = data.env;
		episode_results_.push_back(std::move(episode_result));
		return true;
	}

	return false;
}

void SimRunner::train_update(const drla::TrainUpdateData& timestep_data)
{
}

torch::Tensor SimRunner::interactive_step()
{
	return {};
}

void SimRunner::save(int steps, const std::filesystem::path& path)
{
}
