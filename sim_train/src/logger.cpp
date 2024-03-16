#include "logger.h"

#include "sim_agent/utility.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <string>

using namespace sim;
using namespace drla;

SimTrainingLogger::SimTrainingLogger(sim::ConfigData config, const std::filesystem::path& path, bool resume)
		: config_(config), metrics_logger_(path, resume)
{
	std::filesystem::path buffer_save_path = std::visit(
		[](auto& agent) {
			return std::visit([](auto& train_algorithm) { return train_algorithm.buffer_save_path; }, agent.train_algorithm);
		},
		config_.agent);
	if (!buffer_save_path.empty())
	{
		if (buffer_save_path.is_relative())
		{
			buffer_path_ = path / buffer_save_path;
		}
		else
		{
			buffer_path_ = buffer_save_path;
		}
		// Make sure the path exists
		std::filesystem::create_directory(buffer_path_);
	}
}

void SimTrainingLogger::train_init(const drla::InitData& data)
{
	int env_count = 0;
	int start_timestep = 0;
	int total_timesteps = 0;

	std::visit(
		[&](auto& agent) {
			env_count = agent.env_count;
			std::visit(
				[&](auto& train_algorithm) {
					total_timesteps = train_algorithm.total_timesteps;
					start_timestep = train_algorithm.start_timestep;
				},
				agent.train_algorithm);
		},
		config_.agent);

	num_actors_ = data.env_config.num_actors;

	fmt::print("{:=<80}\n", "");
	fmt::print("Training Sim Agent\n");
	fmt::print("Train timesteps: {}\n", total_timesteps);
	fmt::print("Envs: {}\n", env_count);
	if (num_actors_ > 1)
	{
		fmt::print("Actors: {}\n", num_actors_);
	}
	fmt::print("Start timestep: {}\n", start_timestep);
	fmt::print("{:=<80}\n", "");

	current_episodes_.resize(data.env_output.size());
	for (auto& ep : current_episodes_) { ep.id = total_game_count_++; }

	metrics_logger_.init(total_timesteps);
}

drla::AgentResetConfig SimTrainingLogger::env_reset(const drla::StepData& data)
{
	std::lock_guard lock(m_step_);
	EpisodeResult& episode_result = current_episodes_.at(data.env);
	episode_result.eval_episode = data.eval_mode;
	episode_result.name = data.name;
	return {false, episode_result.render_gif};
}

bool SimTrainingLogger::env_step(const drla::StepData& data)
{
	std::lock_guard lock(m_step_);
	EpisodeResult& episode_result = current_episodes_[data.env];

	if (episode_result.step_data.empty())
	{
		episode_result.reward.resize(num_actors_);
		episode_result.score.resize(num_actors_);
		for (int i = 0; i < num_actors_; ++i)
		{
			episode_result.reward[i] = torch::zeros(data.reward.sizes());
			episode_result.score[i] = torch::zeros(data.env_data.reward.sizes());
		}

		episode_result.reward.at(actor_index_) += data.reward;
		episode_result.score.at(actor_index_) += data.env_data.reward;
	}
	else
	{
		auto turn_index = episode_result.step_data.back().env_data.turn_index;
		episode_result.reward.at(turn_index) += data.reward;
		episode_result.score.at(turn_index) += data.env_data.reward;
	}

	episode_result.step_data.push_back(data);

	if (data.env_data.state.episode_end)
	{
		episode_result.env = data.env;
		episode_results_.push_back(std::move(episode_result));
		episode_result = {};
		episode_result.id = total_game_count_++;
		episode_result.render_final = total_episode_count_ == next_final_capture_ep_;
		episode_result.render_gif = total_episode_count_ == next_gif_capture_ep_;
		if (!data.eval_mode)
		{
			++total_episode_count_;
		}
	}
	else
	{
		if (!episode_result.render_gif && !episode_result.eval_episode && episode_result.step_data.size() > 1)
		{
			episode_result.step_data.pop_front();
		}
		episode_result.length++;
	}

	return false;
}

void SimTrainingLogger::train_update(const drla::TrainUpdateData& timestep_data)
{
	metrics_logger_.update(timestep_data);

	m_step_.lock();
	for (auto& episode_result : episode_results_)
	{
		if (episode_result.eval_episode)
		{
			metrics_logger_.add_scalar("environment/eval", "episode_length_eval", static_cast<double>(episode_result.length));

			if (num_actors_ > 1)
			{
				for (int i = 0; i < num_actors_; ++i)
				{
					const float episode_reward = episode_result.reward.at(i)[0].item<float>();
					metrics_logger_.add_scalar("environment/eval", "reward_actor_eval" + std::to_string(i), episode_reward);
				}
			}
			else
			{
				const float episode_reward = episode_result.reward.at(0)[0].item<float>();
				metrics_logger_.add_scalar("environment/eval", "reward_eval", episode_reward);
			}

			const float episode_score = episode_result.reward.at(actor_index_).item<float>();
			metrics_logger_.add_scalar("environment/eval", "score_eval", episode_score);
		}
		else
		{
			metrics_logger_.add_scalar("environment/train", "episode_length", static_cast<double>(episode_result.length));

			if (num_actors_ > 1)
			{
				for (int i = 0; i < num_actors_; ++i)
				{
					const float episode_reward = episode_result.reward.at(i)[0].item<float>();
					metrics_logger_.add_scalar("environment/train", "reward_actor" + std::to_string(i), episode_reward);
				}
			}
			else
			{
				const float episode_reward = episode_result.reward.at(0)[0].item<float>();
				metrics_logger_.add_scalar("environment/train", "reward", episode_reward);
			}

			const float episode_score = episode_result.reward.at(actor_index_).item<float>();
			metrics_logger_.add_scalar("environment/train", "score", episode_score);

			if (episode_result.render_final)
			{
				auto& final_obs = episode_result.step_data.back().env_data.observation.front();
				if (final_obs.dim() > 1)
				{
					metrics_logger_.add_image("observations", "final_frame", final_obs);
				}
			}
		}
		if (episode_result.render_gif || episode_result.eval_episode)
		{
			std::vector<torch::Tensor> images;
			images.reserve(episode_result.step_data.size());
			for (auto& step_data : episode_result.step_data) { images.push_back(step_data.visualisation.front().cpu()); }
			metrics_logger_.add_animation(
				"episode", episode_result.eval_episode ? "eval" : "train", images, config_.gif_playback_speed);
		}
		if (!episode_result.name.empty() && !buffer_path_.empty())
		{
			save_episode_metrics(episode_result);
		}
	}

	if (timestep_data.timestep % config_.observation_gif_save_period == 0)
	{
		next_gif_capture_ep_ = total_episode_count_ + 1;
	}
	if (timestep_data.timestep % config_.observation_save_period == 0)
	{
		next_final_capture_ep_ = total_episode_count_ + 1;
	}

	episode_results_.clear();
	m_step_.unlock();

	if (timestep_data.timestep >= 0 && ((timestep_data.timestep % config_.metric_image_log_period) == 0))
	{
		const auto& train_data = timestep_data.metrics.get_data();
		for (auto [name, data] : train_data)
		{
			metrics_logger_.add_animation("metrics", name, data.front(), config_.gif_playback_speed);
		}
	}

	metrics_logger_.print(timestep_data, total_episode_count_);
}

torch::Tensor SimTrainingLogger::interactive_step()
{
	return {};
}

void SimTrainingLogger::save(int steps, const std::filesystem::path& path)
{
	auto config = config_;
	std::visit(
		[&](auto& agent) {
			std::visit([&](auto& train_algorithm) { train_algorithm.start_timestep = steps; }, agent.train_algorithm);
		},
		config.agent);
	sim::utility::save_config(config, path);

	fmt::print("Configuration saved to: {}\n", path.string());
	fmt::print("{:-<80}\n", "");
}

void SimTrainingLogger::save_episode_metrics(const EpisodeResult& episode)
{
	auto path = buffer_path_ / ("episode_" + episode.name);
	std::filesystem::create_directory(path);
	nlohmann::json json;
	json["length"] = episode.length;
	{
		std::vector<float> rewards;
		for (auto& reward : episode.reward) { rewards.push_back(reward.sum().item<float>()); }
		json["reward"] = rewards;
	}
	{
		std::vector<float> scores;
		for (auto& score : episode.score) { scores.push_back(score.sum().item<float>()); }
		json["score"] = scores;
	}

	// Save to file
	std::ofstream metrics_file(path / "metrics.json");
	metrics_file << json.dump(2);
	metrics_file.close();
}
