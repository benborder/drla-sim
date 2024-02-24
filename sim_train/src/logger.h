#pragma once

#include "sim_agent/configuration.h"

#include <drla/auxiliary/metrics_logger.h>
#include <drla/callback.h>

#include <chrono>
#include <deque>
#include <filesystem>
#include <vector>

struct EpisodeResult
{
	std::string name;
	int id = 0;
	int env = 0;
	int length = 0;
	std::vector<int> life_length;
	std::vector<float> life_reward;

	// The reward for each actor (if only single actor env, then only one element in the vector is used)
	std::vector<torch::Tensor> reward;
	std::vector<torch::Tensor> score;
	std::deque<drla::StepData> step_data;

	// Indicates that this episode should be rendered
	bool render_final = false;
	bool render_gif = false;
	bool eval_episode = false;
};

class SimTrainingLogger : public drla::AgentCallbackInterface
{
public:
	SimTrainingLogger(sim::ConfigData config, const std::filesystem::path& path, bool resume);

private:
	void train_init(const drla::InitData& data) override;
	drla::AgentResetConfig env_reset(const drla::StepData& data) override;
	bool env_step(const drla::StepData& data) override;
	void train_update(const drla::TrainUpdateData& data) override;
	torch::Tensor interactive_step() override;

	void save(int steps, const std::filesystem::path& path) override;

	void save_episode_metrics(const EpisodeResult& episode);

	sim::ConfigData config_;
	std::filesystem::path buffer_path_;

	drla::TrainingMetricsLogger metrics_logger_;

	std::mutex m_step_;

	std::vector<EpisodeResult> current_episodes_;
	std::vector<EpisodeResult> episode_results_;

	int total_episode_count_ = 0;
	int total_game_count_ = 0;
	int next_gif_capture_ep_ = 0;
	int next_final_capture_ep_ = 0;
	int num_actors_ = 0;
	int actor_index_ = 0;
};
