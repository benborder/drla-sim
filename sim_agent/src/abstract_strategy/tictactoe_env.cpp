#include "tictactoe_env.h"

#include "mcts/mcts.h"
#include "render/render.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>

using namespace sim;

Tictactoe::Tictactoe(const Config::SimEnv& config)
		: config_(std::get<Config::TictactoeEnv>(config)), gen_(std::random_device{}())
{
	state_.board.fill(0);
	visualisation_ = cv::Mat::zeros(3 * config_.visualisation_size + 1, 3 * config_.visualisation_size + 1, CV_8UC3);
}

drla::EnvironmentConfiguration Tictactoe::get_configuration() const
{
	drla::EnvironmentConfiguration config;
	config.observation_shapes.push_back({{3, 3, 3}});
	config.observation_dtypes.push_back(torch::kFloat);
	config.action_space = {drla::ActionSpaceType::kDiscrete, {9}};
	config.action_set = {0, 1, 2, 3, 4, 5, 6, 7, 8};
	config.reward_types = {"winner"};
	config.num_actors = 2;
	return config;
}

drla::EnvStepData Tictactoe::step(torch::Tensor action)
{
	int a = action[0].item<int>();

	assert(state_.board[a] == 0);

	state_.board.at(a) = state_.player;

	bool winner = has_winner();

	episode_end_ = winner || legal_actions().empty();

	state_.player *= -1;

	auto reward = winner ? torch::ones({1}) : torch::zeros({1});
	return {
		{get_observation()},
		reward,
		{std::make_any<TictactoeState>(state_), step_, episode_end_, max_episode_steps_},
		legal_actions(),
		state_.player == 1 ? 0 : 1};
}

drla::EnvStepData Tictactoe::reset(const drla::State& initial_state)
{
	step_ = 0;
	episode_end_ = false;
	max_episode_steps_ = initial_state.max_episode_steps;

	const auto& state = std::any_cast<TictactoeState>(initial_state.env_state);
	state_ = state;

	return {
		{get_observation()},
		torch::zeros(1),
		{std::make_any<TictactoeState>(state_), step_, episode_end_},
		legal_actions(),
		state_.player == 1 ? 0 : 1};
}

drla::Observations Tictactoe::get_visualisations()
{
	visualisation_.setTo(cv::Scalar(0));
	cv::Size2i tile_size(config_.visualisation_size, config_.visualisation_size);
	RenderShapeOptions options{cv::Scalar(255, 255, 255)};
	for (size_t i = 0; i < state_.board.size(); ++i)
	{
		auto pos = decode_position(i, 3);
		cv::Point p(pos.x * tile_size.width, pos.y * tile_size.height);
		if (state_.board[i] == 1)
		{
			draw_shape(visualisation_, Shape::kCross, p, tile_size, options);
		}
		else if (state_.board[i] == -1)
		{
			draw_shape(visualisation_, Shape::kCircle, p, tile_size, options);
		}
	}

	draw_grid(visualisation_, {3, 3}, cv::Scalar(128, 128, 128), 1);

	return {torch::from_blob(
						visualisation_.data, {visualisation_.rows, visualisation_.cols, visualisation_.channels()}, torch::kByte)
						.clone()};
}

torch::Tensor Tictactoe::get_observation() const
{
	auto obs = torch::empty({3, 9});
	auto aobs = obs.accessor<float, 2>();
	for (size_t i = 0; i < state_.board.size(); ++i)
	{
		aobs[0][i] = state_.board[i] == 1 ? 1 : 0;
		aobs[1][i] = state_.board[i] == -1 ? 1 : 0;
		aobs[2][i] = state_.player;
	}
	return {obs.view({3, 3, 3})};
}

torch::Tensor Tictactoe::expert_agent()
{
	MCTSConfig config;
	config.num_simulations = 2000;
	MCTS mcts(config, {0, 1, 2, 3, 4, 5, 6, 7, 8}, 2);
	MCTSInput input;
	input.turn_index = state_.player == 1 ? 0 : 1;
	input.legal_actions = legal_actions();
	input.heuristic = [](Environment* env) { return static_cast<Tictactoe*>(env)->get_heuristic(); };
	auto result = mcts.search(this, input);

	std::vector<double> node_visits(9, 0);
	auto& nodes = result.root.get_children();
	assert(!nodes.empty());
	int max_visits = 0;
	size_t max_action = 0;
	for (size_t i = 0; i < nodes.size(); ++i)
	{
		auto index = input.legal_actions.at(i);
		auto visits = nodes[i].get_visit_count();
		if (visits > max_visits)
		{
			max_visits = visits;
			max_action = index;
		}
		node_visits.at(index) = static_cast<double>(visits);
	}

	double sum_count = 0;
	for (auto& node : node_visits)
	{
		node = std::pow(node, 1.0 / config_.expert_temperature);
		if (std::isinf(node))
		{
			node = std::numeric_limits<double>::max() / nodes.size();
		}
		sum_count += node;
	}
	for (auto& node : node_visits) { node /= sum_count; }

	size_t action_index;
	if (config_.expert_temperature == 0.0F)
	{
		action_index = max_action;
	}
	else if (config_.expert_temperature == std::numeric_limits<float>::infinity())
	{
		std::uniform_int_distribution<size_t> action_dist(0, nodes.size() - 1);
		action_index = action_dist(gen_);
	}
	else
	{
		std::discrete_distribution<size_t> action_dist(node_visits.begin(), node_visits.end());
		action_index = action_dist(gen_);
	}

	auto action = torch::empty(1, torch::kLong);
	action[0] = static_cast<int>(action_index);
	return action;
}

std::unique_ptr<drla::Environment> Tictactoe::clone() const
{
	return std::make_unique<Tictactoe>(*this);
}

drla::ActionSet Tictactoe::legal_actions() const
{
	drla::ActionSet actions;
	for (size_t i = 0; i < state_.board.size(); ++i)
	{
		if (state_.board[i] == 0)
		{
			actions.push_back(static_cast<int>(i));
		}
	}
	return actions;
}

bool Tictactoe::has_winner() const
{
	// Horizontal checks
	if (
		(state_.board[0] == state_.player && state_.board[1] == state_.player && state_.board[2] == state_.player) ||
		(state_.board[3] == state_.player && state_.board[4] == state_.player && state_.board[5] == state_.player) ||
		(state_.board[6] == state_.player && state_.board[7] == state_.player && state_.board[8] == state_.player))
	{
		return true;
	}
	// Vertical checks
	if (
		(state_.board[0] == state_.player && state_.board[3] == state_.player && state_.board[6] == state_.player) ||
		(state_.board[1] == state_.player && state_.board[4] == state_.player && state_.board[7] == state_.player) ||
		(state_.board[2] == state_.player && state_.board[5] == state_.player && state_.board[8] == state_.player))
	{
		return true;
	}
	// Diagonal checks
	if (
		(state_.board[0] == state_.player && state_.board[4] == state_.player && state_.board[8] == state_.player) ||
		(state_.board[2] == state_.player && state_.board[4] == state_.player && state_.board[6] == state_.player))
	{
		return true;
	}

	return false;
}

std::vector<double> Tictactoe::get_heuristic() const
{
	std::vector<double> policy(9, 1.0);

	return policy;
}
