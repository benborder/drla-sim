#include "connectfour_env.h"

#include "mcts/mcts.h"
#include "render/render.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>

using namespace sim;

ConnectFour::ConnectFour(const Config::SimEnv& config)
		: config_(std::get<Config::ConnectFourEnv>(config)), gen_(std::random_device{}())
{
	state_.board.resize(config_.cols * config_.rows, 0);
	for (int i = 0; i < config_.cols; ++i) { action_set_.push_back(i); }
	visualisation_ = cv::Mat::zeros(
		config_.rows * config_.visualisation_size + 1, config_.cols * config_.visualisation_size + 1, CV_8UC3);
}

drla::EnvironmentConfiguration ConnectFour::get_configuration() const
{
	drla::EnvironmentConfiguration config;
	config.observation_shapes.push_back({{3, config_.rows, config_.cols}});
	config.observation_dtypes.push_back(torch::kInt);
	config.action_space = {drla::ActionSpaceType::kDiscrete, {config_.cols}};
	config.action_set = action_set_;
	config.reward_types = {"winner"};
	config.num_actors = 2;
	return config;
}

drla::EnvStepData ConnectFour::step(torch::Tensor action)
{
	int a = action[0].item<int>();
	assert(a >= 0 && a < config_.cols);

	int p = 0;
	for (int r = 0; r < config_.rows; ++r)
	{
		p = r * config_.cols + a;
		if (state_.board[p] == 0)
		{
			state_.board[p] = state_.player;
			break;
		}
	}

	bool winner = max_connected_count(p, state_.player) == config_.num_connect;

	episode_end_ = winner || legal_actions().empty();

	state_.player *= -1;

	auto reward = winner ? torch::ones({1}) : torch::zeros({1});
	return {
		{get_observation()},
		reward,
		{std::make_any<ConnectFourState>(state_), step_, episode_end_, max_episode_steps_},
		legal_actions(),
		state_.player == 1 ? 0 : 1};
}

drla::EnvStepData ConnectFour::reset(const drla::State& initial_state)
{
	step_ = 0;
	max_episode_steps_ = initial_state.max_episode_steps;
	episode_end_ = false;
	auto& state = std::any_cast<const ConnectFourState>(initial_state.env_state);

	if (state.board.size() == state_.board.size())
	{
		state_ = state;
	}
	else
	{
		if (!state.board.empty())
		{
			spdlog::warn("Initial board state does not match current board state. Default initialising board");
		}
		std::fill(state_.board.begin(), state_.board.end(), 0);
	}

	if (state.player != 0)
	{
		state_.player = state.player;
	}
	else
	{
		state_.player = 1;
	}

	auto acts = legal_actions();

	episode_end_ = acts.empty();
	for (size_t p = 0; p < state_.board.size(); ++p)
	{
		bool win = max_connected_count(p, state_.player) == config_.num_connect;
		bool lose = max_connected_count(p, -state_.player) == config_.num_connect;
		episode_end_ |= win || lose;
	}

	return {
		{get_observation()},
		torch::zeros(1),
		{std::make_any<ConnectFourState>(state_), step_, episode_end_},
		std::move(acts),
		0};
}

drla::Observations ConnectFour::get_visualisations()
{
	visualisation_.setTo(cv::Scalar(0));
	cv::Size2i tile_size(config_.visualisation_size, config_.visualisation_size);
	for (size_t i = 0; i < state_.board.size(); ++i)
	{
		auto pos = decode_position(i, config_.cols);
		cv::Point p(pos.x * tile_size.width, (config_.rows - pos.y - 1) * tile_size.height);
		if (state_.board[i] == 1)
		{
			draw_shape(visualisation_, Shape::kCircle, p, tile_size, {cv::Scalar(255, 0, 0), -1, true});
		}
		else if (state_.board[i] == -1)
		{
			draw_shape(visualisation_, Shape::kCircle, p, tile_size, {cv::Scalar(255, 255, 0), -1, true});
		}
	}

	draw_grid(
		visualisation_, {static_cast<int>(config_.cols), static_cast<int>(config_.rows)}, cv::Scalar(128, 128, 128), 1);

	return {torch::from_blob(
						visualisation_.data, {visualisation_.rows, visualisation_.cols, visualisation_.channels()}, torch::kByte)
						.clone()};
}

torch::Tensor ConnectFour::get_observation() const
{
	auto obs = torch::empty({3, config_.rows * config_.cols});
	auto aobs = obs.accessor<float, 2>();
	for (size_t i = 0; i < state_.board.size(); ++i)
	{
		aobs[0][i] = state_.board[i] == 1 ? 1 : 0;
		aobs[1][i] = state_.board[i] == -1 ? 1 : 0;
		aobs[2][i] = state_.player;
	}
	return {obs.view({3, config_.rows, config_.cols})};
}

torch::Tensor ConnectFour::expert_agent()
{
	MCTSConfig config;
	config.num_simulations = 2000;
	MCTS mcts(config, action_set_, 2);
	MCTSInput input;
	input.turn_index = state_.player == 1 ? 0 : 1;
	input.legal_actions = legal_actions();
	input.heuristic = [](Environment* env) { return static_cast<ConnectFour*>(env)->get_heuristic(); };
	auto result = mcts.search(this, input);

	std::vector<double> node_visits(config_.cols, 0);
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

std::unique_ptr<drla::Environment> ConnectFour::clone() const
{
	return std::make_unique<ConnectFour>(*this);
}

drla::ActionSet ConnectFour::legal_actions() const
{
	if (episode_end_)
	{
		return {};
	}
	const int offset = (config_.rows - 1) * config_.cols;
	drla::ActionSet actions;
	for (int i = 0; i < config_.cols; ++i)
	{
		if (state_.board[offset + i] == 0)
		{
			actions.push_back(i);
		}
	}
	return actions;
}

int ConnectFour::connected_line_count(int pos, int step, int player) const
{
	int count = 0;
	for (int j = 0; j < config_.num_connect; ++j)
	{
		int p = pos + j * step;
		if (state_.board[p] == player)
		{
			++count;
		}
		else if (state_.board[p] == -player)
		{
			break;
		}
	}
	return count;
}

int ConnectFour::max_connected_count(int pos, int player) const
{
	const int row = pos / config_.cols; // vert
	const int col = pos % config_.cols; // horiz
	const int xmax = config_.cols - (config_.num_connect - 1);
	const int ymax = config_.rows - (config_.num_connect - 1);
	int max = 0;

	for (int i = 0; i < config_.num_connect; ++i)
	{
		const int xvirt = col - i;
		const int yvirt = row - i;
		const int x2 = col + i;
		const int x = std::max(0, xvirt);
		const int y = std::max(0, yvirt);

		// Horizontal checks
		if (xvirt >= 0 && x < xmax)
		{
			int count = connected_line_count(row * config_.cols + x, 1, player);
			if (count > max)
			{
				max = count;
			}
		}
		// Vertical checks
		if (yvirt >= 0 && y < ymax)
		{
			int count = connected_line_count(y * config_.cols + col, config_.cols, player);
			if (count > max)
			{
				max = count;
			}
		}
		// Diagonal left checks
		if (xvirt >= 0 && yvirt >= 0 && x < xmax && y < ymax)
		{
			int count = connected_line_count(y * config_.cols + x, config_.cols + 1, player);
			if (count > max)
			{
				max = count;
			}
		}
		// Diagonal right checks
		if (x2 >= xmax && x2 < config_.cols && yvirt >= 0 && y < ymax)
		{
			int count = connected_line_count(y * config_.cols + x2, config_.cols - 1, player);
			if (count > max)
			{
				max = count;
			}
		}
	}

	return max;
}

std::vector<double> ConnectFour::get_heuristic() const
{
	std::vector<double> policy(config_.cols, 0);

	for (int c = 0; c < config_.cols; ++c)
	{
		int pos = -1;
		for (int r = config_.rows - 1; r >= 0; --r)
		{
			const int p = r * config_.cols + c;
			if (state_.board[p] == 0)
			{
				pos = p;
			}
		}
		if (pos >= 0)
		{
			const int p1 = max_connected_count(pos, state_.player);
			const int p2 = max_connected_count(pos, -state_.player);
			policy[c] = 0.1 + p2 + p1 * (p1 == (config_.num_connect - 1) ? 10 : p1 + 1);
		}
	}

	return policy;
}
