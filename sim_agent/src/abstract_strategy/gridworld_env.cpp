#include "gridworld_env.h"

#include "mcts/mcts.h"
#include "render/render.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <map>

using namespace sim;

namespace
{
constexpr int kObservationDims = 3;
using ObsState = std::array<unsigned char, kObservationDims>;
static const std::array<ObsState, static_cast<uint32_t>(TileType::kWall) + 1> g_obs_tile_lookup = {
	ObsState{0, 0, 0},			 // kEmpty: Black
	ObsState{0, 255, 0},		 // kFloor: Green
	ObsState{0, 0, 255},		 // kSafe: Blue
	ObsState{0, 255, 128},	 // kHazard: Light Green
	ObsState{0, 128, 255},	 // kOpenDoor: Light Blue
	ObsState{128, 128, 0},	 // kClosedDoor: Dark Yellow
	ObsState{128, 0, 128},	 // kLockedDoor: Dark Magenta
	ObsState{128, 128, 128}, // kWall: Gray
};
// Used for visualisation
static const std::array<cv::Scalar, 2> g_player_colours = {
	cv::Scalar(255, 0, 0),
	cv::Scalar(0, 255, 0),
};

} // namespace

GridWorld::GridWorld(const Config::SimEnv& config)
		: config_(std::get<Config::GridWorldEnv>(config)), gen_(std::random_device{}())
{
	if (config_.action_set.empty())
	{
		spdlog::critical("No Actions defined");
		std::abort();
	}
	state_.grid.resize(config_.width * config_.height, {TileType::kEmpty});
	state_.player_state.resize(config_.players, {{0, 0}, Orientation::k0Deg, 0, {}});
	entered_tile_pos_ = std::numeric_limits<size_t>::max();
	exited_tile_pos_ = std::numeric_limits<size_t>::max();

	if (config_.relative)
	{
		uint32_t window = 2 * config_.window + 1;
		visualisation_ = cv::Mat::zeros(
			window * config_.visualisation_tile_size + 1, window * config_.visualisation_tile_size + 1, CV_8UC3);
	}
	else
	{
		visualisation_ = cv::Mat::zeros(
			config_.height * config_.visualisation_tile_size + 1,
			config_.width * config_.visualisation_tile_size + 1,
			CV_8UC3);
	}

	action_set_.reserve(config_.action_set.size());
	for (auto a : config_.action_set)
	{
		if (a <= Actions::kInvalid || a >= Actions::kActionCount)
		{
			spdlog::error("Invalid action defined");
			std::abort();
		}
		action_set_.emplace_back(static_cast<int32_t>(a));
	}
}

drla::EnvironmentConfiguration GridWorld::get_configuration() const
{
	drla::EnvironmentConfiguration config;
	if (config_.relative)
	{
		config.observation_shapes.push_back({{kObservationDims, (2 * config_.window + 1), (2 * config_.window + 1)}});
	}
	else
	{
		config.observation_shapes.push_back({{kObservationDims, config_.height, config_.width}});
	}
	config.observation_shapes.push_back({{1 + config_.players}});
	config.observation_dtypes = {torch::kByte, torch::kFloat};
	config.action_space = {drla::ActionSpaceType::kDiscrete, {static_cast<int64_t>(action_set_.size())}};
	for (size_t i = 0; i < action_set_.size(); ++i) { config.action_set.push_back(static_cast<int>(i)); }
	config.reward_types = {"winner"};
	config.num_actors = config_.players;
	return config;
}

drla::EnvStepData GridWorld::step(torch::Tensor action)
{
	Actions a = static_cast<Actions>(action_set_[action[0].item<int>()]);
	const PlayerState& current_player_state = state_.player_state.at(state_.player);
	auto& masked_actions = current_player_state.masked_actions;
	// Skip the action if it is masked
	if (std::none_of(masked_actions.begin(), masked_actions.end(), [a](Actions act) { return a == act; }))
	{
		switch (a)
		{
			case Actions::kMoveLeft: move_current_player(Position{-1, 0}); break;
			case Actions::kMoveRight: move_current_player(Position{1, 0}); break;
			case Actions::kMoveBackward: move_current_player(Position{0, -1}); break;
			case Actions::kMoveForward: move_current_player(Position{0, 1}); break;
			case Actions::kMoveLeftBackward: move_current_player(Position{-1, -1}); break;
			case Actions::kMoveLeftForward: move_current_player(Position{-1, 1}); break;
			case Actions::kMoveRightBackward: move_current_player(Position{1, -1}); break;
			case Actions::kMoveRightForward: move_current_player(Position{1, 1}); break;
			case Actions::kRotateClockwise: rotate_current_player(1); break;
			case Actions::kRotateCounterClockwise: rotate_current_player(-1); break;
			case Actions::kInvalid: // intentionally fallthrough
			default: spdlog::error("Invalid action: {}", static_cast<int>(a)); break;
		}
	}

	// Add global triggers
	std::copy(state_.global_triggers.begin(), state_.global_triggers.end(), std::back_inserter(trigger_queue_));

	// Process any triggers to update the grid state
	reward_ = 0.0;
	for (auto& trigger : trigger_queue_) { process_trigger(trigger); }
	trigger_queue_.clear();
	auto reward = torch::zeros({1});
	reward += reward_;

	auto acts = legal_actions();
	episode_end_ |= acts.empty();

	state_.player = (state_.player + 1) % config_.players;
	// Clear the intermediate temporary state so it does not trigger the in next step
	entered_tile_pos_ = std::numeric_limits<size_t>::max();
	exited_tile_pos_ = std::numeric_limits<size_t>::max();

	return {
		get_observations(),
		reward,
		{std::make_any<GridWorldState>(state_), step_, episode_end_, max_episode_steps_},
		legal_actions(), // The legal_actions() from above is not used as in multi player games it may be different
		state_.player};
}

void GridWorld::move_current_player(Position offset)
{
	PlayerState& current_player_state = state_.player_state[state_.player];
	const Position pos = current_player_state.position + offset;
	// Check this new position is valid
	if (
		pos.x < 0 || pos.x >= static_cast<int32_t>(config_.width) || pos.y < 0 ||
		pos.y >= static_cast<int32_t>(config_.height))
	{
		return;
	}
	const size_t p_prev = encode_position(current_player_state.position, config_.width);
	const size_t p = encode_position(pos, config_.width);
	TileState& tile_prev = state_.grid[p_prev];
	TileState& tile = state_.grid[p];
	if (
		tile.type != TileType::kWall && tile.type != TileType::kEmpty && tile.type != TileType::kClosedDoor &&
		tile.type != TileType::kLockedDoor && tile.players.size() < config_.max_players_per_tile)
	{
		tile.players.push_back(state_.player);
		state_.player_state[state_.player].position = pos;
		tile_prev.players.erase(std::find(tile_prev.players.begin(), tile_prev.players.end(), state_.player));
		std::copy(tile.triggers.begin(), tile.triggers.end(), std::back_inserter(trigger_queue_));
		entered_tile_pos_ = p;
		exited_tile_pos_ = p_prev;
	}
}

void GridWorld::rotate_current_player(int direction)
{
	PlayerState& current_player_state = state_.player_state[state_.player];
	int len = static_cast<int>(Orientation::k270Deg) + 1;
	current_player_state.orientation =
		static_cast<Orientation>((len + static_cast<int>(current_player_state.orientation) + direction) % len);
}

inline bool is_opponent(const std::vector<uint32_t>& players, uint32_t opponent)
{
	return std::any_of(players.begin(), players.end(), [opponent](uint32_t player) { return player != opponent; });
}

void GridWorld::process_trigger(const Trigger& trigger)
{
	PlayerState& current_player_state = state_.player_state[state_.player];
	const Position& pos = current_player_state.position;
	const size_t p = encode_position(pos, config_.width);
	TileState& tile = state_.grid[p];

	bool condition_matched = true;
	for (auto condition : trigger.conditions)
	{
		switch (condition)
		{
			using Type = TriggerConditionType;
			case Type::kPlayer1: condition_matched &= contains(tile.players, 0U); break;
			case Type::kPlayer2: condition_matched &= contains(tile.players, 1U); break;
			case Type::kOpponent: condition_matched &= is_opponent(tile.players, state_.player); break;
			case Type::kOnEnterTile: condition_matched &= entered_tile_pos_ == p; break;
			case Type::kOnLeaveTile: condition_matched &= exited_tile_pos_ == p; break;
			case Type::kFloor: condition_matched &= tile.type == TileType::kFloor; break;
			case Type::kWall: condition_matched &= tile.type == TileType::kWall; break;
			case Type::kOpenDoor: condition_matched &= tile.type == TileType::kClosedDoor; break;
			case Type::kClosedDoor: condition_matched &= tile.type == TileType::kClosedDoor; break;
			case Type::kLockedDoor: condition_matched &= tile.type == TileType::kLockedDoor; break;
			case Type::kHazard: condition_matched &= tile.type == TileType::kHazard; break;
			case Type::kSafe: condition_matched &= tile.type == TileType::kSafe; break;
		}
	}
	if (condition_matched)
	{
		switch (trigger.effect)
		{
			case TriggerEffectType::kReward: reward_ += std::get<float>(trigger.value); break;
			case TriggerEffectType::kTerminal: episode_end_ = true; break;
			case TriggerEffectType::kHP:
			{
				current_player_state.hp += std::get<float>(trigger.value);
				if (current_player_state.hp <= 0.0)
				{
					episode_end_ = true;
				}
				break;
			}
			case TriggerEffectType::kMaskAction:
			{
				current_player_state.masked_actions.push_back(std::get<Actions>(trigger.value));
				break;
			}
			case TriggerEffectType::kUnmaskAction:
			{
				auto unmask = std::get<Actions>(trigger.value);
				for (auto it = current_player_state.masked_actions.begin(); it != current_player_state.masked_actions.end();)
				{
					if (*it == unmask)
					{
						it = current_player_state.masked_actions.erase(it);
					}
					else
					{
						++it;
					}
				}
				break;
			}
		}
	}
}

static std::map<TileType, std::vector<TileType>> s_valid_gen_types{
	{TileType::kEmpty, {}},
	{TileType::kFloor, {TileType::kEmpty, TileType::kFloor}},
	{TileType::kWall, {TileType::kEmpty, TileType::kFloor}},
	{TileType::kHazard, {TileType::kFloor}},
	{TileType::kSafe, {TileType::kFloor}},
	{TileType::kOpenDoor, {TileType::kWall}},
	{TileType::kClosedDoor, {TileType::kWall}},
	{TileType::kLockedDoor, {TileType::kWall}}};

drla::EnvStepData GridWorld::reset(const drla::State& initial_state)
{
	step_ = 0;
	max_episode_steps_ = initial_state.max_episode_steps;
	episode_end_ = false;
	reward_ = 0.0;
	auto& state = std::any_cast<const GridWorldState>(initial_state.env_state);

	if (state.grid.size() != state_.grid.size())
	{
		spdlog::error(
			"Unable to initialise grid shape. Expected size of '{}', but got '{}'", state_.grid.size(), state.grid.size());
		std::abort();
	}

	state_ = state;
	state_.player = state.player;
	entered_tile_pos_ = std::numeric_limits<size_t>::max();
	exited_tile_pos_ = std::numeric_limits<size_t>::max();

	for (uint32_t player = 0; player < state_.player_state.size(); ++player)
	{
		auto player_state = state_.player_state[player];
		std::vector<size_t> valid_tiles;
		for (size_t p = 0; p < state_.grid.size(); ++p)
		{
			auto& tile = state_.grid[p];
			if (tile.type == TileType::kFloor)
			{
				valid_tiles.push_back(p);
			}
		}
		// randomly select from valid_tiles
		std::uniform_int_distribution<size_t> tile_dist(0, valid_tiles.size() - 1);
		size_t t = valid_tiles.at(tile_dist(gen_));
		player_state.position = decode_position(t, config_.width);
	}

	for (uint32_t player = 0; player < state_.player_state.size(); ++player)
	{
		const size_t p = encode_position(state_.player_state[player].position, config_.width);
		state_.grid[p].players.emplace_back(player);
	}

	for (auto& tile_gen : config_.grid_generator.tile_generators)
	{
		auto types = s_valid_gen_types.find(tile_gen.type);
		if (types == s_valid_gen_types.end())
		{
			continue;
		}
		std::vector<size_t> valid_tiles;
		for (size_t p = 0; p < state_.grid.size(); ++p)
		{
			auto& tile = state_.grid[p];
			auto pos = decode_position(p, config_.width);
			if (contains(types->second, tile.type))
			{
				switch (tile.type)
				{
					case TileType::kOpenDoor:
					case TileType::kClosedDoor:
					case TileType::kLockedDoor:
					{
						// check horizontal
						if (pos.x <= 0 || pos.x >= static_cast<int>(config_.width - 1))
						{
							continue;
						}
						// check vertical
						if (pos.y <= 0 || pos.y >= static_cast<int>(config_.height - 1))
						{
							continue;
						}
						break;
					}
					default: break;
				}
				valid_tiles.push_back(p);
			}
		}
		// randomly select from valid_tiles
		std::uniform_int_distribution<size_t> tile_dist(0, valid_tiles.size() - 1);
		size_t t = valid_tiles.at(tile_dist(gen_));
		state_.grid[t].type = tile_gen.type;
		state_.grid[t].triggers = tile_gen.triggers;
	}

	for (auto& object : config_.grid_generator.objects)
	{
		std::vector<size_t> valid_tiles;
		for (size_t p = 0; p < state_.grid.size(); ++p)
		{
			auto& tile = state_.grid[p];
			if (tile.type == TileType::kFloor)
			{
				valid_tiles.push_back(p);
			}
		}
		// randomly select from valid_tiles
		std::uniform_int_distribution<size_t> tile_dist(0, valid_tiles.size() - 1);
		size_t t = valid_tiles.at(tile_dist(gen_));
		state_.grid[t].objects.emplace_back(object);
	}

	auto acts = legal_actions();

	episode_end_ |= acts.empty();

	return {
		get_observations(),
		torch::zeros(1),
		{std::make_any<GridWorldState>(state_), step_, episode_end_},
		std::move(acts),
		state_.player};
}

void draw_tile(cv::Mat& mat, const TileType type, const cv::Size2i& tile_size, const cv::Point& p)
{
	RenderShapeOptions options;
	options.is_filled = true;
	switch (type)
	{
		case TileType::kEmpty:
		{
			break;
		}
		case TileType::kFloor:
		{
			options.colour = cv::Scalar(48, 48, 48);
			draw_shape(mat, Shape::kRectangle, p, tile_size, options);
			break;
		}
		case TileType::kWall:
		{
			options.colour = cv::Scalar(128, 128, 128);
			draw_shape(mat, Shape::kRectangle, p, tile_size, options);
			break;
		}
		case TileType::kOpenDoor:
		{
			options.colour = cv::Scalar(0, 128, 128);
			draw_shape(mat, Shape::kRectangle, p, tile_size, options);
			break;
		}
		case TileType::kClosedDoor:
		{
			options.colour = cv::Scalar(128, 64, 64);
			draw_shape(mat, Shape::kRectangle, p, tile_size, options);
			break;
		}
		case TileType::kLockedDoor:
		{
			options.colour = cv::Scalar(64, 128, 64);
			draw_shape(mat, Shape::kRectangle, p, tile_size, options);
			break;
		}
		case TileType::kHazard:
		{
			options.colour = cv::Scalar(128, 0, 0);
			draw_shape(mat, Shape::kRectangle, p, tile_size, options);
			break;
		}
		case TileType::kSafe:
		{
			options.colour = cv::Scalar(0, 0, 192);
			draw_shape(mat, Shape::kRectangle, p, tile_size, options);
			break;
		}
	}
}

void draw_players(
	cv::Mat& mat, const cv::Point& p, const cv::Size2i& tile_size, uint32_t player, const Orientation orientation)
{
	RenderShapeOptions options;
	options.is_filled = true;
	options.colour = g_player_colours[player];
	switch (orientation)
	{
		case Orientation::k0Deg:
		{
			draw_shape(mat, Shape::kTriangleRight, p, tile_size, options);
			break;
		}
		case Orientation::k90Deg:
		{
			draw_shape(mat, Shape::kTriangleUp, p, tile_size, options);
			break;
		}
		case Orientation::k180Deg:
		{
			draw_shape(mat, Shape::kTriangleLeft, p, tile_size, options);
			break;
		}
		case Orientation::k270Deg:
		{
			draw_shape(mat, Shape::kTriangleDown, p, tile_size, options);
			break;
		}
	}
}

drla::Observations GridWorld::get_visualisations()
{
	visualisation_.setTo(cv::Scalar(0));
	const cv::Size2i tile_size(config_.visualisation_tile_size, config_.visualisation_tile_size);

	if (config_.relative)
	{
		const auto& current_player_pos = state_.player_state[state_.player].position;

		const int32_t min_x = current_player_pos.x - config_.window;
		const int32_t max_x = current_player_pos.x + config_.window;
		const int32_t min_y = current_player_pos.y - config_.window;
		const int32_t max_y = current_player_pos.y + config_.window;
		const int32_t window_min_x = std::max<int32_t>(min_x, 0) - min_x;
		const int32_t window_max_x = std::min<int32_t>(max_x, config_.width - 1) - min_x;
		const int32_t window_min_y = std::max<int32_t>(min_y, 0) - min_y;
		const int32_t window_max_y = std::min<int32_t>(max_y, config_.height - 1) - min_y;

		// Draw the tiles
		for (int32_t y = window_min_y; y <= window_max_y; ++y)
		{
			const int32_t yp = min_y + y;
			for (int32_t x = window_min_x; x <= window_max_x; ++x)
			{
				const int32_t xp = min_x + x;
				size_t i = encode_position({xp, yp}, config_.width);
				const TileState& tile = state_.grid.at(i);
				cv::Point p(x * tile_size.width, y * tile_size.height);
				draw_tile(visualisation_, tile.type, tile_size, p);
			}
		}
		// Draw players on the tiles
		for (uint32_t player = 0; player < state_.player_state.size(); ++player)
		{
			const PlayerState& player_state = state_.player_state[player];
			auto rel_pos = player_state.position - Position{min_x, min_y};
			if (
				rel_pos.x >= 0 && rel_pos.x < static_cast<int32_t>(config_.width) && rel_pos.y >= 0 &&
				rel_pos.y < static_cast<int32_t>(config_.height))
			{
				cv::Point p(rel_pos.x * tile_size.width, rel_pos.y * tile_size.height);
				draw_players(visualisation_, p, tile_size, player, player_state.orientation);
			}
		}
		// Draw a grid on the floor tiles
		for (int32_t y = window_min_y; y <= window_max_y; ++y)
		{
			const int32_t yp = min_y + y;
			for (int32_t x = window_min_x; x <= window_max_x; ++x)
			{
				const int32_t xp = min_x + x;
				size_t i = encode_position({xp, yp}, config_.width);
				const TileState& tile = state_.grid.at(i);
				if (tile.type == TileType::kFloor)
				{
					cv::Point p(x * tile_size.width, y * tile_size.height);
					draw_shape(visualisation_, Shape::kRectangle, p, tile_size, {cv::Scalar(128, 128, 128), 1});
				}
			}
		}
	}
	else
	{
		// Draw the tiles
		for (size_t i = 0; i < state_.grid.size(); ++i)
		{
			const TileState& tile = state_.grid[i];
			auto pos = decode_position(i, config_.width);
			cv::Point p(pos.x * tile_size.width, pos.y * tile_size.height);

			draw_tile(visualisation_, tile.type, tile_size, p);
		}
		// Draw players on the tiles
		std::vector<size_t> tile_index;
		for (const PlayerState& player_state : state_.player_state)
		{
			size_t i = encode_position(player_state.position, config_.width);
			if (!contains(tile_index, i))
			{
				tile_index.push_back(i);
			}
		}
		for (size_t i : tile_index)
		{
			auto& tile = state_.grid[i];
			auto pos = decode_position(i, config_.width);
			cv::Point p(pos.x * tile_size.width, pos.y * tile_size.height);
			for (uint32_t player : tile.players)
			{
				draw_players(visualisation_, p, tile_size, player, state_.player_state.at(player).orientation);
			}
		}
		// Draw a grid on the floor tiles
		for (size_t i = 0; i < state_.grid.size(); ++i)
		{
			const TileState& tile = state_.grid[i];
			if (tile.type == TileType::kFloor)
			{
				auto pos = decode_position(i, config_.width);
				cv::Point p(pos.x * tile_size.width, pos.y * tile_size.height);
				draw_shape(visualisation_, Shape::kRectangle, p, tile_size, {cv::Scalar(128, 128, 128), 1});
			}
		}
	}

	return {torch::from_blob(
						visualisation_.data, {visualisation_.rows, visualisation_.cols, visualisation_.channels()}, torch::kByte)
						.clone()};
}

drla::Observations GridWorld::get_observations() const
{
	auto obs_state = torch::zeros({1 + config_.players});
	obs_state[0] = static_cast<float>(state_.player) / static_cast<float>(config_.players);
	for (int p = 0; p < config_.players; ++p) { obs_state[p + 1] = state_.player_state[p].hp; }
	return {get_grid_observation(), obs_state};
}

torch::Tensor GridWorld::get_grid_observation() const
{
	if (config_.relative)
	{
		using namespace torch::indexing;
		uint32_t window = (2 * config_.window + 1);
		auto obs = torch::zeros({kObservationDims, window, window}, torch::kByte);
		auto aobs = obs.accessor<unsigned char, 3>();
		const auto& pos = state_.player_state[state_.player].position;

		const int32_t min_x = pos.x - config_.window;
		const int32_t max_x = pos.x + config_.window;
		const int32_t min_y = pos.y - config_.window;
		const int32_t max_y = pos.y + config_.window;
		const int32_t window_min_x = std::max<int32_t>(min_x, 0) - min_x;
		const int32_t window_max_x = std::min<int32_t>(max_x, config_.width - 1) - min_x;
		const int32_t window_min_y = std::max<int32_t>(min_y, 0) - min_y;
		const int32_t window_max_y = std::min<int32_t>(max_y, config_.height - 1) - min_y;

		for (int32_t y = window_min_y; y <= window_max_y; ++y)
		{
			const int32_t yp = min_y + y;
			for (int32_t x = window_min_x; x <= window_max_x; ++x)
			{
				const int32_t xp = min_x + x;
				size_t p = encode_position({xp, yp}, config_.width);
				const TileState& tile = state_.grid.at(p);
				const auto& obs_state = g_obs_tile_lookup.at(static_cast<uint32_t>(tile.type));
				for (int o = 0; o < kObservationDims; ++o) { aobs[o][y][x] += obs_state[o]; }
				for (uint32_t player : tile.players) { aobs[0][y][x] += player == 0 ? 128U : 64U; }
			}
		}
		return obs;
	}
	else
	{
		auto obs = torch::zeros({kObservationDims, config_.height * config_.width}, torch::kByte);
		auto aobs = obs.accessor<unsigned char, 2>();
		for (size_t i = 0; i < state_.grid.size(); ++i)
		{
			const TileState& tile = state_.grid[i];
			const auto& obs_state = g_obs_tile_lookup.at(static_cast<uint32_t>(tile.type));
			for (int o = 0; o < kObservationDims; ++o) { aobs[o][i] += obs_state[o]; }
			for (uint32_t player : tile.players) { aobs[0][i] += player == 0 ? 128U : 64U; }
		}
		return obs.view({kObservationDims, config_.height, config_.width});
	}
}

torch::Tensor GridWorld::expert_agent()
{
	MCTSConfig config;
	config.num_simulations = 2000;
	MCTS mcts(config, action_set_, config_.players);
	MCTSInput input;
	input.turn_index = state_.player;
	input.legal_actions = legal_actions();
	input.heuristic = [](Environment* env) { return static_cast<GridWorld*>(env)->get_heuristic(); };
	auto result = mcts.search(this, input);

	std::vector<double> node_visits(action_set_.size(), 0);
	auto& nodes = result.root.get_children();
	assert(!nodes.empty());
	int max_visits = 0;
	int max_action = 0;
	for (const SearchNode& node : nodes)
	{
		int index = node.get_action();
		int visits = node.get_visit_count();
		if (visits > max_visits)
		{
			max_visits = visits;
			max_action = index;
		}
		node_visits.at(index) = static_cast<double>(visits);
	}

	double sum_count = 0;
	for (double& node : node_visits)
	{
		node = std::pow(node, 1.0 / config_.expert_temperature);
		if (std::isinf(node))
		{
			node = std::numeric_limits<double>::max() / nodes.size();
		}
		sum_count += node;
	}
	for (double& node : node_visits) { node /= sum_count; }

	int action_index;
	if (config_.expert_temperature == 0.0F)
	{
		action_index = max_action;
	}
	else if (config_.expert_temperature == std::numeric_limits<float>::infinity())
	{
		std::uniform_int_distribution<int> action_dist(0, nodes.size() - 1);
		action_index = action_dist(gen_);
	}
	else
	{
		std::discrete_distribution<int> action_dist(node_visits.begin(), node_visits.end());
		action_index = action_dist(gen_);
	}

	auto action = torch::empty(1, torch::kLong);
	action[0] = action_index;
	return action;
}

std::unique_ptr<drla::Environment> GridWorld::clone() const
{
	return std::make_unique<GridWorld>(*this);
}

drla::ActionSet GridWorld::legal_actions() const
{
	drla::ActionSet actions;
	if (episode_end_)
	{
		// There are no legal actions if the episode has ended
		return actions;
	}
	const PlayerState& current_player_state = state_.player_state[state_.player];
	auto add_valid_move = [&](int32_t action_index, const Position offset) {
		const Position pos = current_player_state.position + offset;
		// Check this position is valid
		if (
			pos.x < 0 || pos.x >= static_cast<int32_t>(config_.width) || pos.y < 0 ||
			pos.y >= static_cast<int32_t>(config_.height))
		{
			return;
		}
		size_t p = encode_position(pos, config_.width);
		const TileState& tile = state_.grid[p];
		if (
			tile.type != TileType::kWall && tile.type != TileType::kEmpty && tile.type != TileType::kClosedDoor &&
			tile.type != TileType::kLockedDoor)
		{
			actions.push_back(action_index);
		}
	};
	for (size_t i = 0; i < action_set_.size(); ++i)
	{
		auto action = static_cast<Actions>(action_set_[i]);
		if (std::any_of(
					current_player_state.masked_actions.begin(),
					current_player_state.masked_actions.end(),
					[action](Actions act) { return act == action; }))
		{
			// the action is not legal if it is masked
			continue;
		}
		switch (action)
		{
			case Actions::kMoveLeft: add_valid_move(static_cast<int>(i), Position{-1, 0}); break;
			case Actions::kMoveRight: add_valid_move(static_cast<int>(i), Position{1, 0}); break;
			case Actions::kMoveBackward: add_valid_move(static_cast<int>(i), Position{0, -1}); break;
			case Actions::kMoveForward: add_valid_move(static_cast<int>(i), Position{0, 1}); break;
			case Actions::kMoveLeftBackward: add_valid_move(static_cast<int>(i), Position{-1, -1}); break;
			case Actions::kMoveLeftForward: add_valid_move(static_cast<int>(i), Position{-1, 1}); break;
			case Actions::kMoveRightBackward: add_valid_move(static_cast<int>(i), Position{1, -1}); break;
			case Actions::kMoveRightForward: add_valid_move(static_cast<int>(i), Position{1, 1}); break;
			// Can always rotate
			case Actions::kRotateClockwise: actions.push_back(static_cast<int>(i)); break;
			case Actions::kRotateCounterClockwise: actions.push_back(static_cast<int>(i)); break;
			case Actions::kInvalid: // intentionally fallthrough
			default: spdlog::error("Invalid action: {}", static_cast<int>(action)); break;
		}
	}
	return actions;
}

std::vector<double> GridWorld::get_heuristic() const
{
	std::vector<double> policy(action_set_.size(), config_.expert_temperature);

	return policy;
}
