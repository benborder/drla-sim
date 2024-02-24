#include "mcts.h"

using namespace sim;

SearchNode::SearchNode() : action_(-1), prior_(0)
{
}

SearchNode::SearchNode(int action, double prior) : action_(action), prior_(prior)
{
}

bool SearchNode::is_expanded() const
{
	return !child_nodes_.empty();
}

double SearchNode::get_value() const
{
	return visit_count_ > 0 ? value_sum_ / visit_count_ : 0;
}

inline void softmax(std::vector<double>& x)
{
	double sum = std::accumulate(x.begin(), x.end(), 0.0, [](double a, double b) { return a + std::exp(b); });
	for (double& v : x) { v = std::exp(v) / sum; }
}

void SearchNode::expand(
	const drla::ActionSet& legal_actions,
	int turn_index,
	double reward,
	std::vector<double> policy,
	std::unique_ptr<drla::Environment> env)
{
	turn_index_ = turn_index;
	reward_ = reward;
	value_sum_ = 0;
	env_ = std::move(env);
	std::vector<double> policy_values(legal_actions.size(), 0);
	if (!policy.empty())
	{
		for (size_t i = 0; i < legal_actions.size(); ++i) { policy_values[i] = policy[legal_actions[i]]; }
		softmax(policy_values);
	}
	child_nodes_.reserve(legal_actions.size());
	for (size_t i = 0; i < legal_actions.size(); ++i) { child_nodes_.emplace_back(legal_actions[i], policy_values[i]); }
}

drla::Environment* SearchNode::get_env() const
{
	return env_.get();
}

const std::vector<SearchNode>& SearchNode::get_children() const
{
	return child_nodes_;
}

int SearchNode::get_visit_count() const
{
	return visit_count_;
}

int SearchNode::get_action() const
{
	return action_;
}

void MinMaxStats::update(double value)
{
	if (value > max)
	{
		max = value;
	}
	if (value < min)
	{
		min = value;
	}
}

double MinMaxStats::normalise(double value)
{
	if (min >= max)
	{
		return value;
	}
	else
	{
		return (value - min) / (max - min);
	}
}

MCTS::MCTS(const MCTSConfig& config, const drla::ActionSet& action_set, int num_actors)
		: config_(config), num_actors_(num_actors), gen_(std::random_device{}())
{
}

MCTSResult MCTS::search(drla::Environment* env, const MCTSInput& input)
{
	if (input.legal_actions.empty())
	{
		throw std::runtime_error("Legal actions must be non zero");
	}

	MCTSResult res;

	res.root.expand(input.legal_actions, input.turn_index, 0, input.heuristic(env), env->clone());

	stats_ = {};

	for (int sim_count = 0; sim_count < config_.num_simulations; ++sim_count)
	{
		auto sim_turn_index = input.turn_index;
		auto* node = &res.root;
		std::vector<SearchNode*> search_path = {node};
		int current_tree_depth = 0;

		SearchNode* parent = nullptr;
		while (node != nullptr && node->is_expanded())
		{
			++current_tree_depth;
			parent = node;
			node = select_child(node);
			if (node != nullptr)
			{
				search_path.push_back(node);
				sim_turn_index = (sim_turn_index + 1) % num_actors_;
			}
		}

		if (node != nullptr)
		{
			auto next_env = parent->get_env()->clone();
			torch::Tensor action = torch::empty(1);
			action[0] = node->action_;
			auto step_data = next_env->step(action);
			auto policy = input.heuristic(next_env.get());
			node->expand(
				step_data.legal_actions,
				sim_turn_index,
				step_data.reward.sum().item<float>(),
				std::move(policy),
				std::move(next_env));
		}

		backpropagate(search_path, sim_turn_index);

		res.max_tree_depth = std::max(res.max_tree_depth, current_tree_depth);
	}

	return res;
}

SearchNode* MCTS::select_child(SearchNode* node)
{
	double max_ucb = -std::numeric_limits<double>::max();
	std::vector<double> ucb_vals;
	for (auto& child : node->child_nodes_)
	{
		double score = ucb_score(node, &child);
		if (score > max_ucb)
		{
			max_ucb = score;
		}
		ucb_vals.push_back(score);
	}

	std::vector<SearchNode*> nodes;
	for (size_t i = 0, ilen = node->child_nodes_.size(); i < ilen; ++i)
	{
		if (ucb_vals[i] == max_ucb)
		{
			nodes.push_back(&node->child_nodes_.at(i));
		}
	}

	if (nodes.empty())
	{
		return nullptr;
	}
	else if (nodes.size() == 1)
	{
		return nodes.back();
	}
	else
	{
		std::uniform_int_distribution<> dist(0, nodes.size() - 1);
		return nodes.at(dist(gen_));
	}
}

double MCTS::ucb_score(const SearchNode* parent, const SearchNode* child)
{
	double pb_c = std::log((parent->visit_count_ + config_.pb_c_base + 1) / config_.pb_c_base) + config_.pb_c_init;
	pb_c *= std::sqrt(parent->visit_count_) / (child->visit_count_ + 1);
	const double prior_score = pb_c * child->prior_;
	double value_score = 0;
	if (child->visit_count_ > 0)
	{
		const double value = num_actors_ == 1 ? child->get_value() : -child->get_value();
		value_score = stats_.normalise(child->reward_ + config_.gamma * value);
	}
	return value_score + prior_score;
}

void MCTS::backpropagate(const std::vector<SearchNode*>& search_path, int turn_index)
{
	double value = 0;
	for (auto node_iter = search_path.rbegin(); node_iter != search_path.rend(); ++node_iter)
	{
		SearchNode* node = *node_iter;
		bool same_index = node->turn_index_ == turn_index;
		node->value_sum_ += same_index ? value : -value;
		++node->visit_count_;
		const double node_value = (num_actors_ == 2 ? -node->get_value() : node->get_value());
		stats_.update(node->reward_ + config_.gamma * node_value);

		value = (same_index ? -node->reward_ : node->reward_) + config_.gamma * value;
	}
}
