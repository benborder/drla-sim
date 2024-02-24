#pragma once

#include <drla/environment.h>

#include <functional>
#include <random>
#include <vector>

namespace sim
{

struct StepOutput
{
	drla::Environment* env;
	std::vector<double> policy;
	int action;
	double reward;
	double value;
};

class SearchNode
{
	friend class MCTS;

public:
	SearchNode();
	SearchNode(int action, double prior);

	bool is_expanded() const;

	double get_value() const;

	void expand(
		const drla::ActionSet& legal_actions,
		int turn_index,
		double reward,
		std::vector<double> policy,
		std::unique_ptr<drla::Environment> env);

	drla::Environment* get_env() const;

	const std::vector<SearchNode>& get_children() const;

	int get_visit_count() const;

	int get_action() const;

private:
	std::vector<SearchNode> child_nodes_;
	double reward_;
	std::unique_ptr<drla::Environment> env_;
	double value_sum_;
	const int action_;
	double prior_;
	int turn_index_ = -1;
	int visit_count_ = 0;
};

struct MinMaxStats
{
	void update(double value);
	double normalise(double value);

	double max = -std::numeric_limits<double>::max();
	double min = std::numeric_limits<double>::max();
};

struct MCTSInput
{
	drla::ActionSet legal_actions;
	int turn_index;
	std::function<std::vector<double>(drla::Environment*)> heuristic;
};

struct MCTSResult
{
	SearchNode root;
	int max_tree_depth = 0;
	int root_predicted_value = 0;
};

/// @brief MCTS specific configuration
struct MCTSConfig
{
	// Number of future moves self-simulated
	int num_simulations = 10;
	// UCB formula c_base constant
	double pb_c_base = 19652;
	// UCB formula c_init constant
	double pb_c_init = 1.25;
	// The discount factor for value calculation
	double gamma = 1.0;
};

class MCTS
{
public:
	MCTS(const MCTSConfig& config, const drla::ActionSet& action_set, int num_actors);

	MCTSResult search(drla::Environment* env, const MCTSInput& input);

protected:
	SearchNode* select_child(SearchNode* node);
	double ucb_score(const SearchNode* parent, const SearchNode* child);
	void backpropagate(const std::vector<SearchNode*>& search_path, int turn_index);

private:
	const MCTSConfig config_;
	const int num_actors_;

	MinMaxStats stats_;

	std::mt19937 gen_;
};

} // namespace sim
