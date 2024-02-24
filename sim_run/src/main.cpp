#include "runner.h"
#include "sim_agent/configuration.h"
#include "sim_agent/utility.h"

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <filesystem>

// BUG: https://github.com/pytorch/pytorch/issues/49460
// This dummy function is a hack to fix an issue with loading pytorch models. It's unnecessary to invoke this function,
// just enforce library compiled
void dummy()
{
	std::regex regstr("Why");
	std::string s = "Why crashed";
	std::regex_search(s, regstr);
}

int main(int argc, char** argv)
{
	cxxopts::Options options("Sim Run", "Runs an agent, optionally saving a gif");
	options.add_options()("p,data-path", "The data path to load the model from", cxxopts::value<std::string>())(
		"g,save-gif", "Saves a gif of the episode(s)", cxxopts::value<bool>()->default_value("false"))(
		"d,debug", "Enable debug logging", cxxopts::value<bool>()->default_value("false"))(
		"e,env-count", "Number of envs to run", cxxopts::value<int>()->default_value("1"))(
		"m,max-steps", "Maximum number of steps to run. 0 Implies infinite", cxxopts::value<int>()->default_value("0"))(
		"h,help", "This printout", cxxopts::value<bool>()->default_value("false"));
	options.allow_unrecognised_options();
	auto result = options.parse(argc, argv);

	if (result["help"].as<bool>())
	{
		options.set_width(100);
		spdlog::fmt_lib::print("{}", options.help());
		return 0;
	}

	std::filesystem::path data_path = result["data-path"].as<std::string>();
	bool save_gif = result["save-gif"].as<bool>();
	bool debug = result["debug"].as<bool>();
	int env_count = result["env-count"].as<int>();
	int max_steps = result["max-steps"].as<int>();

	spdlog::set_level(debug ? spdlog::level::debug : spdlog::level::info);
	spdlog::set_pattern("[%^%l%$] %v");

	auto config = sim::utility::load_config(data_path);

	SimRunner runner(config, data_path);

	runner.run(env_count, max_steps, save_gif);

	return 0;
}
