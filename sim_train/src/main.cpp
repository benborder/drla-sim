#include "logger.h"
#include "sim_agent.h"
#include "sim_agent/configuration.h"
#include "sim_agent/utility.h"

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <csignal>
#include <cstdio>
#include <filesystem>
#include <functional>

namespace
{
std::function<void(int)> shutdown_handler;

void signal_handler(int signum)
{
	shutdown_handler(signum);
}
} // namespace

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
	cxxopts::Options options(
		"DRLA Simulator Trainer", "Trains an agent, periodically saving the model and a tensorboard event file.");
	options.add_options()(
		"c,config",
		"The config directory path or full file path. Relative paths use the data path as the base.",
		cxxopts::value<std::string>()->default_value(""))(
		"d,data", "The data path for saving/loading the model and training state", cxxopts::value<std::string>())(
		"h,help", "This printout", cxxopts::value<bool>()->default_value("false"));
	options.allow_unrecognised_options();
	auto result = options.parse(argc, argv);

	if (result["help"].as<bool>())
	{
		options.set_width(100);
		spdlog::fmt_lib::print("{}", options.help());
		return 0;
	}

	std::filesystem::path config_path = result["config"].as<std::string>();
	std::filesystem::path data_path = result["data"].as<std::string>();

	bool resume = true;
	if (config_path.empty())
	{
		config_path = data_path;
	}
	else if (config_path != data_path)
	{
		std::filesystem::create_directory(data_path);
		data_path = data_path / sim::utility::get_time();
		std::filesystem::create_directory(data_path);
		resume = false;
	}

	spdlog::set_level(spdlog::level::debug);
	spdlog::set_pattern("[%^%l%$] %v");

	auto config = sim::utility::load_config(config_path);

	if (!config.initial_state_path.empty())
	{
		// Save initial states to data path
		std::filesystem::path load_path(config.initial_state_path);
		std::filesystem::path save_path(config.initial_state_path);
		if (load_path.is_relative())
		{
			load_path = config_path.parent_path() / load_path;
			save_path = data_path / save_path;
		}
		std::filesystem::create_directory(save_path.parent_path());
		// If the initial_state_path is a specific file then
		if (load_path.has_filename() && (load_path.extension() == ".json" || load_path.extension() == ".jsonc"))
		{
			spdlog::debug("Found initial state: {}", load_path.filename().string());
			sim::utility::save_state(sim::utility::load_state(load_path), save_path);
		}
		else
		{
			for (const auto& entry : std::filesystem::directory_iterator(load_path))
			{
				auto state_path = entry.path();
				auto ext = state_path.extension().string();
				if (entry.is_regular_file() && (ext == ".json" || ext == ".jsonc"))
				{
					spdlog::debug("Found initial state: {}", load_path.filename().string());
					sim::utility::save_state(sim::utility::load_state(load_path), save_path);
				}
			}
		}
	}

	SimTrainingLogger logger(config, data_path, resume);
	sim::SimAgent sim_agent(std::move(config), &logger, data_path);

	std::signal(SIGINT, ::signal_handler);
	shutdown_handler = [&]([[maybe_unused]] int signum) {
		spdlog::info("Stopping training...");
		sim_agent.stop_train();
		static int shutdown_attempt_count = 0;
		++shutdown_attempt_count;
		if (shutdown_attempt_count > 4)
		{
			std::abort();
		}
	};

	sim_agent.train();

	return 0;
}
