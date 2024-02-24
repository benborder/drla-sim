#pragma once

#include <cmath>
#include <cstddef>

namespace sim
{

namespace Config
{

struct CartPoleEnv
{
	// The magnitude of gravity
	float gravity = 9.8F;
	// The mass of the cart in kg
	float masscart = 1.0F;
	// The mass of the pole in kg
	float masspole = 0.1F;
	// Half the length of the pole in meters
	float length = 0.5F;
	// The magnitude of the force in Newtons
	float force_mag = 10.0F;
	// The seconds between state updates
	float tau = 0.02F;
	// Angle at which to fail the episode
	float theta_threshold_radians = 12 * 2 * M_PI / 360.0F;
	// The max distance to fail the episode
	float x_threshold = 2.4F;
	// The initial state value range
	float initial_range = 0.05F;
	// Use the euler based kinematics solver if true, otherwise use semi-implicit euler solver
	bool kinematics_integrator_euler = true;
	// The resolution for visualisations in pixels per meter
	float resolution = 40.0F;
};

} // namespace Config

struct CartPoleState
{
	// The position of the cart
	float x = 0;
	// The velocity of the cart
	float x_dot = 0;
	// The angle of the pole
	float theta = 0;
	// The angular velocity of the pole
	float theta_dot = 0;
};

} // namespace sim
