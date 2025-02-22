#include <iostream>
#include <chrono>
#include <omp.h>
#include "runge_kutta.hpp"
#include "cartesian_grid_of_speed.hpp"

using namespace Geometry;

Geometry::CloudOfPoints
Numeric::solve_RK4_fixed_vortices( double dt, CartesianGridOfSpeed const& t_velocity, Geometry::CloudOfPoints const& t_points )
{
    constexpr double onesixth = 1./6.;
    using vector = Simulation::Vortices::vector;
    using point  = Simulation::Vortices::point;

    Geometry::CloudOfPoints newCloud(t_points.numberOfPoints());
    // On ne bouge que les points :
	auto start = std::chrono::system_clock::now();
	#pragma omp parallel for
    for ( std::size_t iPoint=0; iPoint<t_points.numberOfPoints(); ++iPoint)
    {
        point  p = t_points[iPoint];
        vector v1 = t_velocity.computeVelocityFor(p);
        point p1 = p + 0.5*dt*v1;
        p1 = t_velocity.updatePosition(p1);
        vector v2 = t_velocity.computeVelocityFor(p1);
        point p2 = p + 0.5*dt*v2;
        p2 = t_velocity.updatePosition(p2);
        vector v3 = t_velocity.computeVelocityFor(p2);
        point p3 = p + dt*v3;
        p3 = t_velocity.updatePosition(p3);
        vector v4 = t_velocity.computeVelocityFor(p3);
        newCloud[iPoint] = t_velocity.updatePosition(p + onesixth*dt*(v1+2.*v2+2.*v3+v4));
    }
	auto diff = std::chrono::duration<double>(std::chrono::system_clock::now() - start);
	std::cout << "Computation fixed vortices (s) : " << std::to_string(diff.count()) << std::endl;
    return newCloud;
}

Geometry::CloudOfPoints
Numeric::solve_RK4_movable_vortices( double dt, CartesianGridOfSpeed& t_velocity, 
                                     Simulation::Vortices& t_vortices, 
                                     Geometry::CloudOfPoints const& t_points )
{
    constexpr double onesixth = 1./6.;
    using vector = Simulation::Vortices::vector;
    using point  = Simulation::Vortices::point;

    Geometry::CloudOfPoints newCloud(t_points.numberOfPoints());
    // On ne bouge que les points :
	auto start = std::chrono::system_clock::now();
	#pragma omp parallel for
    for ( std::size_t iPoint=0; iPoint<t_points.numberOfPoints(); ++iPoint)
    {
        point  p = t_points[iPoint];
        vector v1 = t_velocity.computeVelocityFor(p);
        point p1 = p + 0.5*dt*v1;
        p1 = t_velocity.updatePosition(p1);
        vector v2 = t_velocity.computeVelocityFor(p1);
        point p2 = p + 0.5*dt*v2;
        p2 = t_velocity.updatePosition(p2);
        vector v3 = t_velocity.computeVelocityFor(p2);
        point p3 = p + dt*v3;
        p3 = t_velocity.updatePosition(p3);
        vector v4 = t_velocity.computeVelocityFor(p3);
        newCloud[iPoint] = t_velocity.updatePosition(p + onesixth*dt*(v1+2.*v2+2.*v3+v4));
    }
	auto diff = std::chrono::duration<double>(std::chrono::system_clock::now() - start);
	//std::cout << "1st loop movable vortices (s) : " << std::to_string(diff.count()) << std::endl;

    std::vector<point> newVortexCenter;
    newVortexCenter.reserve(t_vortices.numberOfVortices());

	start = std::chrono::system_clock::now();
    for (std::size_t iVortex=0; iVortex<t_vortices.numberOfVortices(); ++iVortex)
    {
        point p = t_vortices.getCenter(iVortex);
        vector v1 = t_vortices.computeSpeed(p);
        point p1 = p + 0.5*dt*v1;
        p1 = t_velocity.updatePosition(p1);
        vector v2 = t_vortices.computeSpeed(p1);
        point p2 = p + 0.5*dt*v2;
        p2 = t_velocity.updatePosition(p2);
        vector v3 = t_vortices.computeSpeed(p2);
        point p3 = p + dt*v3;
        p3 = t_velocity.updatePosition(p3);
        vector v4 = t_vortices.computeSpeed(p3);
        newVortexCenter.emplace_back(t_velocity.updatePosition(p + onesixth*dt*(v1+2.*v2+2.*v3+v4)));
    }
	diff = std::chrono::duration<double>(std::chrono::system_clock::now() - start);
	//std::cout << "2nd loop movable vortices (s) : " << std::to_string(diff.count()) << std::endl;

	start = std::chrono::system_clock::now();
    for (std::size_t iVortex=0; iVortex<t_vortices.numberOfVortices(); ++iVortex)
    {
        t_vortices.setVortex(iVortex, newVortexCenter[iVortex], 
                             t_vortices.getIntensity(iVortex));
    }

	diff = std::chrono::duration<double>(std::chrono::system_clock::now() - start);
	//std::cout << "3rd loop movable vortices (s) : " << std::to_string(diff.count()) << std::endl;

    t_velocity.updateVelocityField(t_vortices);
    return newCloud;

}