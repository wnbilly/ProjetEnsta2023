#include <SFML/Window/Keyboard.hpp>
#include <ios>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <chrono>
#include "cartesian_grid_of_speed.hpp"
#include "vortex.hpp"
#include "cloud_of_points.hpp"
#include "runge_kutta.hpp"
#include "screen.hpp"
#include <mpi.h>

auto readConfigFile( std::ifstream& input )
{
	using point=Simulation::Vortices::point;

	int isMobile;
	std::size_t nbVortices;
	Numeric::CartesianGridOfSpeed cartesianGrid;
	Geometry::CloudOfPoints cloudOfPoints;
	constexpr std::size_t maxBuffer = 8192;
	char buffer[maxBuffer];
	std::string sbuffer;
	std::stringstream ibuffer;
	// Lit la première ligne de commentaire :
	input.getline(buffer, maxBuffer); // Relit un commentaire
	input.getline(buffer, maxBuffer);// Lecture de la grille cartésienne
	sbuffer = std::string(buffer,maxBuffer);
	ibuffer = std::stringstream(sbuffer);
	double xleft, ybot, h;
	std::size_t nx, ny;
	ibuffer >> xleft >> ybot >> nx >> ny >> h;
	cartesianGrid = Numeric::CartesianGridOfSpeed({nx,ny}, point{xleft,ybot}, h);
	input.getline(buffer, maxBuffer); // Relit un commentaire
	input.getline(buffer, maxBuffer); // Lit mode de génération des particules
	sbuffer = std::string(buffer,maxBuffer);
	ibuffer = std::stringstream(sbuffer);
	int modeGeneration;
	ibuffer >> modeGeneration;
	if (modeGeneration == 0) // Génération sur toute la grille
	{
		std::size_t nbPoints;
		ibuffer >> nbPoints;
		cloudOfPoints = Geometry::generatePointsIn(nbPoints, {cartesianGrid.getLeftBottomVertex(), cartesianGrid.getRightTopVertex()});
	}
	else
	{
		std::size_t nbPoints;
		double xl, xr, yb, yt;
		ibuffer >> xl >> yb >> xr >> yt >> nbPoints;
		cloudOfPoints = Geometry::generatePointsIn(nbPoints, {point{xl,yb}, point{xr,yt}});
	}
	// Lit le nombre de vortex :
	input.getline(buffer, maxBuffer); // Relit un commentaire
	input.getline(buffer, maxBuffer); // Lit le nombre de vortex
	sbuffer = std::string(buffer, maxBuffer);
	ibuffer = std::stringstream(sbuffer);
	try {
		ibuffer >> nbVortices;
	} catch(std::ios_base::failure& err)
	{
		std::cout << "Error " << err.what() << " found" << std::endl;
		std::cout << "Read line : " << sbuffer << std::endl;
		throw err;
	}
	Simulation::Vortices vortices(nbVortices, {cartesianGrid.getLeftBottomVertex(),
											   cartesianGrid.getRightTopVertex()});
	input.getline(buffer, maxBuffer);// Relit un commentaire
	for (std::size_t iVortex=0; iVortex<nbVortices; ++iVortex)
	{
		input.getline(buffer, maxBuffer);
		double x,y,force;
		std::string sbuffer(buffer, maxBuffer);
		std::stringstream ibuffer(sbuffer);
		ibuffer >> x >> y >> force;
		vortices.setVortex(iVortex, point{x,y}, force);
	}
	input.getline(buffer, maxBuffer);// Relit un commentaire
	input.getline(buffer, maxBuffer);// Lit le mode de déplacement des vortex
	sbuffer = std::string(buffer,maxBuffer);
	ibuffer = std::stringstream(sbuffer);
	ibuffer >> isMobile;
	return std::make_tuple(vortices, isMobile, cartesianGrid, cloudOfPoints);
}

// TO LAUNCH :
// make all
// mpirun -n 2 ./vortexSimulation.exe data/simpleSimulation.dat 1280 1024

/*template for timers
auto start = std::chrono::system_clock::now();
auto end = std::chrono::system_clock::now();
std::chrono::duration<double> diff = end - start;
std::cout << std::to_string(diff.count()) << std::endl;
 */


int main( int nargs, char* argv[] )
{
	// Init MPI_COMM_WORLD MPI
	MPI_Init(&nargs, &argv);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Request request = MPI_REQUEST_NULL;

	char const* filename;
	if (nargs==1)
	{
		std::cout << "Usage : vortexsimulator <nom fichier configuration>" << std::endl;
		return EXIT_FAILURE;
	}

	filename = argv[1];
	std::ifstream fich(filename);
	auto config = readConfigFile(fich);
	fich.close();

	std::size_t resx=800, resy=600;
	if (nargs>3)
	{
		resx = std::stoull(argv[2]);
		resy = std::stoull(argv[3]);
	}

	auto vortices = std::get<0>(config);
	auto isMobile = std::get<1>(config);
	auto grid     = std::get<2>(config);
	auto cloud    = std::get<3>(config);

	grid.updateVelocityField(vortices);

	bool animate = false;
	double dt = 0.1;

	// added bool running to easily make the processus 2 stop with the processus 1 (when the window is closed)
	bool running = true;

	if (rank==0) {
		// Code for the display processus

		std::cout << "######## Vortex simulator ########" << std::endl << std::endl;
		std::cout << "Press P for play animation " << std::endl;
		std::cout << "Press S to stop animation" << std::endl;
		std::cout << "Press right cursor to advance step by step in time" << std::endl;
		std::cout << "Press down cursor to halve the time step" << std::endl;
		std::cout << "Press up cursor to double the time step" << std::endl;

		Graphisme::Screen myScreen({resx, resy}, {grid.getLeftBottomVertex(), grid.getRightTopVertex()});

		// Init timers for profiling
		std::chrono::duration<double> totalComm;
		std::chrono::duration<double> totalDisplay;
		unsigned int totalFPS = 0;
		int nbIterFPS = 0;
		int nbIterComm = 0;
		int nbIterDisplay = 0;

		// Display loop
		while (running) {
			auto startDisplay = std::chrono::system_clock::now();
			auto start = std::chrono::system_clock::now();
			bool advance = false;

			// on inspecte tous les évènements de la fenêtre qui ont été émis depuis la précédente itération
			sf::Event event;
			bool ordersToSend;

			while (myScreen.pollEvent(event)) {
				ordersToSend = false;

				// évènement "fermeture demandée" : on ferme la fenêtre
				if (event.type == sf::Event::Closed) {
					myScreen.close();
					running = false;
					ordersToSend = true;
				}
				if (event.type == sf::Event::Resized) {
					// on met à jour la vue, avec la nouvelle taille de la fenêtre
					myScreen.resize(event);
				}
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::P) && !animate) {
					animate = true;
					ordersToSend = true;
				}
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::S) && animate) {
					animate = false;
					ordersToSend = true;
				}
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
					dt *= 2;
					ordersToSend = true;
				}
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
					dt /= 2;
					ordersToSend = true;
				}
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right) && !advance) {
					advance = true;
					ordersToSend = true;
				}
				// On envoie uniquement s'il y a eu des toggle de animate, advance, dt ou fermeture
				// Pour économiser les communications
				if (ordersToSend)
				{
					// On ne compte pas ces temps de communication car elles sont ponctuelles
					std::cout << "0 sending orders" << std::endl;
					MPI_Isend(&running, 1, MPI_CXX_BOOL, 1, 0, MPI_COMM_WORLD, &request);
					MPI_Isend(&animate, 1, MPI_CXX_BOOL, 1, 0, MPI_COMM_WORLD, &request);
					MPI_Isend(&advance, 1, MPI_CXX_BOOL, 1, 0, MPI_COMM_WORLD, &request);
					MPI_Isend(&dt, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &request);
					if (!running) break;
				}
			}

			std::chrono::_V2::system_clock::time_point startComm;
			std::chrono::_V2::system_clock::time_point endComm;

			// Reception of the data to display
			if (animate | advance) {
				startComm = std::chrono::system_clock::now();
				if (isMobile) {
					MPI_Recv(grid.data(), grid.size_for_mpi(), MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(vortices.data(), vortices.size_for_mpi(), MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(cloud.data(), cloud.size_for_mpi(), MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				} else {
					MPI_Recv(grid.data(), grid.size_for_mpi(), MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(cloud.data(), cloud.size_for_mpi(), MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				endComm = std::chrono::system_clock::now();
				nbIterComm++;
			}
			myScreen.clear(sf::Color::Black);
			std::string strDt = std::string("Time step : ") + std::to_string(dt);
			myScreen.drawText(strDt, Geometry::Point<double>{50, double(myScreen.getGeometry().second - 96)});
			myScreen.displayVelocityField(grid, vortices);
			myScreen.displayParticles(grid, vortices, cloud);
			auto end = std::chrono::system_clock::now();
			std::chrono::duration<double> diff = end - start;
			std::string str_fps = std::string("FPS : ") + std::to_string(1. / diff.count());
			myScreen.drawText(str_fps, Geometry::Point<double>{300, double(myScreen.getGeometry().second - 96)});
			myScreen.display();

			// Update the total timers to calculate the means of display time and communication time for the display processus
			auto endDisplay = std::chrono::system_clock::now();
			if (animate | advance)
			{
				totalFPS += 1. / diff.count();
				nbIterFPS++;
			}
			totalComm += endComm - startComm;
			totalDisplay += endDisplay - startDisplay - (endComm - startComm);
			nbIterDisplay++;
		}

		std::cout << "==Processus 0==\nAffichage :" << std::to_string(totalDisplay.count() / nbIterDisplay) << std::endl; // Affichage temps d'affichage moyen
		std::cout << "Communication :" << std::to_string(totalComm.count() / nbIterComm) << std::endl; // Affichage temps de calcul moyen
		std::cout << "FPS :" << std::to_string(totalFPS / nbIterFPS) << std::endl; // Affichage FPS moyen hors pause

	} else if (rank==1) {
		// Code for the computation processus

		// Init timers
		std::chrono::duration<double> totalCalcul;
		std::chrono::duration<double> totalComm;
		int nbIter = 0;

		while (running) {

			bool advance = false;
			int flag = 0;
			MPI_Iprobe(0, 0, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
			if (flag)
			{	// Used MPI_Recv to not miss any toggle of the parameters (for example missing a toggle of running would lead to processus 1 not finishing )
				MPI_Recv(&running, 1, MPI_CXX_BOOL, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv(&animate, 1, MPI_CXX_BOOL, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv(&advance, 1, MPI_CXX_BOOL, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv(&dt, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				std::cout << "1 received params" << std::endl;
				if (!running) break;
			}

			// Used MPI_Isend so the display does not slow down the computing processus
			// It results in the fact that some data could be missed by the display (I guess)
			// but the aim of the project is to improve the performances of the simulation
			if (animate | advance) {
				nbIter++;
				if (isMobile) {
					auto startCalcul = std::chrono::system_clock::now();
					cloud = Numeric::solve_RK4_movable_vortices(dt, grid, vortices, cloud);
					auto endCalcul = std::chrono::system_clock::now();

					auto startComm = std::chrono::system_clock::now();
					MPI_Isend(grid.data(), grid.size_for_mpi(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);
					MPI_Isend(vortices.data(), vortices.size_for_mpi(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);
					MPI_Isend(cloud.data(), cloud.size_for_mpi(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);

					auto endComm= std::chrono::system_clock::now();
					totalCalcul += (endCalcul - startCalcul);
					totalComm += (endComm - startComm);

				} else {
					auto startCalcul = std::chrono::system_clock::now();
					cloud = Numeric::solve_RK4_fixed_vortices(dt, grid, cloud);
					auto endCalcul = std::chrono::system_clock::now();

					auto startComm = std::chrono::system_clock::now();
					MPI_Isend(grid.data(), grid.size_for_mpi(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);
					MPI_Isend(cloud.data(), cloud.size_for_mpi(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);
					auto endComm= std::chrono::system_clock::now();
					totalCalcul += (endCalcul - startCalcul);
					totalComm += (endComm - startComm);
				}
			}
		}
		std::cout << "==Processus 1==\nCommunication :" << std::to_string(totalComm.count() / nbIter) << std::endl; // Affichage temps de communication principale moyen
		std::cout << "Calcul :" << std::to_string(totalCalcul.count() / nbIter) << "\n" << std::endl; // Affichage temps de calcul moyen
	}
	std::cout << "End of processus " << rank << std::endl;
	MPI_Finalize();
	return EXIT_SUCCESS;
}