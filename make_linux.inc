# Configuration pour MSYS 2
# Peut être modifié à votre convenance...
CXXFLAGS = -std=c++17
ifdef DEBUG
CXXFLAGS += -g -O0 -fbounds-check -pedantic -fsanitize=address -fopenmp
CXXFLAGS2 = CXXFLAGS
else
CXXFLAGS2 = ${CXXFLAGS} -O2 -march=native -fopenmp
CXXFLAGS += -O3 -march=native -fopenmp
endif
LIB=-lsfml-graphics -lsfml-window -lsfml-system
