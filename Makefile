CXX = g++
CXXFLAGS = -Wall -Wextra -pedantic -std=c++23 -fopenmp -g
TARGET = rule110
SRC = rule110.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

# Phony targets (not actual files)
.PHONY: all clean

