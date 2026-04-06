# Find your compilation line and update it:
shellFisk: main.cpp
	g++ -O3 -march=native -std=c++17 \
	-I./dependencies/eigen \
	-I./dependencies/nlohmann/include \
	main.cpp \
	-static \
	-o shellFisk.exe \
	-lgdi32 -luser32 -lkernel32