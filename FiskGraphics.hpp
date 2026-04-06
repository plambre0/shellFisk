#ifndef FISK_GRAPHICS_HPP
#define FISK_GRAPHICS_HPP

#include <vector>
#include <string>
#include <iostream>

class FiskCanvas {
private:
    int width, height;
    std::vector<std::string> grid;

public:
    FiskCanvas(int w, int h) : width(w), height(h) {
        grid.assign(height, std::string(width, ' '));
    }

    void set_subpixel(int x, int y) {
        // Map high-res coordinates to ASCII grid
        int char_x = x / 2; 
        int char_y = y / 4;

        if (char_x >= 0 && char_x < width && char_y >= 0 && char_y < height) {
            char current = grid[char_y][char_x];
            // Intensity-based ASCII mapping
            if (current == ' ') grid[char_y][char_x] = '.';
            else if (current == '.') grid[char_y][char_x] = '+';
            else if (current == '+') grid[char_y][char_x] = '*';
            else if (current == '*') grid[char_y][char_x] = '#';
        }
    }

    void draw() {
        // Draw top border
        std::cout << "  +" << std::string(width, '-') << "+\n";
        
        for (int i = 0; i < height; ++i) {
            std::cout << "  |" << grid[i] << "|\n";
        }
        
        // Draw bottom border
        std::cout << "  +" << std::string(width, '-') << "+\n";
    }
};

#endif