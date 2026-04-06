#ifndef FISK_NATIVE_PLOTTER_HPP
#define FISK_NATIVE_PLOTTER_HPP

#ifdef _WIN32
#include <windows.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>

class FiskNativePlotter {
public:
    enum Style { STEPS, POINTS, LINES };

    struct Series {
        std::string label;
        std::vector<double> x;
        std::vector<double> y;
        Style style;
        COLORREF color;
    };

    struct PlotRequest {
        std::string title;
        std::string xlabel;
        std::string ylabel;
        std::vector<Series> series;
    };

    static void Show(PlotRequest* req) {
        HANDLE hThread = CreateThread(NULL, 0, WindowThread, (LPVOID)req, 0, NULL);
        if (!hThread) {
            std::cerr << "[!] Failed to create plot thread.\n";
            delete req;
        } else {
            CloseHandle(hThread); // We don't need to join it
        }
    }

private:
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
        PlotRequest* req = (PlotRequest*)GetWindowLongPtr(hwnd, GWLP_USERDATA);
        switch (uMsg) {
            case WM_PAINT: {
                PAINTSTRUCT ps;
                HDC hdc = BeginPaint(hwnd, &ps);
                if (req) DrawPlot(hdc, hwnd, req);
                EndPaint(hwnd, &ps);
                return 0;
            }
            case WM_DESTROY:
                PostQuitMessage(0);
                return 0;
        }
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }

    static void DrawPlot(HDC hdc, HWND hwnd, PlotRequest* req) {
        RECT rect;
        GetClientRect(hwnd, &rect);
        int w = rect.right, h = rect.bottom;
        int margin = 70;

        double minX = 0, maxX = 1, minY = 0, maxY = 1;
        bool first = true;
        for (const auto& s : req->series) {
            for (size_t i = 0; i < s.x.size(); ++i) {
                if (first) { minX = maxX = s.x[i]; minY = maxY = s.y[i]; first = false; }
                minX = (std::min)(minX, s.x[i]); maxX = (std::max)(maxX, s.x[i]);
                minY = (std::min)(minY, s.y[i]); maxY = (std::max)(maxY, s.y[i]);
            }
        }
        
        double rangeX = (std::abs(maxX - minX) < 1e-9) ? 1.0 : (maxX - minX) * 1.1;
        double rangeY = (std::abs(maxY - minY) < 1e-9) ? 1.0 : (maxY - minY) * 1.1;

        auto toX = [&](double val) { return margin + (int)((val - minX) / rangeX * (w - 2 * margin)); };
        auto toY = [&](double val) { return (h - margin) - (int)((val - minY) / rangeY * (h - 2 * margin)); };

        // Clear Background
        HBRUSH hWhite = CreateSolidBrush(RGB(255, 255, 255));
        FillRect(hdc, &rect, hWhite);
        DeleteObject(hWhite);
        
        // Draw Axes
        HPEN hAxisPen = CreatePen(PS_SOLID, 2, RGB(0, 0, 0));
        SelectObject(hdc, hAxisPen);
        MoveToEx(hdc, margin, margin, NULL); LineTo(hdc, margin, h - margin);
        LineTo(hdc, w - margin, h - margin);
        DeleteObject(hAxisPen);

        SetBkMode(hdc, TRANSPARENT);
        TextOutA(hdc, w/2 - 50, 10, req->title.c_str(), (int)req->title.length());

        for (const auto& s : req->series) {
            if (s.x.empty()) continue;
            HPEN hPen = CreatePen(PS_SOLID, 2, s.color);
            SelectObject(hdc, hPen);

            int px = toX(s.x[0]);
            int py = toY(s.y[0]);

            for (size_t i = 0; i < s.x.size(); ++i) {
                int cx = toX(s.x[i]);
                int cy = toY(s.y[i]);

                if (s.style == STEPS) {
                    MoveToEx(hdc, px, py, NULL); LineTo(hdc, cx, py); LineTo(hdc, cx, cy);
                } else if (s.style == LINES) {
                    MoveToEx(hdc, px, py, NULL); LineTo(hdc, cx, cy);
                } else if (s.style == POINTS) {
                    Ellipse(hdc, cx-3, cy-3, cx+3, cy+3);
                }
                px = cx; py = cy;
            }
            DeleteObject(hPen);
        }
    }

    static DWORD WINAPI WindowThread(LPVOID param) {
        PlotRequest* req = (PlotRequest*)param;
        HINSTANCE hInst = GetModuleHandle(NULL);
        
        const char* CLASS_NAME = "FiskNativePlot";
        WNDCLASS wc = {0};
        wc.lpfnWndProc = WindowProc;
        wc.hInstance = hInst;
        wc.lpszClassName = CLASS_NAME;
        wc.hCursor = LoadCursor(NULL, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

        RegisterClass(&wc);

        HWND hwnd = CreateWindowEx(0, CLASS_NAME, req->title.c_str(), 
                                   WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                                   CW_USEDEFAULT, CW_USEDEFAULT, 800, 600, 
                                   NULL, NULL, hInst, NULL);

        if (!hwnd) {
            std::cerr << "[!] Window creation failed. Error: " << GetLastError() << "\n";
            delete req;
            return 0;
        }

        SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)req);

        MSG msg;
        while (GetMessage(&msg, NULL, 0, 0)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        
        delete req;
        return 0;
    }
};
#endif
#endif