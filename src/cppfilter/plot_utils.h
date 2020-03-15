/*
 * Elia Melucci - (c) 2020
 * Esperienze di Programmazione, UniPi
 */

#pragma once

#include <vector>
#include <numeric>
#include <memory>
#include <string_view>

extern "C" {
#include "gnuplot_i.h"
}

constexpr int FFT_RES = 8192 * 4;
std::vector<std::shared_ptr<gnuplot_ctrl>> g_plots;

std::shared_ptr<gnuplot_ctrl> init_plot_environment() {
    auto gp = std::shared_ptr<gnuplot_ctrl>(gnuplot_init(), [](auto p) { gnuplot_close(p); });
    gnuplot_cmd(gp.get(), "set terminal x11");
    gnuplot_cmd(gp.get(), "set grid xtics lt 0 lw 1 lc 'gray'\n"
                          "set grid ytics lt 0 lw 2 lc 'gray'");
    gnuplot_setstyle(gp.get(), "lines");

    return gp;
}

void plot_fft(double *freqs, double *amps, int data, std::string_view title = "FFT") {
    auto gp = g_plots.emplace_back(init_plot_environment()).get();

    std::string range(std::string("set xrange [0:") + std::to_string(std::round(freqs[data - 1])) +
                      std::string("]"));
    gnuplot_cmd(gp, range.c_str());
    gnuplot_set_xlabel(gp, "Frequencies (Hz)");
    gnuplot_set_ylabel(gp, "Magnitude (dBFS)");
    gnuplot_plot_xy(gp, freqs, amps, data, const_cast<char *>(title.cbegin()));
}

void plot_signal(std::vector<double> &samples, int num = FFT_RES, std::string_view title = "FFT") {
    auto gp = g_plots.emplace_back(init_plot_environment()).get();

    gnuplot_set_xlabel(gp, "Samples");
    gnuplot_set_ylabel(gp, "Intensity");

    std::vector<double> indexes(num);
    std::iota(indexes.begin(), indexes.end(), 0);
    gnuplot_plot_xy(gp, indexes.data(), samples.data(), num, const_cast<char *>(title.cbegin()));
}
