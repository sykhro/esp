/*
 * Elia Melucci - (c) 2020
 * Esperienze di Programmazione, UniPi
 */

#include <algorithm>
#include <cmath>
#include <fftw3.h>
#include <cstddef>

#include "AudioFile.h"
#include "plot_utils.h"

/* Converts FFT output to log-scaled power values */
std::pair<std::vector<double>, std::vector<double>> logscale(const std::vector<fftw_complex> &input, uint32_t rate, int size = FFT_RES) {
    double resolution = rate / (size + .0);
    std::vector<double> freqs{};
    float count = 0;
    std::generate_n(std::back_inserter(freqs), size/2, [resolution, count]() mutable {
        return count++ * resolution;
    });

    std::vector<double> amps{};
    for(int i = 0; i < size/2; i++) {
        amps.push_back(
                    (10*log(std::pow(input[i][0], 2) + std::pow(input[i][1], 2))) / FFT_RES
                );
    }

    return std::make_pair(freqs, amps);
}

/* Performs forward FFT on a signal */
std::vector<fftw_complex> forward_fft(std::vector<double> &input, int size = FFT_RES){
    if(input.size() < (size_t)size) {
        std::cerr << "WARNING - File too small for the chosen FFT window";
        exit(EXIT_FAILURE);
    }

    std::vector<fftw_complex> fftout(size);
    fftw_plan fwd_plan = fftw_plan_dft_r2c_1d(size, input.data(), fftout.data(),  FFTW_ESTIMATE);
    fftw_execute(fwd_plan);

    fftw_destroy_plan(fwd_plan);
    fftw_cleanup();

    return fftout;
}

/* Performs backward FFT on a signal */
std::vector<double> backwards_fft(std::vector<fftw_complex> &input, int size = FFT_RES){
    if(input.size() < (size_t)size) {
        std::cerr << "WARNING - File too small for the chosen FFT window";
        exit(EXIT_FAILURE);
    }

    std::vector<double> fftout(size);
    fftw_plan fwd_plan = fftw_plan_dft_c2r_1d(size, input.data(), fftout.data(),  FFTW_ESTIMATE);
    fftw_execute(fwd_plan);

    fftw_destroy_plan(fwd_plan);
    fftw_cleanup();

    return fftout;
}

/* Applies the Hamming window function to a signal */
void apply_hamming(std::vector<double> &input, int size  = FFT_RES) {
    auto static bl = [size]() {
        std::vector<double> bl(size);
            int v = 0;
            std::fill(bl.begin(), bl.end(), [size, v] () mutable {
                auto res = 25.0/46 - (21.0/46)*cos(2*M_PI*v / (size-1));
                v++;
                return res;
            });

        return bl;
    }();

    std::transform(input.begin(), input.begin() + size,
                    bl.begin(), input.begin(), std::multiplies<double>());
}

int main(int argc, char *argv[]) {

    if(argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        exit(EXIT_FAILURE);
    }

    AudioFile<double> input;
    input.load(argv[1]);

    /* Apply window function and padding */
    apply_hamming(input.samples[0]);
    plot_signal(input.samples[0], FFT_RES, "Windowed input signal (padding hidden)");

    /* Perform the transform on the selected window */
    auto in_transformed = forward_fft(input.samples[0]);
    auto [freqs_n, amps_n] = logscale(in_transformed, input.getSampleRate());
    plot_fft(freqs_n.data(), amps_n.data(), amps_n.size(), "Spectrum of (windowed) input signal");

    /* Zero out the 1kHz band and transform back */
    in_transformed[(int)(1000 * (FFT_RES + .0)/input.getSampleRate())][0] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0)/input.getSampleRate())][1] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0)/input.getSampleRate()) + 1][0] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0)/input.getSampleRate()) + 1][1] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0)/input.getSampleRate()) - 1][0] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0)/input.getSampleRate()) - 1][1] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0)/input.getSampleRate()) + 2][0] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0)/input.getSampleRate()) + 2][1] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0)/input.getSampleRate()) - 2][0] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0)/input.getSampleRate()) - 2][1] = 0;
    auto [freqs, amps] = logscale(in_transformed, input.getSampleRate());
    plot_fft(freqs_n.data(), amps.data(), amps.size(), "Spectrum of (windowed) input signal");

    auto backagain = backwards_fft(in_transformed);
    plot_signal(backagain, FFT_RES, "Back again");
    auto max = *std::max_element(backagain.begin(), backagain.end(), [](double a, double b){return (std::abs(a) < std::abs(b));});
    std::transform(backagain.begin(), backagain.end(), backagain.begin(),
                   [max](double d){ return d / max; });

    /* Attach audio buffer and save cleaned signal */
    std::copy(backagain.begin(), backagain.begin() + FFT_RES, input.samples[0].begin());
    input.save("sigcleaned.wav");

    return 0;
}
