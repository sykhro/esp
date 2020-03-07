/*
 * Elia Melucci - (c) 2020
 * Esperienze di Programmazione, UniPi
 */

#include <algorithm>
#include <cmath>
#include <fftw3.h>

#include "AudioFile.h"
#include "plot_utils.h"

void normalize(AudioFile<double>::AudioBuffer &data, float norm) {
    for(size_t channel = 0; channel < data.size(); channel++) {
        for(size_t i = 0; i < data[channel].size(); i++) {
            data[channel][i] /= norm;
        }
    }
}

/* Converts FFT output to log-scaled power values */
std::pair<std::vector<double>, std::vector<double>> logscale(const std::vector<fftw_complex> &input, uint32_t rate, int size = FFT_RES) {
    double resolution = rate / (size + .0);
    std::vector<double> freqs{};
    std::generate_n(std::back_inserter(freqs), size/2, [resolution](){
        static float count = 0;
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

/* Applies the Hamming window function to a signal */
void apply_hamming(std::vector<double> &input, int size  = FFT_RES) {
    auto static bl = [size]() {
        std::vector<double> bl(size);
            std::transform(bl.begin(), bl.end(), bl.begin(), [size](double) {
                static int v = 0;
                auto res = 25.0/46 - (21.0/46)*cos(2*M_PI*v/size);
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
    std::fill(input.samples[0].begin() + FFT_RES, input.samples[0].end(), 0);
    plot_signal(input.samples[0], input.getNumSamplesPerChannel(), "Windowed input signal");

    /* Perform the transform on the whole file */
    auto transformed = forward_fft(input.samples[0]);
    auto [freqs, amps] = logscale(transformed, input.getSampleRate());
    plot_fft(freqs.data(), amps.data(), amps.size(), "Spectrum of (windowed) input signal");

    return 0;
}