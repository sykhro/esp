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
std::pair<std::vector<double>, std::vector<double>> logscale(const std::vector<fftw_complex> &input,
                                                             uint32_t rate) {

    /* Generate x-axis */
    auto size = input.size();
    double resolution = rate / (size + .0);
    std::vector<double> freqs(size);
    std::generate(freqs.begin(), freqs.end(),
                    [resolution, count = 0]() mutable { return count++ * resolution; });

    /* Normalize and obtain log scale */
    std::vector<double> amps{};
    for (auto n : input) {
        amps.push_back((std::pow(n[0], 2) + std::pow(n[1], 2))/size);
    }

    auto max = *std::max_element(amps.begin(), amps.end());
    std::transform(amps.begin(), amps.end(), amps.begin(), [max](double mag) { return 10 * log(mag/max); });

    return std::make_pair(freqs, amps);
}

/* Performs forward FFT on a signal */
std::vector<fftw_complex> forward_fft(std::vector<double> &input, int size = FFT_RES) {
    if (input.size() < static_cast<std::size_t>(size)) {
        std::cerr << "ERROR - Not enough samples!";
        exit(EXIT_FAILURE);
    }

    std::vector<fftw_complex> fftout(size/2 + 1);
    fftw_plan fwd_plan = fftw_plan_dft_r2c_1d(size, input.data(), fftout.data(), FFTW_ESTIMATE);
    fftw_execute(fwd_plan);

    fftw_destroy_plan(fwd_plan);
    fftw_cleanup();

    return fftout;
}

/* Performs backward FFT on a signal */
std::vector<double> backwards_fft(std::vector<fftw_complex> &input, int size = FFT_RES) {
    if (input.size() != std::size_t(size/2 + 1)) {
        std::cerr << "WARNING - funny complex input size!";
    }

    std::vector<double> fftout(size);
    fftw_plan fwd_plan = fftw_plan_dft_c2r_1d(size, input.data(), fftout.data(), FFTW_ESTIMATE);
    fftw_execute(fwd_plan);

    fftw_destroy_plan(fwd_plan);
    fftw_cleanup();

    return fftout;
}

/* Applies the Hamming window function to a signal */
template <class InputIt>
void apply_hamming(InputIt first, InputIt last) {
    auto size = std::distance(first, last);
    auto static bl = [size]() {
        std::vector<double> values(size);
        std::generate(values.begin(), values.end(), [size, v = 0]() mutable {
            double res = 25.0 / 46 - (21.0 / 46) * cos(2 * M_PI * v / (size - 1));
            v++;
            return res;
        });

        return values;
    }();

   std::transform(first, last, bl.begin(), first,
                   std::multiplies<double>());
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        exit(EXIT_FAILURE);
    }

    AudioFile<double> input;
    input.load(argv[1]);

    /* Apply window function and padding */
    apply_hamming(input.samples[0].begin(), input.samples[0].begin() + FFT_RES);
    plot_signal(input.samples[0], FFT_RES, "Windowed input signal (padding hidden)");

    /* Perform the transform on the very first window */
    auto in_transformed = forward_fft(input.samples[0]);
    auto [freqs_n, amps_n] = logscale(in_transformed, input.getSampleRate());
    plot_fft(freqs_n.data(), amps_n.data(), amps_n.size(), "Spectrum of (windowed) input signal");

    /* Zero out the 1kHz band(s) and transform back */
    in_transformed[(int)(1000 * (FFT_RES + .0) / input.getSampleRate())][0] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0) / input.getSampleRate())][1] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0) / input.getSampleRate()) + 1][0] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0) / input.getSampleRate()) + 1][1] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0) / input.getSampleRate()) - 1][0] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0) / input.getSampleRate()) - 1][1] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0) / input.getSampleRate()) + 2][0] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0) / input.getSampleRate()) + 2][1] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0) / input.getSampleRate()) - 2][0] = 0;
    in_transformed[(int)(1000 * (FFT_RES + .0) / input.getSampleRate()) - 2][1] = 0;
    auto [freqs, amps] = logscale(in_transformed, input.getSampleRate());
    plot_fft(freqs.data(), amps.data(), amps.size(), "Spectrum of filtered signal");

    auto backagain = backwards_fft(in_transformed);
    plot_signal(backagain, FFT_RES, "Back again");
    auto max = *std::max_element(backagain.begin(), backagain.end(),
                                 [](double a, double b) { return (std::abs(a) < std::abs(b)); });
    std::transform(backagain.begin(), backagain.end(), backagain.begin(),
                   [max](double d) { return d / max; });

    /* Attach audio buffer and save cleaned signal */
    std::copy(backagain.begin(), backagain.begin() + FFT_RES, input.samples[0].begin());
    input.save("sigcleaned.wav");

    return 0;
}
