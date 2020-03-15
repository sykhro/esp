/*
 * Elia Melucci - (c) 2020
 * Esperienze di Programmazione, UniPi
 */

#define _USE_MATH_DEFINES
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
    auto size = input.size() - 1;
    double resolution = rate / (2 * (size - 1.0));
    std::vector<double> freqs(size);
    std::generate(freqs.begin(), freqs.end(),
                  [resolution, count = 0]() mutable { return count++ * resolution; });

    /* Normalize and obtain log scale */
    std::vector<double> amps{};
    for (auto n : input) {
        amps.push_back((std::pow(n[0], 2) + std::pow(n[1], 2)) / (2 * (size - 1)));
    }

    auto max = *std::max_element(amps.begin(), amps.end());
    std::transform(amps.begin(), amps.end(), amps.begin(),
                   [max](double mag) { return 10 * log(mag / max); });

    return std::make_pair(freqs, amps);
}

/* Performs forward FFT on a signal */
std::vector<fftw_complex> forward_fft(std::vector<double> &input, int size = FFT_RES) {
    if (input.size() < static_cast<std::size_t>(size)) {
        std::cerr << "ERROR - Not enough samples!";
        exit(EXIT_FAILURE);
    }

    std::vector<fftw_complex> fftout(size / 2 + 1);
    fftw_plan fwd_plan = fftw_plan_dft_r2c_1d(size, input.data(), fftout.data(), FFTW_ESTIMATE);
    fftw_execute(fwd_plan);

    fftw_destroy_plan(fwd_plan);
    fftw_cleanup();

    return fftout;
}

/* Performs backward FFT on a signal */
std::vector<double> backwards_fft(std::vector<fftw_complex> &input, int size = FFT_RES) {
    if (input.size() != std::size_t(size / 2 + 1)) {
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
    auto bl = [size]() {
        std::vector<double> values(size);
        std::generate(values.begin(), values.end(), [size, v = 0]() mutable {
            double res = 25.0 / 46 - (21.0 / 46) * std::cos(2 * M_PI * v / (size - 1));
            v++;
            return res;
        });

        return values;
    }();

    std::transform(first, last, bl.begin(), first, std::multiplies<double>());
}

void process_channel(std::vector<double> &audiodata, uint32_t sample_rate) {
    std::vector<double> out(audiodata.size() + 2 * FFT_RES, 0.0);
    for (std::size_t i = 0; i + FFT_RES / 4 < audiodata.size(); i += FFT_RES / 4) {
        /* Apply Hamming window to the audio data and pad it */
        std::vector<double> procdata(FFT_RES);
        std::copy(audiodata.begin() + i, audiodata.begin() + i + FFT_RES / 2, procdata.begin());
        apply_hamming(procdata.begin(), procdata.begin() + FFT_RES / 2);

        /* Transform and filter */
        auto in_transformed = forward_fft(procdata);
        /*
                auto [f, a] = logscale(in_transformed, sample_rate);
                plot_fft(f.data(), a.data(), f.size(), "Unfiltered");
        */
        in_transformed[(int)(1000 * (FFT_RES + .0) / sample_rate)][0] = 0;
        in_transformed[(int)(1000 * (FFT_RES + .0) / sample_rate)][1] = 0;
        in_transformed[(int)(1000 * (FFT_RES + .0) / sample_rate) - 1][0] = 0;
        in_transformed[(int)(1000 * (FFT_RES + .0) / sample_rate) - 1][1] = 0;
        in_transformed[(int)(1000 * (FFT_RES + .0) / sample_rate) + 1][0] = 0;
        in_transformed[(int)(1000 * (FFT_RES + .0) / sample_rate) + 1][1] = 0;
/*
        auto [f2, a2] = logscale(in_transformed, sample_rate);
        plot_fft(f2.data(), a2.data(), f2.size(), "Filtered");
*/
        /* Revert to signal and overlap-add */
        auto backagain = backwards_fft(in_transformed);
        std::transform(backagain.begin(), backagain.end(), out.begin() + i, out.begin() + i,
                       std::plus<double>());
    }

    /* Normalize signal */
    auto max = std::abs(*std::max_element(
        out.begin(), out.end(), [](double a, double b) { return (std::abs(a) < std::abs(b)); }));
    std::transform(out.begin(), out.end(), out.begin(), [max](double d) { return d / max; });

    /* Overwrite audio buffer and save cleaned signal */
    std::copy(out.begin(), out.begin() + audiodata.size(), audiodata.begin());
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        exit(EXIT_FAILURE);
    }

    AudioFile<double> input;
    if(!input.load(argv[1])) {
        exit(EXIT_FAILURE);
    }

    std::cout << "Processing " << input.getNumSamplesPerChannel() << " samples on "
              << input.getNumChannels() << " channels...\n";

    for (auto &channel : input.samples) {
        process_channel(channel, input.getSampleRate());
    }

    input.save("sigcleaned.wav");

    return 0;
}
