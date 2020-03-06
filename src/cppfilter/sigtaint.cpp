/*
 * Elia Melucci - (c) 2020
 * Esperienze di Programmazione, UniPi
 */

#include <cmath>
#include "AudioFile.h"

AudioFile<double>::AudioBuffer make_test_tone(int channels, int total_samples, float sample_rate, float frequency = 1000.0) {
    AudioFile<double>::AudioBuffer out;
    out.resize(channels);
    for(int i = 0; i < channels; i++) {
        out[i].resize(total_samples);
    }

    for(int i = 0; i < total_samples; i++) {
       double sample = sinf(2. * M_PI * (i / sample_rate) * frequency);

       for(int channel = 0; channel < channels; channel++) {
           /* ~-6db */
           out[channel][i] = sample * 0.5;
       }
    }

    return out;
}

void normalize(AudioFile<double>::AudioBuffer &data, float norm) {
    for(size_t channel = 0; channel < data.size(); channel++) {
        for(size_t i = 0; i < data[channel].size(); i++) {
            data[channel][i] /= norm;
        }
    }
}

int main(int argc, char *argv[])
{
    if(argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        exit(EXIT_FAILURE);
    }

    AudioFile<double> input;
    input.load(argv[1]);
    auto chans = input.getNumChannels();
    auto samps = input.getNumSamplesPerChannel();
    auto tone = make_test_tone(chans, samps, input.getSampleRate());
    double norm = -1;

    for(int channel = 0; channel < chans; channel++) {
        for(int i = 0; i < samps; i++) {
            input.samples[channel][i] += tone[channel][i];
            /* Keep track of highest peak */
            if(std::fabs(input.samples[channel][i]) > norm) {
                norm = input.samples[channel][i];
            }
        }
    }

    /* Normalize if needed */
    if(norm > 1.0) {
        normalize(input.samples, norm);
    }
    input.save("sum.wav");

    return 0;
}
