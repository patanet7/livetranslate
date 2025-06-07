#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

// WAV file format structure
struct WAVHeader {
    // RIFF header
    char riff_header[4] = {'R', 'I', 'F', 'F'};
    uint32_t file_size = 0;                   // Overall file size - 8
    char wave_header[4] = {'W', 'A', 'V', 'E'};
    
    // fmt sub-chunk
    char fmt_header[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;             // Size of the fmt chunk
    uint16_t audio_format = 1;                // PCM = 1
    uint16_t num_channels = 1;
    uint32_t sample_rate = 16000;
    uint32_t byte_rate = 0;                   // sample_rate * num_channels * bit_depth/8
    uint16_t sample_alignment = 0;            // num_channels * bit_depth/8
    uint16_t bit_depth = 16;                  // bits per sample
    
    // data sub-chunk
    char data_header[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size = 0;                   // Size of actual audio data
};

class WAVFile {
public:
    static bool saveFloat32ToWAV(const std::string& filename, const float* buffer, size_t numSamples, 
                                 int channels = 1, int sampleRate = 16000) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        // Create and initialize the header
        WAVHeader header;
        header.num_channels = static_cast<uint16_t>(channels);
        header.sample_rate = sampleRate;
        header.bit_depth = 16; // We'll convert to 16-bit PCM
        
        // Calculate derived values
        header.sample_alignment = header.num_channels * (header.bit_depth / 8);
        header.byte_rate = header.sample_rate * header.sample_alignment;
        
        // Calculate data size (16-bit samples)
        header.data_size = static_cast<uint32_t>(numSamples * channels * (header.bit_depth / 8));
        header.file_size = 36 + header.data_size; // 36 is the size of the format chunk
        
        // Write header
        file.write(reinterpret_cast<const char*>(&header), sizeof(WAVHeader));
        
        // Convert float to 16-bit PCM and write data
        std::vector<int16_t> pcmBuffer(numSamples);
        for (size_t i = 0; i < numSamples; i++) {
            // Scale and clamp to int16 range
            float sample = buffer[i] * 32767.0f;
            if (sample > 32767.0f) sample = 32767.0f;
            if (sample < -32768.0f) sample = -32768.0f;
            pcmBuffer[i] = static_cast<int16_t>(sample);
        }
        
        // Write PCM data
        file.write(reinterpret_cast<const char*>(pcmBuffer.data()), pcmBuffer.size() * sizeof(int16_t));
        
        return true;
    }
}; 