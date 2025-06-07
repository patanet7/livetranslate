// ... existing code ...
    // Helper functions
    std::vector<AudioDevice> getDeviceList(EDataFlow dataFlow);
    std::wstring getDeviceId(int index, EDataFlow dataFlow);
    bool getDeviceInfo(IMMDevice* pDevice, std::string& name, std::wstring& id);
    
    // FFT processing
    void performFFT(const float* buffer, size_t bufferSize, std::vector<float>& magnitudes);

    // HRESULT helper
    static std::string HResultToString(HRESULT hr);
};
// ... existing code ... 