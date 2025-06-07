#pragma once

#include <string>
#include <ostream>

// Simple WebSocket client for sending audio data
class WebSocketClient {
public:
    WebSocketClient() {}
    ~WebSocketClient() { disconnect(); }
    
    bool connect(const std::string& host, const std::string& port) {
        // In a real implementation, this would connect to a WebSocket server
        // For now, just pretend it's connected
        return true; 
    }
    
    void disconnect() {
        // Close connection if needed
    }
    
    bool sendBinary(const void* data, size_t length) {
        // In a real implementation, this would send binary data to the server
        // For now, just pretend it sent the data
        return true;
    }
}; 