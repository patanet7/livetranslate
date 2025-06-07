void GUI::renderAudioControls() {
    if (ImGui::CollapsingHeader("Audio Controls")) {
        // Recording controls
        if (ImGui::Button(audioCapture_->isRecording() ? "Stop Recording" : "Start Recording")) {
            if (audioCapture_->isRecording()) {
                audioCapture_->stopRecording();
            } else {
                audioCapture_->startRecording();
            }
        }

        // Auto-save directory
        static char saveDir[256] = "";
        if (ImGui::InputText("Save Directory", saveDir, sizeof(saveDir), ImGuiInputTextFlags_EnterReturnsTrue)) {
            audioCapture_->setAutoSaveDirectory(saveDir);
        }

        // ... rest of audio controls ...
    }
} 