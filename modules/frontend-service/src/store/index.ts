import { configureStore } from "@reduxjs/toolkit";
import { TypedUseSelectorHook, useDispatch, useSelector } from "react-redux";

// Import slices
import audioSlice from "./slices/audioSlice";
import botSlice from "./slices/botSlice";
import websocketSlice from "./slices/websocketSlice";
import uiSlice from "./slices/uiSlice";
import systemSlice from "./slices/systemSlice";
import { apiSlice } from "./slices/apiSlice";

export const store = configureStore({
  reducer: {
    audio: audioSlice.reducer,
    bot: botSlice.reducer,
    websocket: websocketSlice.reducer,
    ui: uiSlice.reducer,
    system: systemSlice.reducer,
    api: apiSlice.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types for serialization check
        ignoredActions: [
          "audio/setVisualizationData",
          "bot/updateAudioCapture",
          "websocket/messageReceived",
        ],
        // Ignore these field paths in the state (keep frequency/time data for visualization)
        ignoredActionsPaths: ["payload.frequencyData", "payload.timeData"],
        ignoredPaths: [
          "audio.visualization.frequencyData",
          "audio.visualization.timeData",
        ],
      },
    }).concat(apiSlice.middleware),
  devTools: process.env.NODE_ENV !== "production",
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Typed hooks for TypeScript
export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;

export default store;
