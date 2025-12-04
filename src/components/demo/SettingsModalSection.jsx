import { useState } from "react";

// Buffer size options
const BUFFER_SIZE_OPTIONS = [3, 5, 10, 15, 20];

// Default settings values
const DEFAULT_SETTINGS = {
  sensitivity: 70,
  autoSpeak: false,
  mirrorVideo: true,
  camera: "user",
  voiceGender: "female",
  bufferSize: 5,
};

export function SettingsModalSection({ isOpen, onClose }) {
  const [sensitivity, setSensitivity] = useState(DEFAULT_SETTINGS.sensitivity);
  const [autoSpeak, setAutoSpeak] = useState(DEFAULT_SETTINGS.autoSpeak);
  const [mirrorVideo, setMirrorVideo] = useState(DEFAULT_SETTINGS.mirrorVideo);
  const [camera, setCamera] = useState(DEFAULT_SETTINGS.camera);
  const [voiceGender, setVoiceGender] = useState(DEFAULT_SETTINGS.voiceGender);
  const [bufferSize, setBufferSize] = useState(DEFAULT_SETTINGS.bufferSize);

  const resetToDefault = () => {
    setSensitivity(DEFAULT_SETTINGS.sensitivity);
    setAutoSpeak(DEFAULT_SETTINGS.autoSpeak);
    setMirrorVideo(DEFAULT_SETTINGS.mirrorVideo);
    setCamera(DEFAULT_SETTINGS.camera);
    setVoiceGender(DEFAULT_SETTINGS.voiceGender);
    setBufferSize(DEFAULT_SETTINGS.bufferSize);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-md"
        onClick={onClose}
      />
      <div className="relative w-full max-w-md bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-8 shadow-2xl">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-6 right-6 text-white/50 hover:text-white transition-colors"
        >
          <svg
            className="w-6 h-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>

        {/* Title */}
        <h3 className="font-heading text-2xl font-semibold text-white mb-2">
          Settings
        </h3>

        {/* Subtitle */}
        <p className="text-white/60 font-sans text-sm mb-6">
          Configure your demo experience.
        </p>

        <div className="space-y-5">
          {/* Camera Selection */}
          <div>
            <label className="block text-white/80 text-sm font-sans font-medium mb-2">
              Camera
            </label>
            <select
              value={camera}
              onChange={(e) => setCamera(e.target.value)}
              className="w-full px-4 py-3 bg-white/5 backdrop-blur border border-white/20 rounded-lg text-white text-sm font-sans focus:outline-none focus:border-white/40 transition-colors appearance-none cursor-pointer"
            >
              <option value="user" className="bg-[#1a1a1a]">
                Front Camera
              </option>
              <option value="environment" className="bg-[#1a1a1a]">
                Back Camera
              </option>
            </select>
          </div>

          {/* Detection Sensitivity */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-white/80 text-sm font-sans font-medium">
                Detection Sensitivity
              </label>
              <span className="text-brand-primary text-sm font-sans font-semibold">
                {sensitivity}%
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={sensitivity}
              onChange={(e) => setSensitivity(Number(e.target.value))}
              className="w-full h-2 bg-white/10 rounded-full appearance-none cursor-pointer accent-brand-primary"
            />
            <div className="flex justify-between text-white/40 text-xs font-sans mt-1">
              <span>Low</span>
              <span>High</span>
            </div>
          </div>

          {/* Auto-speak toggle */}
          <div className="flex items-center justify-between py-2">
            <label className="text-white/80 text-sm font-sans font-medium">
              Auto-speak detected text
            </label>
            <button
              onClick={() => setAutoSpeak(!autoSpeak)}
              className={`w-12 h-7 rounded-full relative transition-colors ${
                autoSpeak
                  ? "bg-brand-primary"
                  : "bg-white/10 border border-white/20"
              }`}
            >
              <span
                className={`absolute left-1 top-1 w-5 h-5 bg-white rounded-full transition-all duration-200 shadow-md ${
                  autoSpeak ? "translate-x-5" : "translate-x-0"
                }`}
              />
            </button>
          </div>

          {/* Mirror Video toggle */}
          <div className="flex items-center justify-between py-2">
            <label className="text-white/80 text-sm font-sans font-medium">
              Mirror video
            </label>
            <button
              onClick={() => setMirrorVideo(!mirrorVideo)}
              className={`w-12 h-7 rounded-full relative transition-colors ${
                mirrorVideo
                  ? "bg-brand-primary"
                  : "bg-white/10 border border-white/20"
              }`}
            >
              <span
                className={`absolute left-1 top-1 w-5 h-5 bg-white rounded-full transition-all duration-200 shadow-md ${
                  mirrorVideo ? "translate-x-5" : "translate-x-0"
                }`}
              />
            </button>
          </div>

          {/* Voice Gender Selection */}
          <div>
            <label className="block text-white/80 text-sm font-sans font-medium mb-2">
              Voice
            </label>
            <div className="flex gap-2">
              <button
                onClick={() => setVoiceGender("male")}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg border transition-colors ${
                  voiceGender === "male"
                    ? "bg-brand-primary/20 border-brand-primary text-white"
                    : "bg-white/5 border-white/20 text-white/60 hover:bg-white/10"
                }`}
              >
                <svg
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z"
                  />
                </svg>
                <span className="text-sm font-sans font-medium">Male</span>
              </button>
              <button
                onClick={() => setVoiceGender("female")}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg border transition-colors ${
                  voiceGender === "female"
                    ? "bg-brand-primary/20 border-brand-primary text-white"
                    : "bg-white/5 border-white/20 text-white/60 hover:bg-white/10"
                }`}
              >
                <svg
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z"
                  />
                </svg>
                <span className="text-sm font-sans font-medium">Female</span>
              </button>
            </div>
          </div>

          {/* Text Buffer Size */}
          <div>
            <label className="block text-white/80 text-sm font-sans font-medium mb-2">
              Text Buffer Size
            </label>
            <div className="flex gap-2">
              {BUFFER_SIZE_OPTIONS.map((size) => (
                <button
                  key={size}
                  onClick={() => setBufferSize(size)}
                  className={`flex-1 px-3 py-2.5 rounded-lg border text-sm font-sans font-medium transition-colors ${
                    bufferSize === size
                      ? "bg-brand-primary/20 border-brand-primary text-white"
                      : "bg-white/5 border-white/20 text-white/60 hover:bg-white/10"
                  }`}
                >
                  {size}
                </button>
              ))}
            </div>
            <p className="text-white/40 text-xs font-sans mt-1.5">
              Number of words to buffer before processing
            </p>
          </div>
        </div>

        {/* Buttons */}
        <div className="flex gap-3 mt-6">
          <button
            onClick={resetToDefault}
            className="flex-1 px-5 py-3 bg-white/5 border border-white/20 text-white font-sans text-sm font-medium rounded-full hover:bg-white/10 transition-colors"
          >
            Reset to Default
          </button>
          <button
            onClick={onClose}
            className="flex-1 px-5 py-3 bg-white text-black font-sans text-sm font-semibold rounded-full hover:bg-white/90 transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
