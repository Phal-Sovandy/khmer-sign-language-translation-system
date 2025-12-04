import { useState, useRef, useEffect, useCallback } from "react";
import Header from "../components/layout/Header";
import Footer from "../components/layout/Footer";
import { SimpleTooltip } from "../components/ui/Tooltip";
import { SEO } from "../components/seo";
// Load MediaPipe using dynamic imports with error handling
const loadMediaPipe = async () => {
  try {
    // Dynamic import from node_modules
    const handsModule = await import("@mediapipe/hands");
    const drawingUtilsModule = await import("@mediapipe/drawing_utils");

    // Handle different export formats
    const Hands = handsModule.Hands || handsModule.default?.Hands;
    const HAND_CONNECTIONS =
      handsModule.HAND_CONNECTIONS || handsModule.default?.HAND_CONNECTIONS;
    const drawConnectors =
      drawingUtilsModule.drawConnectors ||
      drawingUtilsModule.default?.drawConnectors;
    const drawLandmarks =
      drawingUtilsModule.drawLandmarks ||
      drawingUtilsModule.default?.drawLandmarks;

    if (!Hands || !HAND_CONNECTIONS || !drawConnectors || !drawLandmarks) {
      throw new Error("MediaPipe modules not properly exported");
    }

    return {
      Hands,
      HAND_CONNECTIONS,
      drawConnectors,
      drawLandmarks,
    };
  } catch (error) {
    console.error("Error loading MediaPipe:", error);
    // Try CDN fallback
    try {
      const handsUrl =
        "https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469404/hands.js";
      const drawingUrl =
        "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.3.1620248257/drawing_utils.js";

      // Inject script tags as fallback
      const loadScript = (url) => {
        return new Promise((resolve, reject) => {
          if (document.querySelector(`script[src="${url}"]`)) {
            resolve();
            return;
          }
          const script = document.createElement("script");
          script.src = url;
          script.type = "module";
          script.onload = resolve;
          script.onerror = reject;
          document.head.appendChild(script);
        });
      };

      await Promise.all([loadScript(handsUrl), loadScript(drawingUrl)]);
      await new Promise((resolve) => setTimeout(resolve, 500));

      if (window.Hands && window.HAND_CONNECTIONS) {
        return {
          Hands: window.Hands,
          HAND_CONNECTIONS: window.HAND_CONNECTIONS,
          drawConnectors: window.drawConnectors,
          drawLandmarks: window.drawLandmarks,
        };
      }
    } catch (cdnError) {
      console.error("CDN fallback also failed:", cdnError);
    }

    throw new Error("Failed to load MediaPipe. Please refresh the page.");
  }
};

// Default settings values
const DEFAULT_SETTINGS = {
  sensitivity: 70,
  autoSpeak: false,
  mirrorVideo: true,
  camera: "user",
  voiceGender: "female",
  bufferSize: 5,
};

// Buffer size options
const BUFFER_SIZE_OPTIONS = [3, 5, 10, 15, 20];

// Settings Modal
function SettingsModal({ isOpen, onClose }) {
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

// Video feed component with real-time camera and hand landmark detection
function VideoFeed({ videoRef, isCameraActive, error, onRetry, canvasRef }) {
  return (
    <div
      className={`relative rounded-xl overflow-hidden bg-black transition-colors ${
        isCameraActive
          ? "border-2 border-[#3B82F6]/50"
          : "border border-white/10"
      }`}
    >
      {/* Camera feed */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full aspect-video object-cover"
        style={{ transform: "scaleX(-1)" }}
      />

      {/* Canvas overlay for hand landmarks */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />

      {/* Camera off state */}
      {!isCameraActive && !error && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black">
          <svg
            className="w-16 h-16 text-white/20 mb-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z"
            />
          </svg>
          <p className="text-white/40 font-sans text-sm">Camera is off</p>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/90">
          <svg
            className="w-16 h-16 text-white/30 mb-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M12 18.75H4.5a2.25 2.25 0 01-2.25-2.25V9m12.841 9.091L16.5 19.5m-1.409-1.409c.407-.407.659-.97.659-1.591v-9a2.25 2.25 0 00-2.25-2.25h-9c-.621 0-1.184.252-1.591.659m12.182 12.182L2.909 5.909M1.5 4.5l1.409 1.409"
            />
          </svg>
          <p className="text-white/60 font-sans text-sm text-center max-w-xs">
            Camera access denied or unavailable.
            <br />
            Please allow camera access to use the demo.
          </p>
          <button
            onClick={onRetry}
            className="mt-4 px-5 py-2 bg-white/10 border border-white/20 rounded-lg text-white text-sm font-medium hover:bg-white/20 transition-colors"
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
}

// Control buttons for the detected text area
function ControlButtons({
  isCameraActive,
  onToggleCamera,
  onOpenSettings,
  onSpeak,
  onReset,
  isSpeaking,
}) {
  return (
    <div className="flex items-center gap-2">
      {/* Camera toggle */}
      <SimpleTooltip
        content={
          isCameraActive
            ? "Turn off camera (Press S)"
            : "Turn on camera (Press S)"
        }
        position="bottom"
      >
        <button
          onClick={onToggleCamera}
          className={`p-2 rounded-lg border transition-colors ${
            isCameraActive
              ? "bg-brand-primary/20 border-brand-primary/50 text-brand-primary"
              : "bg-white/5 border-white/10 text-white/60 hover:text-white hover:bg-white/10"
          }`}
        >
          {isCameraActive ? (
            <svg
              className="w-5 h-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z"
              />
            </svg>
          ) : (
            <svg
              className="w-5 h-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M12 18.75H4.5a2.25 2.25 0 01-2.25-2.25V9m12.841 9.091L16.5 19.5m-1.409-1.409c.407-.407.659-.97.659-1.591v-9a2.25 2.25 0 00-2.25-2.25h-9c-.621 0-1.184.252-1.591.659m12.182 12.182L2.909 5.909M1.5 4.5l1.409 1.409"
              />
            </svg>
          )}
        </button>
      </SimpleTooltip>

      {/* Settings */}
      <SimpleTooltip content="Settings" position="bottom">
        <button
          onClick={onOpenSettings}
          className="p-2 rounded-lg bg-white/5 border border-white/10 text-white/60 hover:text-white hover:bg-white/10 transition-colors"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
        </button>
      </SimpleTooltip>

      {/* Speaker/Audio */}
      <SimpleTooltip
        content={isSpeaking ? "Stop speaking" : "Read aloud"}
        position="bottom"
      >
        <button
          onClick={onSpeak}
          className={`p-2 rounded-lg border transition-colors ${
            isSpeaking
              ? "bg-green-500/20 border-green-500/50 text-green-400"
              : "bg-white/5 border-white/10 text-white/60 hover:text-white hover:bg-white/10"
          }`}
        >
          {isSpeaking ? (
            <svg
              className="w-5 h-5 animate-pulse"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-4.72a.75.75 0 011.28.53v15.88a.75.75 0 01-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 012.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75z"
              />
            </svg>
          ) : (
            <svg
              className="w-5 h-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-4.72a.75.75 0 011.28.53v15.88a.75.75 0 01-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 012.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75z"
              />
            </svg>
          )}
        </button>
      </SimpleTooltip>

      {/* Refresh/Reset */}
      <SimpleTooltip content="Clear text" position="bottom">
        <button
          onClick={onReset}
          className="p-2 rounded-lg bg-white/5 border border-white/10 text-white/60 hover:text-white hover:bg-white/10 transition-colors"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99"
            />
          </svg>
        </button>
      </SimpleTooltip>

      {/* Report Issue */}
      <SimpleTooltip content="Report Issue" position="bottom">
        <a
          href="https://github.com/Phal-Sovandy/Khmer-Sign-Language-Translation-System/issues/new"
          target="_blank"
          rel="noopener noreferrer"
          className="p-2 rounded-lg bg-white/5 border border-white/10 text-white/60 hover:text-white hover:bg-white/10 transition-colors inline-flex items-center"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
            />
          </svg>
        </a>
      </SimpleTooltip>
    </div>
  );
}

// Detected text output area
function DetectedTextArea({
  text,
  isCameraActive,
  onToggleCamera,
  onOpenSettings,
  onSpeak,
  onReset,
  isSpeaking,
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="mt-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-semibold font-sans">Detected Text</h3>
        <ControlButtons
          isCameraActive={isCameraActive}
          onToggleCamera={onToggleCamera}
          onOpenSettings={onOpenSettings}
          onSpeak={onSpeak}
          onReset={onReset}
          isSpeaking={isSpeaking}
        />
      </div>

      {/* Text Area */}
      <div className="relative bg-[#0a0a0a] border border-white/10 rounded-xl p-4 min-h-[180px] max-h-[500px] resize-y overflow-auto">
        {/* Copy button */}
        <button
          onClick={handleCopy}
          disabled={!text}
          className="absolute top-3 right-3 inline-flex items-center gap-1.5 px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-white/60 text-xs font-medium hover:bg-white/10 hover:text-white transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
        >
          {copied ? (
            <>
              <svg
                className="w-3.5 h-3.5 text-green-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M5 13l4 4L19 7"
                />
              </svg>
              Copied!
            </>
          ) : (
            <>
              <svg
                className="w-3.5 h-3.5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M15.666 3.888A2.25 2.25 0 0013.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 01-.75.75H9a.75.75 0 01-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 011.927-.184"
                />
              </svg>
              Copy
            </>
          )}
        </button>

        {/* Detected text */}
        {text ? (
          <p className="text-white/80 font-khmer text-lg leading-relaxed pr-20">
            {text}
          </p>
        ) : (
          <p className="text-white/30 font-sans text-sm italic">
            {isCameraActive
              ? "Waiting for sign language detection..."
              : "Turn on the camera to start detecting sign language."}
          </p>
        )}
      </div>
    </div>
  );
}

export default function DemoPage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handsRef = useRef(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [detectedText, setDetectedText] = useState(
    "សួស្តីខ្ញុំឈ្មោះវ៉ាន់ វាត់ ខ្ញុំមានអាយុ ១២ឆ្នាំ"
  );
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // Initialize MediaPipe Hands
  const initializeHands = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;

    try {
      // Load MediaPipe modules
      const { Hands, HAND_CONNECTIONS, drawConnectors, drawLandmarks } =
        await loadMediaPipe();

      // Clean up existing hands instance
      if (handsRef.current) {
        handsRef.current.close();
      }

      const hands = new Hands({
        locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        },
      });

      hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      // Store drawing functions in closure
      const drawHands = (results) => {
        if (!canvasRef.current || !videoRef.current) return;

        const canvasCtx = canvasRef.current.getContext("2d");
        const video = videoRef.current;

        // Set canvas dimensions to match video
        if (video.videoWidth && video.videoHeight) {
          canvasRef.current.width = video.videoWidth;
          canvasRef.current.height = video.videoHeight;
        }

        // Clear canvas
        canvasCtx.save();
        canvasCtx.clearRect(
          0,
          0,
          canvasRef.current.width,
          canvasRef.current.height
        );

        // Mirror the canvas context to match the mirrored video
        canvasCtx.scale(-1, 1);
        canvasCtx.translate(-canvasRef.current.width, 0);

        // Draw hand landmarks
        if (results.multiHandLandmarks) {
          for (const landmarks of results.multiHandLandmarks) {
            // Draw connections (hand skeleton)
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
              color: "#ffffff",
              lineWidth: 3,
            });
            // Draw landmarks (hand joints)
            drawLandmarks(canvasCtx, landmarks, {
              color: "#0BA6DF",
              lineWidth: 1,
              radius: 5,
            });
          }
        }

        canvasCtx.restore();
      };

      hands.onResults(drawHands);

      handsRef.current = hands;

      // Process video frames
      const processFrame = async () => {
        if (
          videoRef.current &&
          handsRef.current &&
          videoRef.current.readyState === 4
        ) {
          try {
            await handsRef.current.send({ image: videoRef.current });
          } catch (error) {
            console.error("Error processing frame:", error);
          }
        }
        if (isCameraActive) {
          requestAnimationFrame(processFrame);
        }
      };

      // Start processing frames when video is ready
      if (videoRef.current.readyState >= 2) {
        processFrame();
      } else {
        videoRef.current.addEventListener("loadeddata", () => {
          processFrame();
        });
      }
    } catch (error) {
      console.error("Error initializing MediaPipe Hands:", error);
      setCameraError("Failed to initialize hand detection");
    }
  }, [isCameraActive]);

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
        setCameraError(null);

        // Wait for video to be ready, then initialize hands
        const handleLoadedMetadata = () => {
          // Small delay to ensure video is fully ready
          setTimeout(() => {
            initializeHands();
          }, 100);
          videoRef.current.removeEventListener(
            "loadedmetadata",
            handleLoadedMetadata
          );
        };

        videoRef.current.addEventListener(
          "loadedmetadata",
          handleLoadedMetadata
        );

        // If already loaded
        if (videoRef.current.readyState >= 2) {
          handleLoadedMetadata();
        }
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      setCameraError(err.message);
      setIsCameraActive(false);
    }
  }, [initializeHands]);

  // Stop camera
  const stopCamera = useCallback(() => {
    // Clean up MediaPipe hands
    if (handsRef.current) {
      handsRef.current.close();
      handsRef.current = null;
    }

    // Stop video stream
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    // Clear canvas
    if (canvasRef.current) {
      const canvasCtx = canvasRef.current.getContext("2d");
      if (canvasCtx) {
        canvasCtx.clearRect(
          0,
          0,
          canvasRef.current.width,
          canvasRef.current.height
        );
      }
    }

    setIsCameraActive(false);
  }, []);

  // Toggle camera
  const toggleCamera = useCallback(() => {
    if (isCameraActive) {
      stopCamera();
    } else {
      startCamera();
    }
  }, [isCameraActive, startCamera, stopCamera]);

  // Text-to-speech
  const speakText = useCallback(() => {
    if (!detectedText) return;

    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    if (isSpeaking) {
      setIsSpeaking(false);
      return;
    }

    const utterance = new SpeechSynthesisUtterance(detectedText);
    utterance.lang = "km-KH"; // Khmer language
    utterance.rate = 0.9;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
  }, [detectedText, isSpeaking]);

  // Reset/clear text
  const resetText = useCallback(() => {
    setDetectedText("");
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  }, []);

  // Keyboard shortcut for toggling camera (S key)
  useEffect(() => {
    const handleKeyPress = (event) => {
      // Only trigger if not typing in an input field
      if (
        event.target.tagName === "INPUT" ||
        event.target.tagName === "TEXTAREA" ||
        event.target.isContentEditable
      ) {
        return;
      }

      // Press 'S' or 's' to toggle camera
      if (event.key === "s" || event.key === "S") {
        event.preventDefault();
        toggleCamera();
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => {
      window.removeEventListener("keydown", handleKeyPress);
    };
  }, [toggleCamera]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
      window.speechSynthesis.cancel();

      // Clean up hands instance
      if (handsRef.current) {
        handsRef.current.close();
        handsRef.current = null;
      }
    };
  }, [stopCamera]);

  return (
    <div className="min-h-screen bg-brand-background overflow-x-clip">
      <SEO
        title="Demo"
        url="/demo"
        keywords="sign language demo, try sign language translator, real-time gesture recognition, webcam translation, test sign language, live demo, gesture detection, Khmer sign language demo, interactive translator, sign language test"
      />
      {/* Header */}
      <Header showDemoButton={false} />

      {/* Main Content */}
      <main className="px-8 pt-28 pb-16 max-w-[1200px] mx-auto">
        {/* Video Feed */}
        <VideoFeed
          videoRef={videoRef}
          canvasRef={canvasRef}
          isCameraActive={isCameraActive}
          error={cameraError}
          onRetry={startCamera}
        />

        {/* Detected Text Output */}
        <DetectedTextArea
          text={detectedText}
          isCameraActive={isCameraActive}
          onToggleCamera={toggleCamera}
          onOpenSettings={() => setIsSettingsOpen(true)}
          onSpeak={speakText}
          onReset={resetText}
          isSpeaking={isSpeaking}
        />
      </main>

      {/* Footer */}
      <Footer />

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
      />
    </div>
  );
}
