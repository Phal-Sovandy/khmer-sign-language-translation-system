import { useState } from "react";
import { ControlButtons } from "./ControlButtons";

// Detected text output area
export function DetectedTextSection({
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
