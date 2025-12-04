import { useMediaPipeHands } from "./hooks";

// Video feed component with real-time camera and hand landmarks
export function VideoFeedSection({ videoRef, isCameraActive, error, onRetry }) {
  const canvasRef = useMediaPipeHands(videoRef, isCameraActive);

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
      {isCameraActive && (
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
          style={{
            transform: "scaleX(-1)",
            objectFit: "cover",
          }}
        />
      )}

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
