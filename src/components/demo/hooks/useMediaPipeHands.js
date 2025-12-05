import { useRef, useEffect, useCallback } from "react";

// Custom drawing functions for hand landmarks
// Note: Canvas is mirrored via CSS, so we draw coordinates as-is (no flipping needed)
function drawConnectors(ctx, points, connections, style = {}) {
  ctx.strokeStyle = style.color || "#00FF00";
  ctx.lineWidth = style.lineWidth || 2;

  for (const connection of connections) {
    // Handle both [start, end] and {start, end} formats
    const startIdx = Array.isArray(connection)
      ? connection[0]
      : connection.start;
    const endIdx = Array.isArray(connection) ? connection[1] : connection.end;

    const start = points[startIdx];
    const end = points[endIdx];

    if (start && end) {
      // Draw coordinates as-is - CSS mirroring handles the flip
      const startX = start.x * ctx.canvas.width;
      const startY = start.y * ctx.canvas.height;
      const endX = end.x * ctx.canvas.width;
      const endY = end.y * ctx.canvas.height;

      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, endY);
      ctx.stroke();
    }
  }
}

function drawLandmarks(ctx, points, style = {}) {
  ctx.fillStyle = style.color || "#FF0000";
  ctx.strokeStyle = style.color || "#FF0000";
  ctx.lineWidth = style.lineWidth || 1;
  const radius = style.radius || 3;

  for (const point of points) {
    // Draw coordinates as-is - CSS mirroring handles the flip
    const x = point.x * ctx.canvas.width;
    const y = point.y * ctx.canvas.height;

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }
}

export function useMediaPipeHands(videoRef, isCameraActive) {
  const handsRef = useRef(null);
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const handConnectionsRef = useRef(null);

  const onResults = useCallback((results) => {
    if (!canvasRef.current || !videoRef.current || !handConnectionsRef.current)
      return;

    const canvasCtx = canvasRef.current.getContext("2d");
    const video = videoRef.current;

    // Set canvas dimensions to match video
    if (
      canvasRef.current.width !== video.videoWidth ||
      canvasRef.current.height !== video.videoHeight
    ) {
      canvasRef.current.width = video.videoWidth;
      canvasRef.current.height = video.videoHeight;
    }

    // Clear canvas
    canvasCtx.clearRect(
      0,
      0,
      canvasRef.current.width,
      canvasRef.current.height
    );

    // Draw hand landmarks (no flipping needed - CSS handles mirroring)
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      for (const landmarks of results.multiHandLandmarks) {
        // Draw connections
        drawConnectors(canvasCtx, landmarks, handConnectionsRef.current, {
          color: "#ffffff",
          lineWidth: 3,
        });

        // Draw landmarks
        drawLandmarks(canvasCtx, landmarks, {
          color: "#34b4eb",
          lineWidth: 1,
          radius: 5,
        });
      }
    }

    // Draw hand labels (left/right)
    // Swap labels because webcam is mirrored - Left appears as Right and vice versa
    if (results.multiHandedness && results.multiHandedness.length > 0) {
      for (let i = 0; i < results.multiHandedness.length; i++) {
        const hand = results.multiHandedness[i];
        const landmarks = results.multiHandLandmarks[i];
        if (landmarks && landmarks.length > 0) {
          // Draw coordinates as-is - CSS mirroring handles the flip
          const x = landmarks[0].x * canvasRef.current.width;
          const y = landmarks[0].y * canvasRef.current.height;

          // Swap Left/Right labels because the video feed is mirrored
          const swappedLabel = hand.label === "Left" ? "Right" : "Left";

          // Flip text back so it reads correctly (canvas is mirrored via CSS)
          canvasCtx.save();
          canvasCtx.scale(-1, 1);
          canvasCtx.translate(-canvasRef.current.width, 0);

          canvasCtx.fillStyle = "#00FF00";
          canvasCtx.font = "bold 16px Arial";
          canvasCtx.fillText(
            swappedLabel,
            canvasRef.current.width - x - 10,
            y - 10
          );
          canvasCtx.restore();
        }
      }
    }
  }, []);

  // Process video frames
  const processFrame = useCallback(async () => {
    if (
      !isCameraActive ||
      !videoRef.current ||
      !handsRef.current ||
      videoRef.current.readyState !== videoRef.current.HAVE_ENOUGH_DATA
    ) {
      return;
    }

    try {
      await handsRef.current.send({ image: videoRef.current });
    } catch (error) {
      console.error("Error processing frame:", error);
    }

    animationFrameRef.current = requestAnimationFrame(processFrame);
  }, [isCameraActive]);

  // Initialize MediaPipe Hands
  useEffect(() => {
    if (!isCameraActive) {
      // Clean up when camera is off
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      if (handsRef.current) {
        handsRef.current.close();
        handsRef.current = null;
      }
      return;
    }

    if (!videoRef.current) return;

    // Load MediaPipe Hands
    let isMounted = true;
    let startProcessingFn = null;

    const initializeHands = () => {
      if (!isMounted || !videoRef.current) return;

      // Check if Hands is already available (from script tag or previous load)
      if (window.Hands && window.HAND_CONNECTIONS) {
        createHandsInstance(window.Hands, window.HAND_CONNECTIONS);
        return;
      }

      // Check if script is already loading
      const existingScript = document.querySelector(
        "script[data-mediapipe-hands]"
      );
      if (existingScript) {
        // Wait for it to load
        existingScript.addEventListener("load", () => {
          if (window.Hands && window.HAND_CONNECTIONS) {
            createHandsInstance(window.Hands, window.HAND_CONNECTIONS);
          }
        });
        return;
      }

      // Load MediaPipe Hands from CDN via script tag
      const script = document.createElement("script");
      script.src = "https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js";
      script.async = true;
      script.setAttribute("data-mediapipe-hands", "true");

      script.onload = () => {
        // Define HAND_CONNECTIONS manually (21 hand landmarks connections)
        const HAND_CONNECTIONS_DATA = [
          [0, 1],
          [1, 2],
          [2, 3],
          [3, 4], // Thumb
          [0, 5],
          [5, 6],
          [6, 7],
          [7, 8], // Index finger
          [5, 9],
          [9, 10],
          [10, 11],
          [11, 12], // Middle finger
          [9, 13],
          [13, 14],
          [14, 15],
          [15, 16], // Ring finger
          [13, 17],
          [0, 17],
          [17, 18],
          [18, 19],
          [19, 20], // Pinky
        ];

        // Check multiple possible global variable names
        const HandsClass =
          window.Hands ||
          window.mediapipe?.hands?.Hands ||
          window.mp?.hands?.Hands;

        if (HandsClass && typeof HandsClass === "function") {
          createHandsInstance(HandsClass, HAND_CONNECTIONS_DATA);
        } else {
          console.error("Hands not found on window after script load");
          console.log("window.Hands:", window.Hands);
          console.log("window.mediapipe:", window.mediapipe);
          console.log("window.mp:", window.mp);
        }
      };

      script.onerror = () => {
        console.error("Failed to load MediaPipe Hands script from CDN");
      };

      document.head.appendChild(script);
    };

    const createHandsInstance = (HandsClass, HAND_CONNECTIONS_DATA) => {
      if (!isMounted || !videoRef.current) return;

      if (!HandsClass || typeof HandsClass !== "function") {
        console.error("Hands is not a constructor");
        return;
      }

      // Store HAND_CONNECTIONS
      if (HAND_CONNECTIONS_DATA) {
        handConnectionsRef.current = HAND_CONNECTIONS_DATA;
      }

      try {
        const hands = new HandsClass({
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

        hands.onResults(onResults);
        handsRef.current = hands;

        // Wait for video to be ready, then start processing
        startProcessingFn = () => {
          if (videoRef.current && videoRef.current.readyState >= 2) {
            processFrame();
          } else {
            // Retry after a short delay
            setTimeout(startProcessingFn, 100);
          }
        };

        // Start processing when video is loaded
        if (videoRef.current.readyState >= 2) {
          processFrame();
        } else {
          videoRef.current.addEventListener("loadeddata", startProcessingFn, {
            once: true,
          });
        }
      } catch (error) {
        console.error("Failed to load MediaPipe Hands:", error);
      }
    };

    initializeHands();

    return () => {
      isMounted = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      if (handsRef.current) {
        handsRef.current.close();
        handsRef.current = null;
      }
      if (videoRef.current && startProcessingFn) {
        videoRef.current.removeEventListener("loadeddata", startProcessingFn);
      }
    };
  }, [isCameraActive, onResults, processFrame]);

  return canvasRef;
}
