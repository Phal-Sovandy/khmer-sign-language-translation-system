import { useState } from "react";
import { Link } from "react-router-dom";
import TiltedCard from "../../components/visuals/TitledCard";
import MediaWrapper from "../../components/ui/MediaWrapper";
import demoQr from "../../assets/images/demo-qr.webp";

const DEMO_URL = "https://khmer-sign-language-translation-sys.vercel.app/demo";

// Card content component for the TiltedCard overlay
function DemoCard({ copied }) {
  return (
    <div className="w-full h-[280px] sm:h-[320px] md:h-[380px] bg-gradient-to-br from-[#1a1a1a] to-[#0d0d0d] rounded-2xl p-6 sm:p-8 md:p-10 border border-white/10 flex flex-col justify-between relative cursor-pointer">
      {/* Copied indicator */}
      {copied && (
        <div className="absolute top-4 right-4 flex items-center gap-1.5 bg-green-500/20 border border-green-500/30 text-green-400 text-xs px-3 py-1.5 rounded-full">
          <svg
            className="w-3.5 h-3.5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M5 13l4 4L19 7"
            />
          </svg>
          Copied!
        </div>
      )}

      {/* Top Section */}
      <div className="space-y-2 sm:space-y-3 md:space-y-4">
        <p className="text-xs sm:text-sm md:text-xl text-white/50 font-display tracking-widest uppercase">
          Khmer Sign Language Translation System
        </p>
        <h3 className="text-xl sm:text-2xl md:text-3xl font-bold text-white font-heading">
          Try it Now!
        </h3>
        <p className="text-xs sm:text-sm md:text-base text-white/60 font-sans leading-relaxed">
          Experience our system converting gestures into readable text and
          speech—fast, accurate, and effortless.
        </p>
      </div>

      {/* Bottom Section - Logo and QR */}
      <div className="flex items-end justify-between">
        {/* KSLTS Logo */}
        <span className="text-3xl sm:text-4xl md:text-6xl font-display text-white tracking-wider">
          KSLTS
        </span>

        {/* QR Code Placeholder */}
        <div className="flex flex-col items-center gap-1 sm:gap-2">
          <div className="w-16 h-16 sm:w-20 sm:h-20 md:w-28 md:h-28 bg-r rounded-lg p-1.5 sm:p-2 md:p-2.5 flex items-center justify-center">
            {/* QR Code Pattern */}
            <MediaWrapper src={demoQr} alt="QR Code for demo" />
          </div>
          <span className="text-xs sm:text-sm md:text-2xl text-white/50 font-display tracking-wider">
            Scan for Demo!
          </span>
        </div>
      </div>
    </div>
  );
}

export default function TryDemoSection() {
  const [copied, setCopied] = useState(false);

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(DEMO_URL);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  return (
    <section className="relative w-screen py-32 bg-brand-background overflow-hidden">
      {/* Background gradient */}
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute right-0 top-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-brand-primary/5 rounded-full blur-[150px]" />
      </div>

      <div className="relative z-10 flex flex-col lg:flex-row items-center justify-between gap-16 px-8 max-w-[1700px] mx-auto">
        {/* Left Content */}
        <div className="flex flex-col gap-6 max-w-lg">
          {/* Title */}
          <h2 className="font-heading text-[clamp(2.5rem,5vw,4rem)] font-semibold text-white">
            Try it Now!
          </h2>

          {/* Description */}
          <p className="text-white/60 font-sans leading-relaxed">
            Experience our Khmer Sign Language Translation System in action!
            Instantly translate gestures and speech into accurate, readable text
            and see how communication becomes effortless.
          </p>

          {/* Requirements */}
          <div className="space-y-3">
            <p className="text-white font-semibold font-sans">
              Requirements to try the demo:
            </p>
            <ul className="space-y-2 text-white/60 font-sans">
              <li className="flex items-start gap-2">
                <span className="text-white/50">•</span>A modern web browser
                (Chrome, Edge, Firefox, or Safari)
              </li>
              <li className="flex items-start gap-2">
                <span className="text-white/50">•</span>
                Webcam enabled for gesture capture
              </li>
              <li className="flex items-start gap-2">
                <span className="text-white/50">•</span>
                Stable internet connection for real-time processing
              </li>
            </ul>
          </div>

          {/* CTA Button */}
          <Link
            to="/demo"
            className="inline-flex items-center gap-2.5 w-fit px-7 py-3 bg-brand-primary hover:bg-brand-primary/90 rounded-full text-white font-sans text-sm font-semibold transition-colors shadow-[0_15px_45px_rgba(47,107,255,0.45)]"
          >
            Start a Demo
            <svg
              className="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z"
              />
            </svg>
          </Link>
        </div>

        {/* Right Content - TiltedCard */}
        <div
          onClick={handleCopyLink}
          className="w-full lg:flex-1 lg:max-w-[900px] xl:max-w-[1000px] mx-auto lg:mx-0"
        >
          <div className="w-full h-[280px] sm:h-[320px] md:h-[380px] lg:h-[420px]">
            <TiltedCard
              imageSrc=""
              captionText="Click to copy demo link"
              containerHeight="100%"
              containerWidth="100%"
              imageHeight="100%"
              imageWidth="100%"
              scaleOnHover={1.05}
              rotateAmplitude={12}
              showMobileWarning={false}
              showTooltip={true}
              displayOverlayContent={true}
              overlayContent={<DemoCard copied={copied} />}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
