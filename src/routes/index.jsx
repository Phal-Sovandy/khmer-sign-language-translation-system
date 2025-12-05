import { lazy, Suspense } from "react";
import { Routes, Route } from "react-router-dom";
import ScrollToTop from "../components/ScrollToTop";

// Lazy load pages for better performance
const App = lazy(() => import("../App"));
const DemoPage = lazy(() => import("../pages/demo"));
const AboutPage = lazy(() => import("../pages/AboutPage"));
const DocumentationPage = lazy(() => import("../pages/DocumentationPage"));
const ChangelogPage = lazy(() => import("../pages/ChangelogPage"));
const ContactPage = lazy(() => import("../pages/ContactPage"));

// Loading component
const LoadingFallback = () => (
  <div className="min-h-screen bg-brand-background flex items-center justify-center">
    <div className="text-white/60 font-sans">Loading...</div>
  </div>
);

export function AppRoutes() {
  return (
    <>
      <ScrollToTop />
      <Suspense fallback={<LoadingFallback />}>
        <Routes>
          <Route path="/" element={<App />} />
          <Route path="/demo" element={<DemoPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/documentation" element={<DocumentationPage />} />
          <Route path="/changelog" element={<ChangelogPage />} />
          <Route path="/contact" element={<ContactPage />} />
        </Routes>
      </Suspense>
    </>
  );
}
