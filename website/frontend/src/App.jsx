import { useEffect, useState } from "react";
import { onAuthStateChanged, signOut } from "firebase/auth";

import SignIn from "./pages/SignIn";
import SignUp from "./pages/SignUp";
import { auth, firebaseConfigError, isFirebaseConfigured } from "./lib/firebase";
import { saveAnalysisRecord, subscribeToHistory } from "./lib/historyStore";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const emptyResult = null;
const LANGUAGE_OPTIONS = [
  "English",
  "Hindi",
  "Kannada",
  "Tamil",
  "Telugu",
  "Malayalam",
  "Marathi",
  "Bengali",
  "Gujarati",
  "Punjabi",
  "Odia",
];
const NAV_ITEMS = ["Diagnose", "History", "About"];

function parseTreatmentPlan(plan) {
  if (!plan) return [];

  const sections = [];
  let currentSection = null;

  for (const rawLine of plan.split("\n")) {
    const line = rawLine.trim();
    if (!line) continue;

    if (line.endsWith(":") && !line.startsWith("-")) {
      currentSection = { title: line.slice(0, -1), items: [] };
      sections.push(currentSection);
      continue;
    }

    const colonIndex = line.indexOf(":");
    if (colonIndex > 0 && !line.startsWith("-")) {
      const title = line.slice(0, colonIndex).trim();
      const value = line.slice(colonIndex + 1).trim();
      currentSection = { title, items: value ? [value] : [] };
      sections.push(currentSection);
      continue;
    }

    if (line.startsWith("-")) {
      const bullet = line.replace(/^-+\s*/, "").trim();
      if (!currentSection) {
        currentSection = { title: "Recommendation", items: [] };
        sections.push(currentSection);
      }
      if (bullet) currentSection.items.push(bullet);
      continue;
    }

    if (!currentSection) {
      currentSection = { title: "Recommendation", items: [] };
      sections.push(currentSection);
    }
    currentSection.items.push(line);
  }

  return sections;
}

function formatHistoryDate(timestamp) {
  return new Intl.DateTimeFormat("en-IN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(timestamp));
}

function App() {
  const [activePage, setActivePage] = useState("Diagnose");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(emptyResult);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  // Track selected region and treatment plans per region
  const [selectedRegionIndex, setSelectedRegionIndex] = useState(0);
  const [treatmentPlans, setTreatmentPlans] = useState({});
  const [isGeneratingPlan, setIsGeneratingPlan] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState("English");
  const [historyItems, setHistoryItems] = useState([]);
  const [currentUser, setCurrentUser] = useState(null);
  const [authMessage, setAuthMessage] = useState("");

  const currentRegion = result && result.regions?.length > 0 ? result.regions[selectedRegionIndex] : null;
  const headline = currentRegion ? currentRegion.disease_label : "Plant Health";
  const confidencePercent = currentRegion ? (currentRegion.disease_confidence * 100).toFixed(1) : null;
  const treatmentSections = parseTreatmentPlan(treatmentPlans[selectedRegionIndex] || "");

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    if (!isFirebaseConfigured || !auth) {
      setCurrentUser(null);
      setHistoryItems([]);
      return () => {};
    }

    return onAuthStateChanged(auth, (user) => {
      if (!user) {
        setCurrentUser(null);
        setHistoryItems([]);
        return;
      }

      setCurrentUser({
        uid: user.uid,
        name: user.displayName || user.email || "PlantGuard User",
        email: user.email || "",
      });
    });
  }, []);

  useEffect(() => {
    if (!currentUser?.uid) {
      setHistoryItems([]);
      return () => {};
    }

    return subscribeToHistory(
      currentUser.uid,
      setHistoryItems,
      (historyError) => {
        setError(historyError.message || "Unable to load your saved diagnosis history.");
      },
    );
  }, [currentUser?.uid]);

  function redirectToSignIn(message = "Please sign in or create an account to upload and analyze images.") {
    setError("");
    setAuthMessage(isFirebaseConfigured ? message : firebaseConfigError);
    setActivePage("Sign In");
  }

  function handleFileChange(event) {
    const nextFile = event.target.files?.[0] || null;

    if (nextFile && !currentUser) {
      event.target.value = "";
      redirectToSignIn();
      return;
    }

    setFile(nextFile);
    setResult(emptyResult);
    setTreatmentPlans({});
    setSelectedRegionIndex(0);
    setError("");

    if (!nextFile) {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl("");
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(nextFile));
  }

  async function fetchTreatmentPlan(index, disease_label, selected_part, language = selectedLanguage) {
    setIsGeneratingPlan(true);
    try {
      const treatmentRes = await fetch(`${API_BASE_URL}/api/treatment`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          disease_label,
          selected_part,
          language,
        }),
      });
      if (treatmentRes.ok) {
        const treatmentData = await treatmentRes.json();
        setTreatmentPlans((prev) => ({ ...prev, [index]: treatmentData.treatment_plan }));
        return treatmentData.treatment_plan;
      } else {
        setTreatmentPlans((prev) => ({ ...prev, [index]: "Failed to load personalized treatment plan." }));
        return "Failed to load personalized treatment plan.";
      }
    } catch {
      setTreatmentPlans((prev) => ({ ...prev, [index]: "Error generating treatment plan." }));
      return "Error generating treatment plan.";
    } finally {
      setIsGeneratingPlan(false);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (!currentUser) {
      redirectToSignIn();
      return;
    }

    if (!file) {
      setError("Please select an image before starting analysis.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setIsSubmitting(true);
    setError("");

    try {
      const endpoint = "/api/predict-ensemble";
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Inference request failed.");
      }

      setResult(payload);
      setTreatmentPlans({});
      setSelectedRegionIndex(0);
      setActivePage("Diagnose");
      const primaryRegion = payload.regions && payload.regions.length > 0 ? payload.regions[0] : null;
      let primaryTreatmentPlan = "";

      if (primaryRegion && primaryRegion.disease_label !== "Uncertain") {
        primaryTreatmentPlan = await fetchTreatmentPlan(
          0,
          primaryRegion.disease_label,
          primaryRegion.selected_part,
          selectedLanguage,
        );
      }

      try {
        await saveAnalysisRecord({
          language: selectedLanguage,
          result: payload,
          treatmentPlan: primaryTreatmentPlan,
          user: {
            uid: currentUser.uid,
            email: currentUser.email,
            displayName: currentUser.name,
          },
        });
      } catch (storageError) {
        setAuthMessage(storageError.message || "Analysis finished, but saving to Firebase failed.");
      }
    } catch (requestError) {
      setError(requestError.message);
      setResult(emptyResult);
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleRegionChange(e) {
    const newIdx = parseInt(e.target.value, 10);
    setSelectedRegionIndex(newIdx);
    const region = result.regions[newIdx];
    if (!treatmentPlans[newIdx] && region && region.disease_label !== "Uncertain") {
      fetchTreatmentPlan(newIdx, region.disease_label, region.selected_part, selectedLanguage);
    }
  }

  function handleLanguageChange(event) {
    const nextLanguage = event.target.value;
    setSelectedLanguage(nextLanguage);
    setTreatmentPlans({});

    if (currentRegion && currentRegion.disease_label !== "Uncertain") {
      fetchTreatmentPlan(
        selectedRegionIndex,
        currentRegion.disease_label,
        currentRegion.selected_part,
        nextLanguage,
      );
    }
  }

  function handleAuthSuccess(payload) {
    setCurrentUser((prev) => ({
      uid: prev?.uid || auth?.currentUser?.uid || "",
      name: payload.name,
      email: payload.email,
    }));
    setAuthMessage(payload.message);
    setError("");
    setActivePage("Diagnose");
  }

  async function handleLogout() {
    try {
      if (!auth) {
        setCurrentUser(null);
        setAuthMessage(isFirebaseConfigured ? "You have been signed out." : firebaseConfigError);
        setActivePage("Diagnose");
        return;
      }

      await signOut(auth);
      setAuthMessage("You have been signed out.");
      setActivePage("Diagnose");
    } catch (signOutError) {
      setError(signOutError.message || "Unable to sign out right now.");
    }
  }

  const ringRadius = 30;
  const ringCircumference = 2 * Math.PI * ringRadius;
  const ringOffset = currentRegion
    ? ringCircumference - currentRegion.disease_confidence * ringCircumference
    : ringCircumference;

  return (
    <div className="page-shell">
      <nav className="navbar">
        <div className="nav-brand">
          <div className="nav-logo" aria-hidden="true">
            <span className="nav-logo-core" />
            <span className="nav-logo-leaf nav-logo-leaf-a" />
            <span className="nav-logo-leaf nav-logo-leaf-b" />
          </div>
          <span className="nav-brand-text">PlantGuard AI</span>
        </div>
        <div className="nav-links">
          {NAV_ITEMS.map((item) => (
            <button
              key={item}
              className={`nav-link ${activePage === item ? "active" : ""}`}
              onClick={() => setActivePage(item)}
            >
              {item}
            </button>
          ))}
        </div>
        <div className="nav-actions">
          {!currentUser ? (
            <>
              <button className="nav-auth-link" onClick={() => setActivePage("Sign In")}>
                Sign In
              </button>
              <button className="nav-auth-button" onClick={() => setActivePage("Sign Up")}>
                Sign Up
              </button>
            </>
          ) : (
            <>
              <div className="nav-user-pill">
                <span className="status-dot" />
                {currentUser.name}
              </div>
              <button className="nav-auth-link" onClick={handleLogout}>
                Logout
              </button>
            </>
          )}
          <div className="nav-status">
            <span className="status-dot" />
            Models Ready
          </div>
        </div>
      </nav>

      <main className="layout">
        {authMessage && <div className="auth-feedback-banner">{authMessage}</div>}

        {activePage === "Sign In" && (
          <SignIn
            onAuthSuccess={handleAuthSuccess}
            onSwitchToSignUp={() => setActivePage("Sign Up")}
          />
        )}

        {activePage === "Sign Up" && (
          <SignUp
            onAuthSuccess={handleAuthSuccess}
            onSwitchToSignIn={() => setActivePage("Sign In")}
          />
        )}

        {activePage === "History" && (
          <section className="info-page">
            <div className="info-page-hero">
              <span className="eyebrow">📚 Saved Diagnoses</span>
              <h1>Recent Analysis History</h1>
              <p className="hero-text">
                Review recent predictions, treatment plans, and image links saved in Firebase.
              </p>
            </div>

            {historyItems.length ? (
              <div className="history-grid">
                {historyItems.map((item) => (
                  <article className="history-card" key={item.id}>
                    {item.imageUrl ? (
                      <img
                        className="history-preview"
                        src={item.imageUrl}
                        alt={`${item.diseaseLabel} prediction preview`}
                      />
                    ) : (
                      <div className="empty-state" style={{ minHeight: 180 }}>
                        <p>No saved image URL</p>
                      </div>
                    )}
                    <div className="history-card-body">
                      <div className="history-card-top">
                        <span className="history-badge">{item.selectedPart}</span>
                        <span className="history-date">{formatHistoryDate(item.timestamp)}</span>
                      </div>
                      <h3>{item.plantName} · {item.diseaseLabel}</h3>
                      <p>{item.filename}</p>
                      <div className="history-metrics">
                        <span>{(item.confidence * 100).toFixed(1)}% confidence</span>
                        <span>{item.language}</span>
                      </div>
                      {item.treatmentPlan && <p>{item.treatmentPlan.slice(0, 140)}...</p>}
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <div className="empty-page-state">
                <h2>No analysis history yet</h2>
                <p>Run your first diagnosis and it will appear here automatically.</p>
              </div>
            )}
          </section>
        )}

        {activePage === "About" && (
          <section className="info-page info-page-about">
            <div className="info-page-hero about-hero">
              <span className="eyebrow">🌱 About PlantGuard AI</span>
              <h1>How the platform works</h1>
              <p className="hero-text">
                PlantGuard AI helps farmers and growers detect plant diseases, localize affected parts, and get treatment guidance in multiple Indian languages.
              </p>
            </div>

            <div className="about-grid">
              <article className="about-card">
                <h3>Diagnosis pipeline</h3>
                <p>YOLOv8 finds the affected plant part, InceptionV3 classifies the disease, and Gemini generates treatment guidance.</p>
              </article>
              <article className="about-card">
                <h3>Language support</h3>
                <p>Treatment plans can be generated in English, Hindi, Kannada, Tamil, Telugu, Malayalam, Marathi, Bengali, Gujarati, Punjabi, and Odia.</p>
              </article>
              <article className="about-card">
                <h3>Best results</h3>
                <p>Upload one clear plant part at a time in good lighting. Sharper photos improve both disease detection and treatment relevance.</p>
              </article>
              <article className="about-card">
                <h3>Important note</h3>
                <p>AI recommendations are decision support, not a replacement for local agronomy advice. Severe spread or uncertain results should be checked by an expert.</p>
              </article>
            </div>
          </section>
        )}

        {activePage === "Diagnose" && (
          <>
        <section className="hero-section">
          <div className="hero-copy">
            <span className="eyebrow">🔬 AI-Powered Plant Pathology</span>
            <h1>
              Diagnose <span className="gradient-text">{headline}</span> Instantly
            </h1>
            <p className="hero-text">
              Upload a photo of any plants leaf, fruit. Our
              multi-stage AI pipeline segments, classifies, and recommends
              specific treatments — all in seconds.
            </p>
            <div className="pipeline-steps">
              <span className="pipe-step">
                <span className="step-num">1</span> YOLOv8 Ensemble
              </span>
              <span className="pipe-step">
                <span className="step-num">2</span> InceptionV3
              </span>
              <span className="pipe-step">
                <span className="step-num">3</span> Gemini AI
              </span>
            </div>
          </div>

          <div className="hero-showcase">
            <div className="showcase-grid">
              <div className="showcase-card">
                <img
                  src="/images/healthy-leaf.png"
                  alt="Healthy green leaf with water droplets"
                />
                <div className="card-label">
                  <span className="dot dot-green" />
                  Healthy
                </div>
              </div>
              <div className="showcase-card">
                <img
                  src="/images/diseased-leaf.png"
                  alt="Diseased plant leaf with brown spots"
                />
                <div className="card-label">
                  <span className="dot dot-amber" />
                  Diseased
                </div>
              </div>
              <div className="showcase-card showcase-hero-card">
                <img
                  src="/images/hero-plants.png"
                  alt="Beautiful indoor plants on a shelf"
                />
                <div className="card-label">
                  <span className="dot dot-green" />
                  Keep your plants thriving
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="upload-card">
          <div className="upload-card-header">
            <div className="header-icon">📤</div>
            <h2>Upload & Analyze</h2>
          </div>

          <form className="upload-card-body" onSubmit={handleSubmit}>
            <label className="drop-zone" htmlFor="image-upload">
              <input
                id="image-upload"
                type="file"
                accept="image/png,image/jpeg,image/webp,image/bmp"
                onChange={handleFileChange}
              />
              <div className="drop-icon">🍃</div>
              <span className="drop-title">
                Drop or browse a plant image
              </span>
              <span className="drop-subtitle">
                JPG, PNG, WEBP, BMP — up to 10 MB
              </span>
              {file && <span className="file-badge">📎 {file.name}</span>}
            </label>

            <div className="upload-preview-area">
              {previewUrl ? (
                <img
                  className="upload-preview-img"
                  src={previewUrl}
                  alt="Selected plant preview"
                />
              ) : (
                <div className="empty-state" style={{ minHeight: 200 }}>
                  <p>Image preview will appear here</p>
                </div>
              )}

              <div className="btn-row">
                <button
                  className="primary-button"
                  type="submit"
                  disabled={isSubmitting}
                >
                  {isSubmitting ? "Analyzing…" : "Run Analysis"}
                </button>
              </div>

              {error && <p className="error-banner">{error}</p>}
            </div>
          </form>
        </section>

        {result && (
          <>
            {result.regions.length > 1 && (
              <div className="region-selector-container">
                <label htmlFor="region-select" className="region-label">Select Detected Region: </label>
                <select 
                  id="region-select" 
                  className="region-select" 
                  value={selectedRegionIndex} 
                  onChange={handleRegionChange}
                >
                  {result.regions.map((region, idx) => (
                    <option key={idx} value={idx}>
                      Region {idx + 1} - {region.class_name} (Confidence: {(region.confidence * 100).toFixed(0)}%)
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="section-label">Diagnosis Results</div>
            <section className="content-grid">
              
              <article className="panel">
                <div className="panel-heading">
                  <h2>
                    <span className="panel-icon">📊</span>
                    Inference Summary
                  </h2>
                  <p>{currentRegion?.advisory}</p>
                </div>

                <div className="summary-stack">
                  <div className="confidence-ring-wrap">
                    <div className="confidence-ring">
                      <svg width="72" height="72" viewBox="0 0 72 72">
                        <defs>
                          <linearGradient id="greenGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stopColor="#166534" />
                            <stop offset="50%" stopColor="#16a34a" />
                            <stop offset="100%" stopColor="#34d37a" />
                          </linearGradient>
                        </defs>
                        <circle className="confidence-ring-bg" cx="36" cy="36" r={ringRadius} />
                        <circle
                          className="confidence-ring-fill"
                          cx="36"
                          cy="36"
                          r={ringRadius}
                          strokeDasharray={ringCircumference}
                          strokeDashoffset={ringOffset}
                        />
                      </svg>
                      <div className="confidence-ring-text">
                        {confidencePercent}%
                      </div>
                    </div>
                    <div className="confidence-meta">
                      <span className="confidence-label">
                        {currentRegion?.disease_label}
                      </span>
                      <span className="confidence-sub">
                        {currentRegion?.selected_part} · {currentRegion?.segmentation_mode}
                      </span>
                    </div>
                  </div>

                  <div className="metric-row">
                    <span>Disease</span>
                    <strong>{currentRegion?.disease_label}</strong>
                  </div>
                  <div className="metric-row">
                    <span>Confidence</span>
                    <strong>{confidencePercent}%</strong>
                  </div>
                  <div className="metric-row">
                    <span>Selected Part</span>
                    <strong>{currentRegion?.selected_part}</strong>
                  </div>
                  <div className="metric-row">
                    <span>Segmentation</span>
                    <strong>{currentRegion?.segmentation_mode}</strong>
                  </div>
                  <div className="metric-row">
                    <span>Classifier Model</span>
                    <strong>InceptionV3</strong>
                  </div>
                  <div className="metric-row">
                    <span>Classifier Input</span>
                    <strong>
                      {result.input_resolution?.join(" × ")}
                    </strong>
                  </div>

                  <div className="probability-list">
                    {currentRegion?.probabilities.map((item) => (
                      <div className="probability-item" key={item.label}>
                        <div className="probability-meta">
                          <span>{item.label}</span>
                          <span>
                            {(item.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="probability-track">
                          <div
                            className="probability-bar"
                            style={{ width: `${item.confidence * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </article>

              <article className="panel">
                <div className="panel-heading">
                  <h2>
                    <span className="panel-icon">🔗</span>
                    Ensemble Detection Map
                  </h2>
                  <p>YOLOv8 Consensus Output</p>
                </div>

                <div className="summary-stack">
                  <div className="ensemble-stats-grid">
                    <div className="metric-row compact-metric">
                    <span>Total Detections</span>
                    <strong>{result.total_detections}</strong>
                  </div>
                  <div className="metric-row compact-metric">
                    <span>✅ Consensus (Both Models)</span>
                    <strong>{result.consensus_detections}</strong>
                  </div>
                  <div className="metric-row compact-metric">
                    <span>⚠️ Single Model</span>
                    <strong>{result.single_model_detections}</strong>
                  </div>
                  </div>

                  <div className="ensemble-info">
                    <h4>Detections by Class:</h4>
                    <div className="detection-list">
                      {result.regions.map((det, idx) => (
                        <div key={idx} className={`detection-item ${det.consensus ? "consensus" : "single"}`}>
                          <div className="detection-header">
                            <span className="class-badge">
                              {det.consensus ? "✅" : "⚠️"} {det.class_name}
                            </span>
                            <span className="confidence-badge">
                              {(det.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="detection-meta">
                            <span>Agreement {det.agreement}/2</span>
                            <span>Box {det.box.map((v) => v.toFixed(0)).join(", ")}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </article>
            </section>

            <div className="section-label">Visual Outputs</div>
            <section className="results-strip">
              <article className="panel wide-panel">
                <div className="panel-heading">
                  <h2>
                    <span className="panel-icon">🔍</span>
                    Localized Region
                  </h2>
                  <p>Crop sent to InceptionV3 from the selected ensemble box.</p>
                </div>
                {currentRegion?.segmented_image && (
                  <img
                    className="preview-image"
                    src={currentRegion.segmented_image}
                    alt="Segmented plant region"
                  />
                )}
              </article>

              <article className="panel wide-panel">
                <div className="panel-heading">
                  <h2>
                    <span className="panel-icon">🖼️</span>
                    Ensemble Overlay
                  </h2>
                  <p>Consensus and standalone bounding boxes.</p>
                </div>
                <img
                  className="preview-image"
                  src={result.annotated_image}
                  alt="Ensemble detection overlay"
                />
                <div className="chip-row">
                  {result.regions.map((det, idx) => (
                    <span
                      key={`${det.class_name}-${idx}-${det.box.join("-")}`}
                      className={det.consensus ? "chip chip-active" : "chip"}
                    >
                      {det.class_name} {(det.confidence * 100).toFixed(0)}%
                    </span>
                  ))}
                </div>
              </article>
            </section>

            <div className="section-label">AI Treatment Plan</div>
            <section className="treatment-strip">
              <article className="panel">
                <div className="panel-heading">
                  <h2>
                    <span className="panel-icon">💊</span>
                    Personalized Treatment
                  </h2>
                  <p>Disease-specific solutions via Gemini AI for the selected region.</p>
                </div>

                <div className="language-toolbar">
                  <label className="language-label" htmlFor="treatment-language">
                    Treatment language
                  </label>
                  <select
                    id="treatment-language"
                    className="language-select"
                    value={selectedLanguage}
                    onChange={handleLanguageChange}
                  >
                    {LANGUAGE_OPTIONS.map((language) => (
                      <option key={language} value={language}>
                        {language}
                      </option>
                    ))}
                  </select>
                </div>

                {isGeneratingPlan && !treatmentPlans[selectedRegionIndex] ? (
                  <div className="empty-state">
                    <div className="spinner-wrap">
                      <div className="spinner" />
                      <span className="spinner-text">
                        Generating treatment recommendations…
                      </span>
                    </div>
                  </div>
                ) : treatmentPlans[selectedRegionIndex] ? (
                  <div className="treatment-sheet">
                    <div className="treatment-sheet-header">
                      <div>
                        <p className="treatment-kicker">Detected Condition</p>
                        <h3>
                          {currentRegion?.disease_label} on {currentRegion?.selected_part}
                        </h3>
                      </div>
                      <div className="treatment-meta-badge">
                        {confidencePercent}% confidence
                      </div>
                    </div>

                    <div className="treatment-sheet-body">
                      {treatmentSections.map((section) => (
                        <section
                          className="treatment-section"
                          key={`${selectedRegionIndex}-${section.title}`}
                        >
                          <h4>{section.title}</h4>
                          <ul>
                            {section.items.map((item, idx) => (
                              <li key={`${section.title}-${idx}`}>{item}</li>
                            ))}
                          </ul>
                        </section>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="empty-state">
                    <p>No treatment plan available.</p>
                  </div>
                )}
              </article>
            </section>
          </>
        )}
          </>
        )}

        <footer className="footer">
          PlantGuard AI · YOLOv8 · InceptionV3 · Gemini · Built with 🌱
        </footer>
      </main>
    </div>
  );
}

export default App;
