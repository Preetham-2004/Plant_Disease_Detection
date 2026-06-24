import { useState } from "react";
import {
  GoogleAuthProvider,
  signInWithEmailAndPassword,
  signInWithRedirect,
  signInWithPopup,
} from "firebase/auth";

import { auth, firebaseConfigError, isFirebaseConfigured } from "../lib/firebase";

function SignIn({ onAuthSuccess, onSwitchToSignUp }) {
  const [formData, setFormData] = useState({ email: "", password: "" });
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isGoogleSubmitting, setIsGoogleSubmitting] = useState(false);

  function handleChange(event) {
    const { name, value } = event.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setIsSubmitting(true);

    try {
      if (!isFirebaseConfigured || !auth) {
        throw new Error(firebaseConfigError);
      }

      const credentials = await signInWithEmailAndPassword(
        auth,
        formData.email.trim(),
        formData.password,
      );

      onAuthSuccess({
        message: "Signed in successfully.",
        name: credentials.user.displayName || credentials.user.email || "PlantGuard User",
        email: credentials.user.email || formData.email.trim(),
      });
    } catch (requestError) {
      setError(requestError.message || "Sign in failed.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleGoogleSignIn() {
    setError("");
    setIsGoogleSubmitting(true);

    try {
      if (!isFirebaseConfigured || !auth) {
        throw new Error(firebaseConfigError);
      }

      const provider = new GoogleAuthProvider();
      let credentials;

      try {
        credentials = await signInWithPopup(auth, provider);
      } catch (popupError) {
        const shouldFallbackToRedirect = [
          "auth/popup-blocked",
          "auth/popup-closed-by-user",
          "auth/cancelled-popup-request",
        ].includes(popupError?.code);

        if (!shouldFallbackToRedirect) {
          throw popupError;
        }

        await signInWithRedirect(auth, provider);
        return;
      }

      onAuthSuccess({
        message: "Signed in with Google successfully.",
        name: credentials.user.displayName || credentials.user.email || "PlantGuard User",
        email: credentials.user.email || "",
      });
    } catch (requestError) {
      setError(requestError.message || "Google sign in failed.");
    } finally {
      setIsGoogleSubmitting(false);
    }
  }

  return (
    <section className="auth-page">
      <div className="auth-card auth-card-signin">
        <div className="auth-showcase">
          <div className="auth-showcase-copy">
            <span className="eyebrow">Secure Access</span>
            <h2>Healthy crops start with fast, confident diagnosis.</h2>
            <p>
              Sign in to keep your PlantGuard workspace ready for image analysis, treatment suggestions, and future personalized tools.
            </p>
          </div>

          <div className="auth-showcase-frame">
            <img src="/images/hero-plants.png" alt="Healthy plants arranged on shelves" />
            <div className="auth-showcase-badge">Trusted plant support</div>
          </div>

          <div className="auth-showcase-metrics">
            <div className="auth-metric">
              <strong>3-step AI</strong>
              <span>Detection, classification, guidance</span>
            </div>
            <div className="auth-metric">
              <strong>11 languages</strong>
              <span>Treatment help for Indian farmers</span>
            </div>
          </div>
        </div>

        <div className="auth-form-panel">
          <div className="auth-intro">
            <span className="eyebrow">Welcome Back</span>
            <h1>Sign in to PlantGuard AI</h1>
            <p className="hero-text">
              Access your plant analysis workspace and keep your account ready for faster diagnosis sessions.
            </p>
          </div>

          <form className="auth-form" onSubmit={handleSubmit}>
            <button
              className="google-auth-button"
              type="button"
              onClick={handleGoogleSignIn}
              disabled={isGoogleSubmitting || isSubmitting}
            >
              {isGoogleSubmitting ? "Connecting Google..." : "Continue with Google"}
            </button>

            <div className="auth-divider">
              <span>or use email</span>
            </div>

            <label className="auth-field">
              <span>Email address</span>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                placeholder="farmer@example.com"
                required
              />
            </label>

            <label className="auth-field">
              <span>Password</span>
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="Enter your password"
                required
              />
            </label>

            {error && <p className="error-banner">{error}</p>}

            <button className="primary-button auth-submit" type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Signing In..." : "Sign In"}
            </button>
          </form>

          <p className="auth-switch">
            New here?
            <button type="button" className="auth-switch-link" onClick={onSwitchToSignUp}>
              Create an account
            </button>
          </p>
        </div>
      </div>
    </section>
  );
}

export default SignIn;
