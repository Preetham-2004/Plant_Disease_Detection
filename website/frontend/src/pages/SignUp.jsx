import { useState } from "react";
import {
  createUserWithEmailAndPassword,
  GoogleAuthProvider,
  signInWithRedirect,
  signInWithPopup,
  updateProfile,
} from "firebase/auth";

import { auth, firebaseConfigError, isFirebaseConfigured } from "../lib/firebase";

function SignUp({ onAuthSuccess, onSwitchToSignIn }) {
  const [formData, setFormData] = useState({ name: "", email: "", password: "" });
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

      const credentials = await createUserWithEmailAndPassword(
        auth,
        formData.email.trim(),
        formData.password,
      );
      await updateProfile(credentials.user, {
        displayName: formData.name.trim(),
      });

      onAuthSuccess({
        message: "Account created successfully.",
        name: formData.name.trim(),
        email: credentials.user.email || formData.email.trim(),
      });
    } catch (requestError) {
      setError(requestError.message || "Sign up failed.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleGoogleSignUp() {
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
      setError(requestError.message || "Google sign up failed.");
    } finally {
      setIsGoogleSubmitting(false);
    }
  }

  return (
    <section className="auth-page">
      <div className="auth-card auth-card-signup">
        <div className="auth-showcase">
          <div className="auth-showcase-copy">
            <span className="eyebrow">Create Account</span>
            <h2>Bring diagnosis, treatment, and crop care into one calm workspace.</h2>
            <p>
              Create your account to keep PlantGuard ready for secure sign-ins and a smoother image analysis experience.
            </p>
          </div>

          <div className="auth-showcase-frame">
            <img src="/images/healthy-leaf.png" alt="Healthy green leaf close-up" />
            <div className="auth-showcase-badge">Fresh, focused analysis</div>
          </div>

          <div className="auth-showcase-metrics">
            <div className="auth-metric">
              <strong>Secure storage</strong>
              <span>Credentials are saved with Firebase Authentication</span>
            </div>
            <div className="auth-metric">
              <strong>Fast workflow</strong>
              <span>Jump from sign-in to upload in seconds</span>
            </div>
          </div>
        </div>

        <div className="auth-form-panel">
          <div className="auth-intro">
            <span className="eyebrow">Join PlantGuard</span>
            <h1>Sign up for PlantGuard AI</h1>
            <p className="hero-text">
              Create a secure account so your credentials and diagnosis history stay available across devices.
            </p>
          </div>

          <form className="auth-form" onSubmit={handleSubmit}>
            <button
              className="google-auth-button"
              type="button"
              onClick={handleGoogleSignUp}
              disabled={isGoogleSubmitting || isSubmitting}
            >
              {isGoogleSubmitting ? "Connecting Google..." : "Continue with Google"}
            </button>

            <div className="auth-divider">
              <span>or create with email</span>
            </div>

            <label className="auth-field">
              <span>Full name</span>
              <input
                type="text"
                name="name"
                value={formData.name}
                onChange={handleChange}
                placeholder="john Doe"
                required
              />
            </label>

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
                placeholder="Minimum 6 characters"
                minLength="6"
                required
              />
            </label>

            {error && <p className="error-banner">{error}</p>}

            <button className="primary-button auth-submit" type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Creating Account..." : "Create Account"}
            </button>
          </form>

          <p className="auth-switch">
            Already have an account?
            <button type="button" className="auth-switch-link" onClick={onSwitchToSignIn}>
              Sign in instead
            </button>
          </p>
        </div>
      </div>
    </section>
  );
}

export default SignUp;
