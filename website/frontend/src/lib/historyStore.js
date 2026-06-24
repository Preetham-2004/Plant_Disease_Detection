import {
  addDoc,
  collection,
  limit,
  onSnapshot,
  orderBy,
  query,
  serverTimestamp,
} from "firebase/firestore";

import { db, firebaseConfigError, isFirebaseConfigured } from "./firebase";

function mapHistoryDoc(doc) {
  const data = doc.data();
  const createdAt = data.createdAt?.toDate?.() ?? null;

  return {
    id: doc.id,
    timestamp: createdAt ? createdAt.toISOString() : data.timestamp || new Date().toISOString(),
    filename: data.filename || "analysis-image",
    imageUrl: data.imageUrl || "",
    plantName: data.plantName || "Unknown plant",
    diseaseLabel: data.diseaseLabel || "Unknown",
    selectedPart: data.selectedPart || "Unknown part",
    confidence: data.confidence || 0,
    treatmentPlan: data.treatmentPlan || "",
    language: data.language || "English",
  };
}

export function subscribeToHistory(userId, onUpdate, onError) {
  if (!isFirebaseConfigured || !db) {
    onError?.(new Error(firebaseConfigError));
    onUpdate([]);
    return () => {};
  }

  const historyQuery = query(
    collection(db, "users", userId, "analysisHistory"),
    orderBy("createdAt", "desc"),
    limit(12),
  );

  return onSnapshot(
    historyQuery,
    (snapshot) => onUpdate(snapshot.docs.map(mapHistoryDoc)),
    (error) => onError?.(error),
  );
}

export async function saveAnalysisRecord({
  language,
  result,
  treatmentPlan,
  user,
}) {
  if (!isFirebaseConfigured || !db) {
    throw new Error(firebaseConfigError);
  }

  const primaryRegion = result.regions?.[0] || null;
  const rawClassName = primaryRegion?.class_name || "";
  const plantName = rawClassName
    ? rawClassName.split("_")[0].replace(/^\w/, (char) => char.toUpperCase())
    : "Unknown plant";

  await addDoc(collection(db, "users", user.uid, "analysisHistory"), {
    filename: result.filename || "analysis-image",
    imageUrl: result.image_url || "",
    plantName,
    diseaseLabel: primaryRegion?.disease_label || "Unknown",
    selectedPart: primaryRegion?.selected_part || "Unknown part",
    confidence: primaryRegion?.disease_confidence || 0,
    treatmentPlan: treatmentPlan || "",
    language,
    timestamp: new Date().toISOString(),
    createdAt: serverTimestamp(),
    userEmail: user.email || "",
    userName: user.displayName || "",
  });
}
