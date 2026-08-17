// Persist operator-tuned settings (ROI, contrast, thresholds, etc.) across
// page reloads/sessions via localStorage, so a race-day operator only has
// to tune the camera once, not before every start-group's session.

const KEY = 'cxtt.settings.v1';

export function loadSettings() {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    return null;
  }
}

export function saveSettings(values) {
  try {
    localStorage.setItem(KEY, JSON.stringify(values));
  } catch (e) {
    // Ignore quota errors / private-browsing restrictions — settings just won't persist.
  }
}
