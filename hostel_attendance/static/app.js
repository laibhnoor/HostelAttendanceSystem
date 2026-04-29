const state = {
  stream: null,
  captureTimer: null,
  mode: null,
  alertBox: null,
  lastResult: null,
  snapshotInterval: 700,
  videoWidth: 640,
  videoHeight: 480,
  jpegQuality: 0.9,
};

function setAlert(message, kind) {
  if (!state.alertBox) return;
  state.alertBox.textContent = message;
  state.alertBox.classList.remove("success", "error");
  if (kind) {
    state.alertBox.classList.add(kind);
  }
}

function loadSettings() {
  const body = document.body;
  if (!body) return;
  const interval = Number(body.dataset.interval);
  const width = Number(body.dataset.width);
  const height = Number(body.dataset.height);
  const quality = Number(body.dataset.quality);
  if (!Number.isNaN(interval)) state.snapshotInterval = interval;
  if (!Number.isNaN(width)) state.videoWidth = width;
  if (!Number.isNaN(height)) state.videoHeight = height;
  if (!Number.isNaN(quality)) state.jpegQuality = quality;
}

async function setupCamera(videoEl) {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: state.videoWidth, height: state.videoHeight },
      audio: false,
    });
    videoEl.srcObject = state.stream;
    await videoEl.play();
  } catch (error) {
    setAlert("Camera access denied or unavailable.", "error");
  }
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
  }
  if (state.captureTimer) {
    clearInterval(state.captureTimer);
  }
}

function captureFrame(videoEl, canvasEl) {
  const ctx = canvasEl.getContext("2d");
  ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
  return new Promise((resolve) => {
    canvasEl.toBlob((blob) => resolve(blob), "image/jpeg", state.jpegQuality);
  });
}

async function sendEnrollFrame(payload) {
  try {
    const response = await fetch("/api/enroll", {
      method: "POST",
      body: payload,
    });
    return await response.json();
  } catch (error) {
    return { ok: false, message: "Enrollment request failed." };
  }
}

async function sendRecognizeFrame(payload) {
  try {
    const response = await fetch("/api/recognize", {
      method: "POST",
      body: payload,
    });
    return await response.json();
  } catch (error) {
    return { ok: false, message: "Recognition request failed." };
  }
}

function startEnrollment() {
  const videoEl = document.getElementById("video");
  const canvasEl = document.getElementById("canvas");
  const statusEl = document.getElementById("capture-status");
  const formEl = document.getElementById("enroll-form");

  if (!videoEl || !canvasEl || !formEl) return;

  state.alertBox = document.getElementById("alert-box");
  state.mode = "enroll";
  setAlert("Stand in front of the camera. Capturing samples...", "success");

  setupCamera(videoEl);

  state.captureTimer = setInterval(async () => {
    const formData = new FormData(formEl);
    const blob = await captureFrame(videoEl, canvasEl);
    formData.append("frame", blob, "frame.jpg");

    const result = await sendEnrollFrame(formData);
    if (result.ok) {
      statusEl.textContent = `${result.captured} / ${result.required}`;
      if (result.captured >= result.required) {
        setAlert("Enrollment complete!", "success");
        stopCamera();
      }
    } else {
      setAlert(result.message, "error");
    }
  }, state.snapshotInterval);
}

function startRecognition() {
  const videoEl = document.getElementById("video");
  const canvasEl = document.getElementById("canvas");
  const hostelEl = document.getElementById("hostel");
  if (!videoEl || !canvasEl || !hostelEl) return;

  state.alertBox = document.getElementById("alert-box");
  state.mode = "recognize";
  setAlert("Recognition active. Keep the face centered.", "success");

  setupCamera(videoEl);

  state.captureTimer = setInterval(async () => {
    const formData = new FormData();
    const blob = await captureFrame(videoEl, canvasEl);
    formData.append("frame", blob, "frame.jpg");
    formData.append("hostel", hostelEl.value);

    const result = await sendRecognizeFrame(formData);
    if (result.ok) {
      state.lastResult = result;
    }
  }, state.snapshotInterval);

  const source = new EventSource("/api/alerts");
  source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (!data || !data.message) return;
    setAlert(data.message, data.kind);
  };
}

function boot() {
  loadSettings();
  const page = document.body.dataset.page;
  if (page === "enroll") {
    startEnrollment();
  }
  if (page === "recognize") {
    startRecognition();
  }
}

window.addEventListener("beforeunload", stopCamera);
window.addEventListener("DOMContentLoaded", boot);
