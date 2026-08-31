/**
 * Quorum video frontend
 * ---------------------
 * Owns picking one local video, POSTing it to the video endpoint, and
 * rendering its single analysis result. The API contract lives in
 * docs/VIDEO-FRONTEND-HANDOVER.md.
 */
(function () {
  "use strict";

  var ANALYZE_VIDEO_URL = "/api/analyze-video";
  var MAX_VIDEO_BYTES = 45 * 1024 * 1024;
  var RING_CIRCUMFERENCE = 2 * Math.PI * 38;
  var VIDEO_EXTENSIONS = ["mp4", "webm", "mov"];
  var VERDICT_LABELS = { likely_ai: "Likely AI", likely_real: "Likely real", uncertain: "Uncertain" };

  var selectedVideo = null;
  var previewUrl = null;
  var stages = ["video-upload-stage", "video-loading-stage", "video-error-stage", "video-results-stage"].map(function (id) { return document.getElementById(id); });
  var form = document.getElementById("video-upload-form");
  var input = document.getElementById("video-input");
  var dropzone = document.getElementById("video-dropzone");
  var previewWrap = document.getElementById("video-preview-wrap");
  var preview = document.getElementById("video-preview");
  var filename = document.getElementById("video-filename");
  var clearButton = document.getElementById("video-clear-btn");
  var analyzeButton = document.getElementById("video-analyze-btn");
  var formError = document.getElementById("video-form-error");
  var errorCopy = document.getElementById("video-error-copy");
  var resultVideo = document.getElementById("result-video");
  var verdictLabel = document.getElementById("video-verdict-label");
  var confidenceValue = document.getElementById("video-confidence-value");
  var confidenceRing = document.getElementById("video-confidence-ring");
  var explanation = document.getElementById("video-explanation");
  var signalList = document.getElementById("video-signal-list");

  function showStage(id) {
    stages.forEach(function (stage) { stage.classList.toggle("is-active", stage.id === id); });
  }

  function setFormError(message) {
    formError.textContent = message || "";
    formError.classList.toggle("is-hidden", !message);
  }

  function hasSupportedExtension(file) {
    var extension = (file.name || "").split(".").pop().toLowerCase();
    return VIDEO_EXTENSIONS.indexOf(extension) !== -1;
  }

  function setVideo(file) {
    if (!file) return;
    if (!hasSupportedExtension(file)) return setFormError("Choose an MP4, WebM, or MOV video.");
    if (file.size > MAX_VIDEO_BYTES) return setFormError("Videos must be 45 MB or smaller.");
    if (previewUrl) URL.revokeObjectURL(previewUrl);

    selectedVideo = file;
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    filename.textContent = file.name;
    previewWrap.classList.remove("is-hidden");
    analyzeButton.disabled = false;
    setFormError("");
  }

  function clearVideo() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    selectedVideo = null;
    previewUrl = null;
    input.value = "";
    preview.removeAttribute("src");
    preview.load();
    previewWrap.classList.add("is-hidden");
    analyzeButton.disabled = true;
    setFormError("");
  }

  function renderSignals(signals) {
    signalList.innerHTML = "";
    Object.keys(signals || {}).forEach(function (key) {
      var raw = signals[key];
      var value = Number(raw);
      var li = document.createElement("li");
      li.className = "signal-row";
      var name = document.createElement("span");
      name.className = "signal-name";
      name.textContent = key.replace(/_/g, " ");
      var track = document.createElement("div");
      track.className = "signal-track";
      var fill = document.createElement("span");
      fill.className = "signal-fill";
      track.appendChild(fill);
      var label = document.createElement("span");
      label.className = "signal-value";
      if (raw === null || raw === undefined || isNaN(value)) {
        fill.classList.add("is-empty");
        label.textContent = "n/a";
      } else {
        value = Math.max(0, Math.min(1, value));
        fill.style.width = Math.round(value * 100) + "%";
        label.textContent = Math.round(value * 100) + "%";
      }
      li.appendChild(name);
      li.appendChild(track);
      li.appendChild(label);
      signalList.appendChild(li);
    });
  }

  function renderResult(data) {
    var verdict = data.verdict || "uncertain";
    var confidence = Math.max(0, Math.min(1, Number(data.confidence) || 0));
    verdictLabel.textContent = VERDICT_LABELS[verdict] || verdict;
    verdictLabel.className = "verdict-label is-" + (verdict === "likely_ai" ? "ai" : verdict === "likely_real" ? "real" : "uncertain");
    confidenceValue.textContent = Math.round(confidence * 100) + "%";
    confidenceRing.style.strokeDasharray = String(RING_CIRCUMFERENCE);
    confidenceRing.style.strokeDashoffset = String(RING_CIRCUMFERENCE * (1 - confidence));
    confidenceRing.style.stroke = verdict === "likely_real" ? "#3f6b4a" : verdict === "uncertain" ? "#9a7b3a" : "#d56a2b";
    explanation.textContent = data.explanation || "";
    renderSignals(data.signals);
  }

  async function analyze() {
    if (!selectedVideo) return;
    showStage("video-loading-stage");
    var formData = new FormData();
    formData.append("video", selectedVideo, selectedVideo.name);
    try {
      var response = await fetch(ANALYZE_VIDEO_URL, { method: "POST", body: formData });
      var payload = await response.json().catch(function () { return null; });
      if (!response.ok || !payload || !payload.result) throw new Error((payload && payload.error) || "The server returned an unexpected response.");
      resultVideo.src = previewUrl;
      renderResult(payload.result);
      showStage("video-results-stage");
    } catch (error) {
      errorCopy.textContent = error.message || "Could not reach the video analyzer.";
      showStage("video-error-stage");
    }
  }

  dropzone.addEventListener("click", function () { input.click(); });
  dropzone.addEventListener("keydown", function (event) { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); } });
  input.addEventListener("change", function () { setVideo(input.files[0]); input.value = ""; });
  ["dragenter", "dragover"].forEach(function (type) { dropzone.addEventListener(type, function (event) { event.preventDefault(); dropzone.classList.add("is-dragover"); }); });
  ["dragleave", "drop"].forEach(function (type) { dropzone.addEventListener(type, function (event) { event.preventDefault(); dropzone.classList.remove("is-dragover"); }); });
  dropzone.addEventListener("drop", function (event) { setVideo(event.dataTransfer.files[0]); });
  form.addEventListener("submit", function (event) { event.preventDefault(); analyze(); });
  clearButton.addEventListener("click", clearVideo);
  document.getElementById("video-retry").addEventListener("click", analyze);
  document.getElementById("video-new-btn").addEventListener("click", function () { clearVideo(); showStage("video-upload-stage"); });
})();
