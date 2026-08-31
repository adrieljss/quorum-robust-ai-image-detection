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
  var SINGLE_SIGNALS = ["general", "tampered", "regularity"];
  var VERDICT_LABELS = { likely_ai: "Likely AI", likely_real: "Likely real", uncertain: "Uncertain" };

  var selectedVideo = null;
  var previewUrl = null;
  var frameResults = [];
  var analysisDuration = 0;
  var activeFrame = null;
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
  var frameCanvas = document.getElementById("video-frame-canvas");
  var videoResultCard = document.querySelector(".video-result-card");
  var videoRegionsPanel = document.getElementById("video-regions-panel");
  var videoAccountedRegions = document.getElementById("video-accounted-region-list");
  var videoUnaccountedRegions = document.getElementById("video-unaccounted-region-list");
  var videoRegionConnectors = document.getElementById("video-region-connectors");
  var scrubber = document.getElementById("video-scrubber");
  var markers = document.getElementById("video-frame-markers");
  var currentTime = document.getElementById("video-current-time");
  var duration = document.getElementById("video-duration");
  var frameHint = document.getElementById("video-frame-hint");
  var frameTitle = document.getElementById("video-frame-title");
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

  function formatTime(seconds) {
    seconds = Math.max(0, Number(seconds) || 0);
    return Math.floor(seconds / 60) + ":" + String(Math.floor(seconds % 60)).padStart(2, "0");
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
    resetAnalysisDisplay();

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
    resetAnalysisDisplay();
    input.value = "";
    preview.removeAttribute("src");
    preview.load();
    previewWrap.classList.add("is-hidden");
    analyzeButton.disabled = true;
    setFormError("");
  }

  /** Clear every result-specific node so a new upload cannot inherit it. */
  function resetAnalysisDisplay() {
    frameResults = [];
    activeFrame = null;
    analysisDuration = 0;
    videoResultCard.classList.remove("has-frame-results");
    videoAccountedRegions.innerHTML = "";
    videoUnaccountedRegions.innerHTML = "";
    videoRegionsPanel.classList.add("is-hidden");
    videoRegionConnectors.innerHTML = "";
    markers.innerHTML = "";
    frameCanvas.width = 0;
    frameCanvas.height = 0;
    frameCanvas.classList.add("is-hidden");
    resultVideo.classList.remove("is-hidden");
    resultVideo.onloadedmetadata = null;
    resultVideo.onloadeddata = null;
    resultVideo.ontimeupdate = null;
  }

  function renderSignals(signals) {
    signalList.innerHTML = "";
    SINGLE_SIGNALS.forEach(function (key) {
      var raw = (signals || {})[key];
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

  function renderResult(data, title) {
    var verdict = data.verdict || "uncertain";
    var confidence = Math.max(0, Math.min(1, Number(data.confidence) || 0));
    frameTitle.textContent = title;
    verdictLabel.textContent = VERDICT_LABELS[verdict] || verdict;
    verdictLabel.className = "verdict-label is-" + (verdict === "likely_ai" ? "ai" : verdict === "likely_real" ? "real" : "uncertain");
    confidenceValue.textContent = Math.round(confidence * 100) + "%";
    confidenceRing.style.strokeDasharray = String(RING_CIRCUMFERENCE);
    confidenceRing.style.strokeDashoffset = String(RING_CIRCUMFERENCE * (1 - confidence));
    confidenceRing.style.stroke = verdict === "likely_real" ? "#3f6b4a" : verdict === "uncertain" ? "#9a7b3a" : "#d56a2b";
    renderSignals(data.signals);
  }

  function nearestFrame(time) {
    if (!frameResults.length) return null;
    return frameResults.reduce(function (closest, frame) {
      return Math.abs(frame.timestamp_seconds - time) < Math.abs(closest.timestamp_seconds - time) ? frame : closest;
    });
  }

  function selectFrameAt(time) {
    var frame = nearestFrame(time);
    if (!frame) return;
    renderResult(frame, "Timestamp " + formatTime(frame.timestamp_seconds));
    renderAnnotatedFrame(frame);
  }

  function showVideoPlayer() {
    frameCanvas.classList.add("is-hidden");
    resultVideo.classList.remove("is-hidden");
  }

  function renderAnnotatedFrame(frame) {
    if (!resultVideo.videoWidth || !resultVideo.videoHeight) return;
    activeFrame = frame;
    var context = frameCanvas.getContext("2d");
    frameCanvas.width = resultVideo.videoWidth;
    frameCanvas.height = resultVideo.videoHeight;
    context.drawImage(resultVideo, 0, 0, frameCanvas.width, frameCanvas.height);
    (frame.regions || []).forEach(function (region) {
      if (!region.bbox) return;
      var box = region.bbox;
      var verdict = region.verdict || "likely_ai";
      var color = verdict === "likely_real" ? "#16c172" : verdict === "uncertain" ? "#f6a609" : "#ff3b30";
      context.save();
      context.beginPath();
      context.rect(box.x, box.y, box.width, box.height);
      context.clip();
      context.strokeStyle = color;
      context.globalAlpha = 0.42;
      context.lineWidth = 8;
      for (var offset = -box.height; offset < box.width; offset += 22) {
        context.beginPath();
        context.moveTo(box.x + offset, box.y);
        context.lineTo(box.x + offset + box.height, box.y + box.height);
        context.stroke();
      }
      context.restore();
      context.strokeStyle = color;
      context.lineWidth = 6;
      context.strokeRect(box.x, box.y, box.width, box.height);
    });
    resultVideo.classList.add("is-hidden");
    frameCanvas.classList.remove("is-hidden");
    renderVideoRegions(frame);
    requestAnimationFrame(function () { renderVideoRegionConnectors(frame); });
  }

  function renderVideoRegions(frame) {
    videoAccountedRegions.innerHTML = "";
    videoUnaccountedRegions.innerHTML = "";
    (frame.regions || []).forEach(function (region, index) {
      var type = (region.type || "").toLowerCase();
      var list = type === "face" ? videoAccountedRegions : type === "text" ? videoUnaccountedRegions : null;
      if (!list || !region.bbox) return;
      var score = Number(region.score);
      var item = document.createElement("li");
      item.className = "region-card is-" + (region.verdict || "uncertain");
      item.setAttribute("data-video-region-index", String(index));
      var details = document.createElement("div");
      details.className = "region-details";
      var crop = document.createElement("canvas");
      crop.width = region.bbox.width;
      crop.height = region.bbox.height;
      crop.getContext("2d").drawImage(
        frameCanvas,
        region.bbox.x,
        region.bbox.y,
        region.bbox.width,
        region.bbox.height,
        0,
        0,
        region.bbox.width,
        region.bbox.height
      );
      var image = document.createElement("img");
      image.className = "region-crop";
      image.src = crop.toDataURL("image/png");
      image.alt = "";
      var title = document.createElement("strong");
      title.textContent = region.type || "Detected region";
      var summary = document.createElement("span");
      summary.textContent = isNaN(score) ? "Score unavailable" : Math.round(Math.max(0, Math.min(1, score)) * 100) + "% AI";
      details.appendChild(title);
      details.appendChild(summary);
      item.appendChild(image);
      item.appendChild(details);
      list.appendChild(item);
    });
    videoRegionsPanel.classList.toggle("is-hidden", !videoAccountedRegions.children.length && !videoUnaccountedRegions.children.length);
  }

  function renderVideoRegionConnectors(frame) {
    videoRegionConnectors.innerHTML = "";
    if (!frameCanvas.width || videoRegionsPanel.classList.contains("is-hidden")) return;
    var cardRect = videoResultCard.getBoundingClientRect();
    var canvasRect = frameCanvas.getBoundingClientRect();
    var scaleX = canvasRect.width / frameCanvas.width;
    var scaleY = canvasRect.height / frameCanvas.height;
    videoRegionConnectors.setAttribute("viewBox", "0 0 " + cardRect.width + " " + cardRect.height);
    (frame.regions || []).forEach(function (region, index) {
      var target = videoResultCard.querySelector('[data-video-region-index="' + index + '"]');
      if (!target || !region.bbox) return;
      var targetRect = target.getBoundingClientRect();
      var box = region.bbox;
      var line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("class", "region-connector is-" + (region.verdict || "uncertain"));
      line.setAttribute("x1", canvasRect.left - cardRect.left + (box.x + box.width) * scaleX);
      line.setAttribute("y1", canvasRect.top - cardRect.top + (box.y + box.height / 2) * scaleY);
      line.setAttribute("x2", targetRect.left - cardRect.left);
      line.setAttribute("y2", targetRect.top - cardRect.top + targetRect.height / 2);
      videoRegionConnectors.appendChild(line);
    });
  }

  function seekToFrame(frame) {
    resultVideo.currentTime = frame.timestamp_seconds;
    currentTime.textContent = formatTime(frame.timestamp_seconds);
    renderResult(frame, "Timestamp " + formatTime(frame.timestamp_seconds));
    resultVideo.addEventListener("seeked", function onSeeked() {
      resultVideo.removeEventListener("seeked", onSeeked);
      renderAnnotatedFrame(frame);
    });
  }

  function renderMarkers() {
    markers.innerHTML = "";
    if (!analysisDuration || !frameResults.length) return;
    frameResults.forEach(function (frame) {
      var marker = document.createElement("span");
      var state = frame.verdict === "likely_real" ? "is-real" : frame.verdict === "uncertain" ? "is-uncertain" : "is-ai";
      marker.className = "video-frame-marker " + state;
      marker.style.left = Math.min(100, Math.max(0, frame.timestamp_seconds / analysisDuration * 100)) + "%";
      marker.title = formatTime(frame.timestamp_seconds) + " · " + (VERDICT_LABELS[frame.verdict] || frame.verdict);
      marker.setAttribute("role", "button");
      marker.setAttribute("tabindex", "0");
      marker.setAttribute("aria-label", "Show analyzed frame at " + formatTime(frame.timestamp_seconds));
      marker.addEventListener("click", function () { seekToFrame(frame); });
      marker.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          seekToFrame(frame);
        }
      });
      markers.appendChild(marker);
    });
  }

  function configureResultPlayer(result) {
    analysisDuration = Number(result.duration_seconds) || 0;
    scrubber.max = String(analysisDuration);
    duration.textContent = formatTime(analysisDuration);
    renderMarkers();
    resultVideo.onloadedmetadata = function () {
      if (!analysisDuration && isFinite(resultVideo.duration)) {
        analysisDuration = resultVideo.duration;
        scrubber.max = String(analysisDuration);
        duration.textContent = formatTime(analysisDuration);
        renderMarkers();
      }
    };
    resultVideo.ontimeupdate = function () {
      scrubber.value = String(resultVideo.currentTime);
      currentTime.textContent = formatTime(resultVideo.currentTime);
      selectFrameAt(resultVideo.currentTime);
    };
  }

  async function analyze() {
    if (!selectedVideo) return;
    resetAnalysisDisplay();
    showStage("video-loading-stage");
    var formData = new FormData();
    formData.append("video", selectedVideo, selectedVideo.name);
    try {
      var response = await fetch(ANALYZE_VIDEO_URL, { method: "POST", body: formData });
      var payload = await response.json().catch(function () { return null; });
      if (!response.ok || !payload || !payload.result) throw new Error((payload && payload.error) || "The server returned an unexpected response.");
      frameResults = Array.isArray(payload.frames) ? payload.frames : [];
      resultVideo.src = previewUrl;
      videoResultCard.classList.toggle("has-frame-results", frameResults.length > 0);
      configureResultPlayer(payload.result);
      if (frameResults.length) {
        resultVideo.onloadeddata = function () { seekToFrame(nearestFrame(0)); };
        if (resultVideo.readyState >= 2) seekToFrame(nearestFrame(0));
        frameHint.textContent = "Colored markers show analyzed frames. Seek along the timeline to inspect the nearest frame prediction.";
      } else {
        showVideoPlayer();
        renderResult(payload.result, "Video summary");
        frameHint.textContent = "No frame-level records were returned for this video.";
      }
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
  scrubber.addEventListener("input", function () {
    var time = Number(scrubber.value);
    if (isFinite(time)) {
      resultVideo.currentTime = time;
      currentTime.textContent = formatTime(time);
      var frame = nearestFrame(time);
      if (frame) seekToFrame(frame);
    }
  });
  window.addEventListener("resize", function () {
    if (activeFrame) requestAnimationFrame(function () { renderVideoRegionConnectors(activeFrame); });
  });
  form.addEventListener("submit", function (event) { event.preventDefault(); analyze(); });
  clearButton.addEventListener("click", clearVideo);
  document.getElementById("video-retry").addEventListener("click", analyze);
  document.getElementById("video-new-btn").addEventListener("click", function () { clearVideo(); showStage("video-upload-stage"); });
})();
