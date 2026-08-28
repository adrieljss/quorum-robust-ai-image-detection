/**
 * Quorum frontend
 * ---------------
 * This file owns all client-side behavior: picking files, sending them to
 * Flask, and rendering results (including the multi-image slideshow).
 *
 * To connect a real backend later, you usually do NOT need to rewrite this
 * file. Keep the /api/analyze contract in docs/FRONTEND-HANDOVER.md and the
 * UI will keep working. Search for "BACKEND CONTRACT" below.
 */

(function () {
  "use strict";

  // -------------------------------------------------------------------------
  // CONFIG
  // Change ANALYZE_URL if the backend engineer mounts the route elsewhere.
  // MAX_FILES / MAX_BYTES_EACH are frontend guards; Flask also has a cap.
  // -------------------------------------------------------------------------
  var ANALYZE_URL = "/api/analyze";
  var MAX_FILES = 24;
  var MAX_BYTES_EACH = 12 * 1024 * 1024;
  var RING_CIRCUMFERENCE = 2 * Math.PI * 38; // matches SVG r="38" in index.html

  // Human labels for backend enums. Edit here if the API adds new values.
  var VERDICT_LABELS = {
    likely_ai: "Likely AI",
    likely_real: "Likely real",
    uncertain: "Uncertain",
  };

  var RELIABILITY_LABELS = {
    high: "High reliability",
    medium: "Medium reliability",
    low: "Low reliability",
  };

  var DEGRADATION_LABELS = {
    clean: "Clean capture",
    light_jpeg: "Light JPEG",
    heavy_jpeg: "Heavy JPEG",
    blur: "Blurred",
    resize: "Resized",
    noise: "Noisy",
  };

  var SIGNAL_META = [
    { key: "general", label: "General" },
    { key: "face", label: "Face" },
    { key: "text", label: "Text" },
    { key: "regularity", label: "Regularity" },
  ];

  // -------------------------------------------------------------------------
  // DOM handles — cached once so we are not querying on every click.
  // -------------------------------------------------------------------------
  var uploadStage = document.getElementById("upload-stage");
  var loadingStage = document.getElementById("loading-stage");
  var errorStage = document.getElementById("error-stage");
  var resultsStage = document.getElementById("results-stage");

  var form = document.getElementById("upload-form");
  var fileInput = document.getElementById("file-input");
  var dropzone = document.getElementById("dropzone");
  var previewWrap = document.getElementById("preview-wrap");
  var previewList = document.getElementById("preview-list");
  var previewCount = document.getElementById("preview-count");
  var clearBtn = document.getElementById("clear-btn");
  var analyzeBtn = document.getElementById("analyze-btn");
  var formError = document.getElementById("form-error");
  var errorCopy = document.getElementById("error-copy");
  var errorRetry = document.getElementById("error-retry");
  var newBatchBtn = document.getElementById("new-batch-btn");

  var prevBtn = document.getElementById("prev-btn");
  var nextBtn = document.getElementById("next-btn");
  var dots = document.getElementById("dots");
  var resultImage = document.getElementById("result-image");
  var resultFilename = document.getElementById("result-filename");
  var slideCounter = document.getElementById("slide-counter");
  var verdictLabel = document.getElementById("verdict-label");
  var confidenceValue = document.getElementById("confidence-value");
  var confidenceRing = document.getElementById("confidence-ring");
  var explanation = document.getElementById("explanation");
  var metaChips = document.getElementById("meta-chips");
  var signalList = document.getElementById("signal-list");
  var loadingCopy = document.getElementById("loading-copy");

  // -------------------------------------------------------------------------
  // App state
  // selectedFiles: File objects waiting to be sent.
  // previewUrls: object-URL strings so we can revoke them and free memory.
  // results: array of { file, objectUrl, data } after a successful analyze.
  // slideIndex: which result the slideshow is showing.
  // -------------------------------------------------------------------------
  var selectedFiles = [];
  var previewUrls = [];
  var results = [];
  var slideIndex = 0;
  var touchStartX = null;

  // -------------------------------------------------------------------------
  // Stage switching
  // Shows exactly one of: upload, loading, error, results.
  // -------------------------------------------------------------------------
  function showStage(stageEl) {
    [uploadStage, loadingStage, errorStage, resultsStage].forEach(function (el) {
      el.classList.toggle("is-active", el === stageEl);
    });
  }

  function setFormError(message) {
    if (!message) {
      formError.textContent = "";
      formError.classList.add("is-hidden");
      return;
    }
    formError.textContent = message;
    formError.classList.remove("is-hidden");
  }

  // -------------------------------------------------------------------------
  // File picking
  // addFiles() is the single entry point used by the file input, drag-drop,
  // and keyboard activation of the dropzone.
  // -------------------------------------------------------------------------
  function addFiles(fileList) {
    var incoming = Array.prototype.slice.call(fileList || []);
    var rejected = [];

    incoming.forEach(function (file) {
      if (!file.type || file.type.indexOf("image/") !== 0) {
        rejected.push(file.name + " is not an image");
        return;
      }
      if (file.size > MAX_BYTES_EACH) {
        rejected.push(file.name + " is larger than 12 MB");
        return;
      }
      // Skip exact duplicates (same name + size) so re-dropping is harmless.
      var already = selectedFiles.some(function (existing) {
        return existing.name === file.name && existing.size === file.size;
      });
      if (already) return;
      selectedFiles.push(file);
    });

    if (selectedFiles.length > MAX_FILES) {
      selectedFiles = selectedFiles.slice(0, MAX_FILES);
      rejected.push("Only the first " + MAX_FILES + " images were kept");
    }

    setFormError(rejected.length ? rejected.join(". ") : "");
    renderPreviews();
  }

  /**
   * Draw thumbnail cards for every queued file.
   * Each card has a remove button that splices that file out of selectedFiles.
   */
  function renderPreviews() {
    previewUrls.forEach(function (url) {
      URL.revokeObjectURL(url);
    });
    previewUrls = [];
    previewList.innerHTML = "";

    var count = selectedFiles.length;
    analyzeBtn.disabled = count === 0;
    previewWrap.classList.toggle("is-hidden", count === 0);
    previewCount.textContent =
      count === 1 ? "1 image ready" : count + " images ready";

    selectedFiles.forEach(function (file, index) {
      var url = URL.createObjectURL(file);
      previewUrls.push(url);

      var li = document.createElement("li");
      li.className = "preview-item";

      var img = document.createElement("img");
      img.src = url;
      img.alt = file.name;

      var remove = document.createElement("button");
      remove.type = "button";
      remove.className = "preview-remove";
      remove.setAttribute("aria-label", "Remove " + file.name);
      remove.textContent = "×";
      remove.addEventListener("click", function (event) {
        event.stopPropagation();
        selectedFiles.splice(index, 1);
        renderPreviews();
      });

      li.appendChild(img);
      li.appendChild(remove);
      previewList.appendChild(li);
    });
  }

  function clearSelection() {
    selectedFiles = [];
    setFormError("");
    fileInput.value = "";
    renderPreviews();
  }

  // -------------------------------------------------------------------------
  // Upload / analyze
  // BACKEND CONTRACT:
  //   POST ANALYZE_URL as multipart/form-data
  //   field name for every file: "images"
  //   expected JSON: { results: [ { verdict, confidence, signals, ... } ] }
  //   results[i] corresponds to the i-th appended file, in the same order.
  // -------------------------------------------------------------------------
  async function analyze() {
    if (!selectedFiles.length) return;

    showStage(loadingStage);
    loadingCopy.textContent =
      selectedFiles.length === 1
        ? "Asking the general, face, text, and regularity models…"
        : "Analyzing " + selectedFiles.length + " images…";

    var formData = new FormData();
    selectedFiles.forEach(function (file) {
      formData.append("images", file, file.name);
    });

    try {
      var response = await fetch(ANALYZE_URL, {
        method: "POST",
        body: formData,
        // Do NOT set Content-Type manually. The browser must add the boundary.
      });

      var payload = await response.json().catch(function () {
        return null;
      });

      if (!response.ok || !payload || !Array.isArray(payload.results)) {
        var message =
          (payload && payload.error) ||
          "The server returned an unexpected response (" + response.status + ").";
        throw new Error(message);
      }

      if (payload.results.length !== selectedFiles.length) {
        throw new Error("Result count did not match the number of uploaded images.");
      }

      // Pair each server payload with a local preview URL for the slideshow.
      results = selectedFiles.map(function (file, index) {
        return {
          file: file,
          objectUrl: URL.createObjectURL(file),
          data: payload.results[index],
        };
      });

      slideIndex = 0;
      renderDots();
      renderSlide();
      showStage(resultsStage);
    } catch (err) {
      errorCopy.textContent = err.message || "Could not reach the analyzer.";
      showStage(errorStage);
    }
  }

  // -------------------------------------------------------------------------
  // Slideshow
  // One result card is reused; renderSlide() overwrites its contents.
  // Nav buttons, dots, arrow keys, and a simple swipe all call goTo().
  // -------------------------------------------------------------------------
  function goTo(index) {
    if (!results.length) return;
    var wrapped = (index + results.length) % results.length;
    slideIndex = wrapped;
    renderSlide();
    updateDots();
  }

  function renderDots() {
    dots.innerHTML = "";
    var showDots = results.length > 1;
    dots.style.display = showDots ? "flex" : "none";
    if (!showDots) return;

    results.forEach(function (_, index) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "dot" + (index === slideIndex ? " is-active" : "");
      button.setAttribute("aria-label", "Show image " + (index + 1));
      button.addEventListener("click", function () {
        goTo(index);
      });
      dots.appendChild(button);
    });
  }

  function updateDots() {
    Array.prototype.forEach.call(dots.children, function (dot, index) {
      dot.classList.toggle("is-active", index === slideIndex);
    });
  }

  /**
   * Paint the currently selected result onto the result card.
   *
   * Fields we SHOW:
   *   verdict, confidence, explanation, reliability, content_type,
   *   degradation_estimate, signals.{general,face,text,regularity}
   *
   * Fields we keep quiet unless they add a cue:
   *   provenance.c2pa / provenance.exif_software — only rendered as chips
   *   when the backend actually found something (not null).
   */
  function renderSlide() {
    var item = results[slideIndex];
    var data = item.data || {};
    var many = results.length > 1;

    prevBtn.disabled = !many;
    nextBtn.disabled = !many;
    document.getElementById("slideshow").classList.toggle("is-single", !many);

    resultImage.src = item.objectUrl;
    resultImage.alt = data.filename || item.file.name;
    resultFilename.textContent = data.filename || item.file.name;
    slideCounter.textContent = slideIndex + 1 + " / " + results.length;

    var verdict = data.verdict || "uncertain";
    verdictLabel.textContent = VERDICT_LABELS[verdict] || verdict;
    verdictLabel.className = "verdict-label";
    if (verdict === "likely_ai") verdictLabel.classList.add("is-ai");
    if (verdict === "likely_real") verdictLabel.classList.add("is-real");
    if (verdict === "uncertain") verdictLabel.classList.add("is-uncertain");

    var confidence = Number(data.confidence);
    if (isNaN(confidence)) confidence = 0;
    confidence = Math.max(0, Math.min(1, confidence));
    confidenceValue.textContent = Math.round(confidence * 100) + "%";
    // Circumference offset: 0 = full ring, CIRC = empty ring.
    confidenceRing.style.strokeDasharray = String(RING_CIRCUMFERENCE);
    confidenceRing.style.strokeDashoffset = String(
      RING_CIRCUMFERENCE * (1 - confidence)
    );
    if (verdict === "likely_real") {
      confidenceRing.style.stroke = "#3f6b4a";
    } else if (verdict === "uncertain") {
      confidenceRing.style.stroke = "#9a7b3a";
    } else {
      confidenceRing.style.stroke = "#d56a2b";
    }

    explanation.textContent = data.explanation || "";
    renderChips(data);
    renderSignals(data.signals || {});
  }

  /**
   * Small pills under the explanation. Intentionally not a dump of every field.
   */
  function renderChips(data) {
    metaChips.innerHTML = "";

    appendChip(RELIABILITY_LABELS[data.reliability] || data.reliability);
    if (data.content_type) {
      appendChip("Scene · " + data.content_type);
    }
    if (data.degradation_estimate) {
      appendChip(
        DEGRADATION_LABELS[data.degradation_estimate] || data.degradation_estimate
      );
    }

    var provenance = data.provenance || {};
    if (provenance.c2pa) {
      appendChip("C2PA · " + provenance.c2pa);
    }
    if (provenance.exif_software) {
      appendChip("Software · " + provenance.exif_software);
    }
  }

  function appendChip(text) {
    if (!text) return;
    var span = document.createElement("span");
    span.className = "chip";
    span.textContent = text;
    metaChips.appendChild(span);
  }

  /**
   * Four model bars. A missing / null signal is shown as "Not measured"
   * rather than 0 — 0 would look like "the model said real".
   */
  function renderSignals(signals) {
    signalList.innerHTML = "";

    SIGNAL_META.forEach(function (meta) {
      var raw = signals[meta.key];
      var li = document.createElement("li");
      li.className = "signal-row";

      var name = document.createElement("span");
      name.className = "signal-name";
      name.textContent = meta.label;

      var track = document.createElement("div");
      track.className = "signal-track";
      var fill = document.createElement("span");
      fill.className = "signal-fill";
      track.appendChild(fill);

      var value = document.createElement("span");
      value.className = "signal-value";

      if (raw === null || raw === undefined || raw === "") {
        fill.classList.add("is-empty");
        value.textContent = "n/a";
        value.title = "This branch did not run (for example, no face was found).";
      } else {
        var n = Math.max(0, Math.min(1, Number(raw)));
        fill.style.width = Math.round(n * 100) + "%";
        value.textContent = Math.round(n * 100) + "%";
      }

      li.appendChild(name);
      li.appendChild(track);
      li.appendChild(value);
      signalList.appendChild(li);
    });
  }

  function resetToUpload() {
    results.forEach(function (item) {
      URL.revokeObjectURL(item.objectUrl);
    });
    results = [];
    slideIndex = 0;
    clearSelection();
    showStage(uploadStage);
  }

  // -------------------------------------------------------------------------
  // Event wiring
  // -------------------------------------------------------------------------
  dropzone.addEventListener("click", function () {
    fileInput.click();
  });

  dropzone.addEventListener("keydown", function (event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", function () {
    addFiles(fileInput.files);
    fileInput.value = "";
  });

  ["dragenter", "dragover"].forEach(function (type) {
    dropzone.addEventListener(type, function (event) {
      event.preventDefault();
      dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach(function (type) {
    dropzone.addEventListener(type, function (event) {
      event.preventDefault();
      dropzone.classList.remove("is-dragover");
    });
  });

  dropzone.addEventListener("drop", function (event) {
    if (event.dataTransfer && event.dataTransfer.files) {
      addFiles(event.dataTransfer.files);
    }
  });

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    analyze();
  });

  clearBtn.addEventListener("click", clearSelection);
  errorRetry.addEventListener("click", analyze);
  newBatchBtn.addEventListener("click", resetToUpload);
  prevBtn.addEventListener("click", function () {
    goTo(slideIndex - 1);
  });
  nextBtn.addEventListener("click", function () {
    goTo(slideIndex + 1);
  });

  document.addEventListener("keydown", function (event) {
    if (!resultsStage.classList.contains("is-active")) return;
    if (event.key === "ArrowLeft") goTo(slideIndex - 1);
    if (event.key === "ArrowRight") goTo(slideIndex + 1);
  });

  // Lightweight swipe for the slideshow on phones (nav arrows are hidden).
  var slideshow = document.getElementById("slideshow");
  slideshow.addEventListener(
    "touchstart",
    function (event) {
      if (!event.changedTouches.length) return;
      touchStartX = event.changedTouches[0].clientX;
    },
    { passive: true }
  );
  slideshow.addEventListener(
    "touchend",
    function (event) {
      if (touchStartX === null || !event.changedTouches.length) return;
      var delta = event.changedTouches[0].clientX - touchStartX;
      touchStartX = null;
      if (Math.abs(delta) < 40) return;
      if (delta < 0) goTo(slideIndex + 1);
      else goTo(slideIndex - 1);
    },
    { passive: true }
  );
})();
