// static/js/app.js
(() => {
  const $ = id => document.getElementById(id);

  // Elements
  const fileInput = $('file-input');
  const uploadBtn = $('upload-btn');
  const uploadStatus = $('upload-status');
  const previewBox = $('preview-box');
  const predictionArea = $('prediction-area');
  const predLabel = $('pred-label');
  const predConfidence = $('pred-confidence');
  const topkList = $('topk-list');
  const preprocessInfo = $('preprocess-info');

  const btnAdvice = $('btn-advice');
  const btnReport = $('btn-report');

  // Modal elements
  const adviceModal = $('advice-modal');
  const modalBackdrop = $('modal-backdrop');
  const modalTitle = $('modal-title');
  const modalBody = $('modal-body');
  const reportForm = $('report-form');
  const reportComment = $('report-comment');
  const reportSubmit = $('report-submit');
  const modalClose = $('modal-close');
  const modalCloseX = $('modal-close-x');

  // Utility
  function setStatus(text = '', isError = false) {
    if (uploadStatus) {
      uploadStatus.textContent = text;
      uploadStatus.style.color = isError ? 'crimson' : '';
    } else {
      if (isError) console.error(text); else console.log(text);
    }
  }

  // last prediction
  let lastPrediction = null;
  window.lastPrediction = null;

  // Preview
  if (fileInput) {
    fileInput.addEventListener('change', (ev) => {
      const f = ev.target.files && ev.target.files[0];
      previewBox.innerHTML = '';
      if (!f) {
        previewBox.innerHTML = '<span class="muted">No image selected</span>';
        return;
      }
      const img = document.createElement('img');
      img.alt = 'preview';
      img.style.opacity = '0';
      previewBox.appendChild(img);
      const reader = new FileReader();
      reader.onload = () => {
        img.src = reader.result;
        requestAnimationFrame(() => { img.style.transition = 'opacity 260ms ease'; img.style.opacity = '1'; });
      };
      reader.readAsDataURL(f);
    });
  }

  // Render prediction
  function handlePredictResponse(json, savedFilename = null) {
    lastPrediction = {
      prediction: json.prediction,
      confidence: json.confidence,
      top_k: json.top_k,
      preprocess: json.preprocess,
      meta: json.meta,
      filename: savedFilename || (json.meta && json.meta.filename) || json.filename || null
    };
    window.lastPrediction = lastPrediction;

    if (predictionArea) predictionArea.style.display = 'block';
    if (btnAdvice) btnAdvice.style.display = 'inline-flex';
    if (btnReport) btnReport.style.display = 'inline-flex';

    if (predLabel) predLabel.textContent = (lastPrediction.prediction || 'unknown').replace(/___/g, ': ').replace(/_/g, ' ');
    if (predConfidence) {
      const conf = (typeof lastPrediction.confidence === 'number') ? (lastPrediction.confidence * 100).toFixed(1) + '%' : (lastPrediction.confidence || '');
      predConfidence.textContent = conf ? `Confidence: ${conf}` : '';
    }

    if (topkList) {
      topkList.innerHTML = '';
      if (Array.isArray(lastPrediction.top_k)) {
        lastPrediction.top_k.forEach(item => {
          const li = document.createElement('li');
          const prob = (typeof item.prob === 'number') ? (item.prob * 100).toFixed(1) + '%' : (item.prob || '');
          li.textContent = `${(item.label || 'unknown').replace(/___/g, ': ').replace(/_/g, ' ')} - ${prob}`;
          topkList.appendChild(li);
        });
      }
    }

    if (preprocessInfo && lastPrediction.preprocess) {
      const shape = Array.isArray(lastPrediction.preprocess.shape) ? lastPrediction.preprocess.shape.join('x') : lastPrediction.preprocess.shape;
      preprocessInfo.textContent = `Preprocess shape: ${shape}  min:${lastPrediction.preprocess.min} max:${lastPrediction.preprocess.max}`;
    }

    const numericConf = (typeof lastPrediction.confidence === 'number') ? lastPrediction.confidence : null;
    if (numericConf !== null && numericConf < 0.6) {
      setStatus('Low confidence — please retake photo or upload a closeup.', true);
    } else {
      setStatus('Prediction ready.');
    }
  }
  window.handlePredictResponse = handlePredictResponse;

  // Modal helpers
  function openModal(title, bodyHtml, showReport = false) {
    if (modalTitle) modalTitle.innerText = title;
    if (modalBody) modalBody.innerHTML = bodyHtml;
    if (reportForm) reportForm.style.display = showReport ? 'block' : 'none';
    if (adviceModal) {
      adviceModal.classList.add('show');
      adviceModal.setAttribute('aria-hidden', 'false');
    }
  }
  function closeModal() {
    if (adviceModal) {
      adviceModal.classList.remove('show');
      adviceModal.setAttribute('aria-hidden', 'true');
    }
    if (reportComment) reportComment.value = '';
  }

  // Fetch advice
  async function fetchAdvice(label) {
    const res = await fetch(`/advice?label=${encodeURIComponent(label)}`);
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `HTTP ${res.status}`);
    }
    return res.json();
  }

  // Submit report
  async function submitReport(payload) {
    const res = await fetch('/report', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `HTTP ${res.status}`);
    }
    return res.json();
  }

  // Advice button
  if (btnAdvice) {
    btnAdvice.addEventListener('click', async () => {
      if (!lastPrediction) return;
      try {
        const data = await fetchAdvice(lastPrediction.prediction);
        openModal(`Treatment Advice — ${lastPrediction.prediction}`, data.advice_html || `<div>${data.advice || 'No advice available.'}</div>`, false);
      } catch (err) {
        console.error('Advice error:', err);
        openModal('Error', 'Could not fetch advice. Please try again later.', false);
      }
    });
  }

  // Report button
  if (btnReport) {
    btnReport.addEventListener('click', () => {
      if (!lastPrediction) return;
      openModal('Report Wrong Result', `<p>Please tell us what was wrong with the prediction for <strong>${lastPrediction.prediction}</strong>.</p>`, true);
    });
  }

  // Modal close handlers
  if (modalCloseX) modalCloseX.addEventListener('click', closeModal);
  if (modalClose) modalClose.addEventListener('click', closeModal);
  if (modalBackdrop) modalBackdrop.addEventListener('click', closeModal);

  // Report submit
  if (reportSubmit) {
    reportSubmit.addEventListener('click', async () => {
      if (!lastPrediction) return;
      const comment = (reportComment && reportComment.value) ? reportComment.value : '';
      const payload = {
        filename: lastPrediction.filename,
        predicted: lastPrediction.prediction,
        confidence: lastPrediction.confidence,
        top_k: lastPrediction.top_k,
        comment: comment,
        timestamp: new Date().toISOString()
      };
      try {
        await submitReport(payload);
        openModal('Thanks', '<p>Your report has been submitted. We will review it to improve the model.</p>', false);
      } catch (err) {
        console.error('Report error:', err);
        openModal('Error', 'Report submission failed. Please try again later.', false);
      }
    });
  }

  // Upload & predict
  async function uploadAndPredict(file) {
    if (!file) {
      setStatus('Select a file first.', true);
      return;
    }
    if (uploadBtn) {
      uploadBtn.disabled = true;
      uploadBtn.textContent = 'Uploading...';
    }
    setStatus('Uploading file...');
    const fd = new FormData();
    fd.append('file', file, file.name);

    try {
      const res = await fetch('/predict', { method: 'POST', body: fd });
      const text = await res.text();
      let data;
      try {
        data = JSON.parse(text);
      } catch (err) {
        console.error('Non-JSON response:', text);
        setStatus('Server returned unexpected response. See console.', true);
        return;
      }

      if (!res.ok) {
        const msg = data.error || data.detail || JSON.stringify(data);
        setStatus(`Server error: ${msg}`, true);
        return;
      }

      setStatus('Prediction received.');
      handlePredictResponse(data, data.filename || null);

      if (btnAdvice) btnAdvice.style.display = 'inline-flex';
      if (btnReport) btnReport.style.display = 'inline-flex';
    } catch (err) {
      console.error('Upload failed:', err);
      setStatus('Upload failed. Check console and server logs.', true);
    } finally {
      if (uploadBtn) {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Upload';
      }
      setTimeout(() => setStatus(''), 3000);
    }
  }

  // Wire upload button
  if (uploadBtn) {
    uploadBtn.addEventListener('click', () => {
      const file = fileInput && fileInput.files && fileInput.files[0];
      if (!file) {
        setStatus('Please select a file to upload.', true);
        return;
      }
      uploadAndPredict(file);
    });
  }

  // Allow Enter to trigger upload
  if (fileInput) {
    fileInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') uploadBtn.click();
    });
  }
})();
