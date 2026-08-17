import { buildSessionJson, buildSessionCsv, triggerTextDownload } from './exporter.js';

const $ = (id) => document.getElementById(id);

const video = $('video');
const videoFileInput = $('videoFile');
const jsonFileInput = $('jsonFile');
const statusLine = $('statusLine');
const logBody = $('logBody');
const countPill = $('countPill');
const btnExportJson = $('btnExportJson');
const btnExportCsv = $('btnExportCsv');

let session = null; // { recordingStartEpochMs, entries: [...] }

function setStatus(msg, level = '') {
  statusLine.textContent = msg;
  statusLine.className = 'status-line' + (level ? ` ${level}` : '');
}

videoFileInput.addEventListener('change', () => {
  const file = videoFileInput.files[0];
  if (!file) return;
  video.src = URL.createObjectURL(file);
  setStatus(`Loaded footage: ${file.name}`);
});

jsonFileInput.addEventListener('change', async () => {
  const file = jsonFileInput.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const raw = JSON.parse(text);
    session = {
      recordingStartEpochMs: raw.recordingStartEpochMs,
      recordingEndEpochMs: raw.recordingEndEpochMs,
      roi: raw.roi,
      settings: raw.settings,
      entries: (raw.entries || []).map((e) => ({
        number: e.number,
        firstSeen: e.firstSeenEpochMs,
        lastSeen: e.lastSeenEpochMs,
        durationMs: e.durationMs,
        hits: e.hits,
        confidence: e.confidence,
        confirmed: false,
      })),
    };
    session.entries.sort((a, b) => a.lastSeen - b.lastSeen);
    renderTable();
    btnExportJson.disabled = false;
    btnExportCsv.disabled = false;
    setStatus(`Loaded ${session.entries.length} log entries.`);
  } catch (e) {
    setStatus(`Could not parse log file: ${e.message}`, 'err');
  }
});

function seekTo(epochMs) {
  if (!session || !session.recordingStartEpochMs) {
    setStatus('Load both the footage and the log file first.', 'warn');
    return;
  }
  const offsetSeconds = (epochMs - session.recordingStartEpochMs) / 1000;
  if (offsetSeconds < 0 || !isFinite(offsetSeconds)) {
    setStatus('This entry falls outside the loaded footage.', 'warn');
    return;
  }
  video.currentTime = offsetSeconds;
  video.play().catch(() => {});
}

function renderTable() {
  logBody.innerHTML = '';
  session.entries.forEach((entry, idx) => {
    const tr = document.createElement('tr');

    const numberTd = document.createElement('td');
    const numberInput = document.createElement('input');
    numberInput.type = 'text';
    numberInput.value = entry.number;
    numberInput.style.width = '70px';
    numberInput.addEventListener('input', () => {
      entry.number = numberInput.value;
    });
    numberTd.appendChild(numberInput);

    const firstTd = document.createElement('td');
    firstTd.textContent = new Date(entry.firstSeen).toLocaleTimeString(undefined, { hour12: false });

    const lastTd = document.createElement('td');
    lastTd.textContent = new Date(entry.lastSeen).toLocaleTimeString(undefined, { hour12: false });

    const confTd = document.createElement('td');
    confTd.textContent = Math.round(entry.confidence ?? 0);

    const seekTd = document.createElement('td');
    const firstBtn = document.createElement('button');
    firstBtn.textContent = 'First';
    firstBtn.addEventListener('click', () => seekTo(entry.firstSeen));
    const lastBtn = document.createElement('button');
    lastBtn.textContent = 'Last';
    lastBtn.style.marginLeft = '4px';
    lastBtn.addEventListener('click', () => seekTo(entry.lastSeen));
    seekTd.appendChild(firstBtn);
    seekTd.appendChild(lastBtn);

    const okTd = document.createElement('td');
    const okCheckbox = document.createElement('input');
    okCheckbox.type = 'checkbox';
    okCheckbox.checked = entry.confirmed;
    okCheckbox.addEventListener('change', () => {
      entry.confirmed = okCheckbox.checked;
    });
    okTd.appendChild(okCheckbox);

    tr.append(numberTd, firstTd, lastTd, confTd, seekTd, okTd);
    logBody.appendChild(tr);
  });
  countPill.textContent = `${session.entries.length} entries`;
}

btnExportJson.addEventListener('click', () => {
  if (!session) return;
  triggerTextDownload(buildSessionJson(session), 'race-log-corrected.json', 'application/json');
});
btnExportCsv.addEventListener('click', () => {
  if (!session) return;
  triggerTextDownload(buildSessionCsv(session), 'race-log-corrected.csv', 'text/csv');
});
