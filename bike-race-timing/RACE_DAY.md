# Race Day Guide

A practical checklist for running the CX TT timing app at an actual race.
Print this or keep it open on a phone next to the laptop.

## Before you leave the house

- [ ] **Browser: Chrome or Edge, recent version.** These give you the
      crash-safe live-save folder (and optional WebGPU preprocessing, see
      below). Firefox/Safari still work fine (OCR + recording both run),
      just without the live-save folder — footage is held in memory and
      only downloaded at the end.
- [ ] **Test the whole flow once, at home, in the actual room/lighting you'll
      use if possible.** Camera → ROI → Start Session → a minute of footage
      → Stop → Download. Confirm the JSON/CSV and video actually download
      and open.
- [ ] **Run the Performance Check** (see "Starting a session" step 6 below)
      on the actual laptop and camera you'll use. Do this at home first, not
      for the first time at the venue.
- [ ] **Charge everything.** A laptop actively running a camera, WebGPU, and
      continuous video recording draws real power. Bring the charger and, if
      the site has no outlets, a battery brick.
- [ ] Pull up this repo once while you still have internet — after that,
      **the app needs zero connectivity**. The OCR engine and its trained
      data are vendored in `vendor/tesseract/`, not fetched live.
- [ ] Pick (or print) a **numeric-only bib scheme** if you have any control
      over it — the OCR is digit-only by default.

## Camera & site setup

- [ ] Mount the camera on a **tripod or fixed stand** — handheld footage
      makes both OCR and manual review much harder. Aim it square at where
      riders' handlebars will be as they approach, roughly waist-to-chest
      height for a seated rider.
- [ ] Frame so the bib number is **legible on-screen for at least several
      frames** as a rider approaches — a number that flashes through in one
      frame won't get read reliably regardless of tuning.
- [ ] Avoid **backlighting** (camera facing directly into the sun) — it
      silhouettes riders and kills contrast. Side or front lighting on the
      numbers is much better. If backlighting is unavoidable, the Contrast/
      Brightness/Binarize controls partially compensate — test it.
- [ ] If it's a long session with variable light (dawn start, midday finish,
      etc.), plan to re-check contrast settings periodically.

## Starting a session

1. Open `capture.html` (`http://localhost:8000/capture.html` if using the
   quick-start server from the README).
2. **Choose Save Folder…** and pick somewhere with room for video (a few GB
   per hour) — this is what makes the session crash-safe. If the button is
   greyed out, your browser doesn't support it; that's fine, footage will
   just download at the end instead.
3. Pick the camera from the dropdown, **Start Camera**.
4. **Drag a box on the video** over the strip where bib numbers will be
   legible. Redraw it any time by dragging again.
5. Leave **Enable WebGPU preprocessing** unchecked (the default) unless
   you've specifically tested it on this machine (see below) — OCR reads
   the plain camera crop as-is, which is what's been tested most. If you do
   enable it, the Contrast/Brightness/Binarize/Threshold sliders start
   doing something; tune them against the "WebGPU preprocessed" preview
   until bib numbers look like clean, high-contrast digits.
6. **Run Performance Check** and read the result: it tells you whether this
   laptop/camera keeps up with the configured Interval, offers to set a
   safe one, and estimates GB/hour so you know how much disk space you
   need. Do this once per venue/lighting setup, not necessarily every
   session.
7. **Start Session.** Confirm the status line says "Session running" and the
   elapsed timer is counting up. Leave "beep on new rider" checked — it's
   your live confirmation that riders are actually being logged; if you go
   a suspiciously long stretch of racing without hearing it, check the
   preview panels.
8. Let it run. The live log table (right side) fills in as riders are
   confirmed; a row briefly highlights when it's added. **If a red banner
   appears at the top of the page**, see Troubleshooting below — don't just
   keep racing past it.

## Stopping & wrapping up

1. **Stop Session** once the group/session is done.
2. If you used a save folder: the footage and log are already on disk (the
   status line names the exact files) — no download needed, though the
   JSON/CSV download buttons are still there for convenience.
3. If you didn't use a save folder: click **Download Footage**, **Download
   Log (JSON)**, and **Download Log (CSV)** right away — they won't survive
   a page reload.
4. **Back up immediately** — copy the session's files off the laptop (USB
   stick, phone, cloud sync once you have signal again) before starting the
   next group.

## After-action review

Open `review.html`, load the footage file and the JSON log for a session:

- Click **First**/**Last** next to any entry to jump the footage to that
  moment and visually confirm the number.
- If OCR misread a number, **edit it directly** in the table.
- Check the box in the **OK** column once you've confirmed an entry against
  the footage.
- **Export corrected JSON/CSV** when done — this is the file to use for
  final results, not the raw capture output.

Pay closest attention to **close finishes and any overtakes** near the
reading line — see the "overtake caveat" in the main README. When in doubt,
scrub the footage manually; it's there for exactly this.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| "Camera error" on Start Camera | Another app (Zoom, another browser tab) is using the camera — close it. Or camera permission was denied — check the browser's site settings and reload. |
| **A red banner appears across the top of the page** | This is the detection hard-stop: the app noticed camera frames stopped reaching the detection canvas and refuses to keep silently logging nothing. **Your footage is still recording** — finish what you're doing, click **Stop Session** to save it, then **reload the page** and start a fresh session for the rest of the race. This is the one alert that means stop and act, not just a note. |
| No riders ever appear in the live log | ROI probably doesn't cover where numbers are legible. Check the "ROI crop" preview — it should show legible bib numbers passing through. (If you've enabled WebGPU preprocessing, also check the second preview looks like clean high-contrast digits, not a grey blur.) |
| Numbers logged, but often wrong digits | Faded/dirty bibs or motion blur cause some misses regardless of tuning — that's what Review is for. If you've enabled WebGPU preprocessing, try increasing Contrast or Binarize; those controls do nothing with it off. |
| "Failed to load OCR engine" | The page isn't being served from the `bike-race-timing/` folder (so `vendor/tesseract/` 404s), or you're opening the HTML file directly (`file://`) instead of via a local server. Restart the local static server from the correct folder. |
| "Choose Save Folder" is greyed out | Your browser doesn't support the File System Access API (Firefox/Safari). Not fatal — footage will just download at the end instead of streaming live. Switch to Chrome/Edge if crash-safety matters more than convenience. |
| App/tab crashed mid-session | If you'd chosen a save folder: the `.webm` and `.jsonl` files on disk are intact up to the last second or so before the crash — open them directly in Review (most players handle a slightly-truncated webm fine). If you hadn't: the in-memory footage for that session is gone; this is exactly why picking a save folder is worth doing. |
| Laptop fan spinning hard / getting warm | Expected — camera capture, continuous OCR, and video encoding is real, sustained work. Make sure it's not also on a lap blocking airflow. |
| WebGPU preprocessing pill says "disabled after errors" mid-session | Expected self-healing, not a bug: it detected a problem and switched to the CPU path automatically — detection keeps running normally. Leave WebGPU off for the rest of the day on this machine. |

## Known limitations (read once, then don't worry about it)

- This is a heuristic aid, not a certified photo-finish system. Treat close
  or contested finishes as "needs manual review," not "trust the log."
- Overtakes near the line can occasionally log in the wrong order — see the
  README's "overtake caveat" section for exactly why, and why the effect is
  usually in the *safer* direction (favoring the rider being overtaken).
- Only digits are read by default (bib numbers with letters won't match).
- WebGPU preprocessing is experimental and off by default for exactly this
  reason — it's an optional accuracy boost, not something the app depends
  on. Everything has been tested with it off.
