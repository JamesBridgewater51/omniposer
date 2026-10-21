/* ============================================================================
   Hero video wall — full-bleed wall of looping evaluation clips.
   The full grid shows immediately (no camera intro): every clip loops
   independently with a phase offset; only off-screen decoding is paused.
   Native video composition owns steady playback; JS does not draw each frame.
   Canvas renderers remain only for diagnostics/static fallback. On scroll the whole hero
   (wall + scrim + overlay) shrinks to half height while the next section
   arrives; the title overlay fades out early so it never lingers tiny.
   Layout comes from static/videos/wall.json, built by tools/build_wall.py —
   no asset is hardcoded here.

   Long-visit playback supervision (the part that must not rot):
   A wall visit lasts for hours, and browsers keep throwing recoverable
   trouble at it — network blips, decoder stalls around native loop wraps,
   background suspensions, autoplay hiccups. The watchdog therefore has:
     • no cumulative retry budget — scattered stalls never sum up;
     • no absorbing failed state — broken clips keep being probed;
     • repair escalation: play() restart → source reload → new element;
     • success declared only on measured fresh frames, never on a resolved
       play() promise; a healthy interval fully replenishes the budget;
     • probes on a capped, staggered, exponential cooldown, so even a
       permanently dead source costs at most ~1 probe/min per cell.
   Liveness prefers per-frame presentation callbacks (rVFC); decoded-frame
   counters and the media clock remain as explicitly weaker fallbacks.
   ========================================================================== */

(function () {
  'use strict';

  const cv = document.getElementById('hero-wall-canvas');
  const driver = document.querySelector('.hw-driver');
  if (!cv || !driver) return;
  const scaleBox = document.getElementById('hw-scale');
  const overlay = document.getElementById('hw-overlay');
  const hint = document.querySelector('.hw-hint');
  const navbar = document.getElementById('op-navbar');

  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const easeInOut = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const MOBILE = () => window.matchMedia('(max-width: 719px)').matches;

  const UNIT = 240;                    // world px per grid unit (matches build_wall.py)
  const SHRINK = 0.52;                 // final wall height fraction on scroll
  const VIEW_MARGIN = 0.6;             // units of culling margin around the view
  const DRAW_INTERVAL = 1000 / 30;    // paint at clip cadence, not display refresh
  const CHECK_INTERVAL = 1000;         // watchdog cadence — repairs are never per-rAF
  const STALL_TIMEOUT = 8000;          // no fresh frames this long → repair
  const HEALTHY_INTERVAL = 30000;      // fresh frames this long → replenish budget
  const COOLDOWN_BASE = 2000;          // probe backoff base
  const COOLDOWN_MAX = 60000;          // probe backoff cap while wanted
  const RING_MAX = 240;                // bounded diagnostic ring (?hw_debug only)
  const DEBUG = new URLSearchParams(location.search).has('hw_debug');
  const DIRECT = !REDUCED && new URLSearchParams(location.search).get('hw_renderer') === 'direct';
  const NATIVE = !REDUCED && !['cache', 'direct'].includes(new URLSearchParams(location.search).get('hw_renderer'));
  let nativeLayer = null;
  const TITLE_FADE = [0.22, 0.62];     // scroll-progress range over which the overlay fades

  let manifest = null;
  let cells = [];                      // {src,dataset,err,col,row,cols,rows, video}
  let worldW = 0, worldH = 0;
  let ctx = null, cw = 0, chh = 0;
  let viewH = 0, viewW = 0;            // full-wall fit, recomputed on resize
  let lastDraw = -Infinity, lastCheck = -Infinity;
  let dirty = true, draws = 0;
  let geometryDirty = true;
  let inView = false, pageActive = true;

  /* ------------------------------------------------------------- plumbing */

  function strHash(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return (h >>> 0) / 4294967295;
  }

  function resize() {
    if (DIRECT) retainDirectFrames(true);
    geometryDirty = true;
    // Keep CSS/world geometry unchanged while avoiding a 4x backing-store cost.
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    dirty = true;
    cw = cv.clientWidth; chh = cv.clientHeight;
    cv.width = Math.max(1, Math.round(cw * dpr));
    cv.height = Math.max(1, Math.round(chh * dpr));
    ctx = cv.getContext('2d');
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // COVER fit: the wall world always fills the canvas edge-to-edge and a
    // thin margin of world crops at the sides — the viewport is never black
    const vpAsp = cw / Math.max(chh, 1);
    const wAsp = worldW / Math.max(worldH, 1);
    if (vpAsp >= wAsp) { viewW = worldW / 1.02; viewH = viewW / vpAsp; }
    else               { viewH = worldH / 1.02; viewW = viewH * vpAsp; }
  }

  /* ------------------------------------------- liveness (fresh-frame proof)
     Strongest signal first: requestVideoFrameCallback fires only when a new
     frame is actually presented. Quality counters and the media clock are
     weaker fallbacks (labelled per cell in diagnostics) — they can advance
     while the picture is frozen, so they are never called "fresh" once the
     presentation callback has proven itself live (rvfcLive).               */

  function decodedFrames(el) {
    if (!el.getVideoPlaybackQuality) return null;
    const quality = el.getVideoPlaybackQuality();
    return quality.totalVideoFrames - (quality.droppedVideoFrames || 0);
  }

  function noteFresh(cell, now, signal) {
    cell.lastFresh = now;
    cell.signal = signal;
    if (signal === 'rvfc') cell.rvfcLive = true;
    if (cell.status === 'starting') {
      // Measured success: fresh frames arrived after the attempt began.
      cell.active = true;
      cell.recoveredAt = now;
      setStatus(cell, 'playing', 'fresh frames (' + signal + ')');
      dirty = true;
    }
  }

  function measureFallback(cell, el, now) {
    if (NATIVE) {
      const frames = decodedFrames(el), clock = el.currentTime;
      if (el.readyState >= 2 && !el.seeking) {
        if (frames !== null && frames !== cell.nativeObservedFrames)
          noteFresh(cell, now, 'native-counter');
        else if (isFinite(clock) && Math.abs(clock - cell.nativeObservedTime) > 0.01)
          noteFresh(cell, now, 'native-clock'); // weaker liveness, not screen proof
      }
      cell.nativeObservedFrames = frames;
      cell.nativeObservedTime = clock;
      return;
    }
    if (DIRECT && cell.directFrame && cell.directFrame.error) return;
    if (cell.frameCache && cell.frameCache.error) return;
    if (cell.rvfcLive && hasFrameWatch(cell)) return;        // presentation callbacks own liveness once live
    const frames = decodedFrames(el);
    if (frames !== null) {
      if (frames > cell.lastFrames) noteFresh(cell, now, 'quality');
      cell.lastFrames = frames;
    } else if (Math.abs(el.currentTime - cell.lastTime) > 0.01) {
      noteFresh(cell, now, 'clock');
    }
    cell.lastTime = el.currentTime;
  }

  function armFrameWatch(cell) {
    const el = cell.video;
    if (!el || typeof el.requestVideoFrameCallback !== 'function') return;
    if (cell.watchGen === cell.gen) return;          // already armed for this element
    cell.watchGen = cell.gen;
    cell.watchEpoch = (cell.watchEpoch || 0) + 1;
    const epoch = cell.watchEpoch;
    const el0 = el, gen = cell.gen;
    const tick = (t, metadata) => {
      if (cell.gen !== gen || cell.video !== el0 || cell.watchEpoch !== epoch) return;   // stale after replacement
      cell.presented = (cell.presented || 0) + 1;
      cell.nativePresentedFrames = metadata ? metadata.presentedFrames : null;
      cell.presentedTime = metadata && isFinite(metadata.mediaTime) ? metadata.mediaTime : null;
      if (captureCellFrame(cell, el0, gen, metadata))
        noteFresh(cell, performance.now(), 'rvfc');
      if (!NATIVE) dirty = true;
      try { cell.watchId = el0.requestVideoFrameCallback(tick); }
      catch (e) { cell.watchGen = 0; }
    };
    try { cell.watchId = el.requestVideoFrameCallback(tick); }
    catch (e) { cell.watchGen = 0; }
  }

  function rearmFrameWatch(cell) {
    // After el.load() pending presentation callbacks are cancelled by the
    // engine; re-arm so reload attempts still get fresh-frame proof.
    const el = cell.video;
    if (cell.watchId != null && el && typeof el.cancelVideoFrameCallback === 'function') {
      try { el.cancelVideoFrameCallback(cell.watchId); } catch (e) { /* best effort */ }
    }
    cell.watchId = null;
    cell.watchGen = 0;
    armFrameWatch(cell);
  }

  /* -------------------------------------------- per-cell playback machine
     States:  idle → starting → playing ⇄ cooldown → starting → …
     'starting' covers the first start and every repair; it ends ONLY on
     fresh frames (never on a resolved play promise). 'cooldown' waits out
     a staggered, capped retry deadline. There is no terminal state.       */

  const ring = DEBUG ? [] : null;
  function note(cell, kind, why) {
    if (!ring) return;
    ring.push({ t: Math.round(performance.now()), cell: cell ? cell.src : null,
                state: cell ? cell.status : null, kind, why: why || null });
    if (ring.length > RING_MAX) ring.splice(0, ring.length - RING_MAX);
  }

  function setStatus(cell, to, why) {
    if (cell.status === to) return;
    note(cell, 'state ' + cell.status + '→' + to, why);
    cell.status = to;
  }

  function jitter(cell) { return strHash(cell.src) * 4000; }   // 0..~4s probe stagger

  function backoff(cell) {
    // Exponential but capped: hours of failure stay at ≤1 probe/min.
    const exp = Math.pow(2, clamp(cell.consecutive - 1, 0, 5));
    return Math.min(COOLDOWN_BASE * exp, COOLDOWN_MAX);
  }

  function invalidate(cell) {          // pending play()/frame callbacks die here
    cell.attemptToken = (cell.attemptToken || 0) + 1;
    cell.pending = false;
  }

  function resetCycle(cell, now) {     // a fresh bounded visit: full budget
    cell.attempts = 0;                 // repair attempts this cycle
    cell.consecutive = 0;              // consecutive failures this cycle
    cell.level = 0;                    // repair ladder: 0 restart, 1 reload, 2 recreate
    cell.recoveredAt = 0;              // when fresh frames last became healthy
    cell.seekAt = 0;
    cell.retryAt = 0;                  // start immediately; only failed probes stagger
    cell.recreateWarned = false;
    if (cell.video && cell.video.error) cell.level = 1;  // an errored element needs a reload
  }

  function replenish(cell) {
    // Only measured health (fresh frames for HEALTHY_INTERVAL) replenishes
    // the failure budget — never a play-promise resolution.
    cell.consecutive = 0;
    cell.level = 0;
    cell.recoveredAt = 0;
    cell.recreateWarned = false;
    note(cell, 'replenish', 'healthy interval');
  }

  function canRun() {
    return inView && pageActive && !document.hidden;
  }

  function makeVideoElement(cell) {
    const el = document.createElement('video');
    el.muted = true; el.loop = true; el.playsInline = true;
    el.preload = REDUCED ? 'auto' : 'none';   // reduced: load the phase frame once
    el.setAttribute('playsinline', '');
    el.src = cell.src;
    el._heroWallSource = cell.src;
    if (NATIVE) attachNativeVideo(cell, el);
    const phase = strHash(cell.phaseSeed || cell.src);
    const gen = cell.gen = (cell.gen || 0) + 1;   // invalidates prior listeners
    cell.video = el;
    if (NATIVE) {
      cell.nativeObservedFrames = decodedFrames(el);
      cell.nativeObservedTime = el.currentTime;
    }
    cell.staticReady = false;
    cell.watchGen = 0; cell.watchId = null; cell.rvfcLive = false;
    if (!cell.status || cell.status === 'paused') {
      if (REDUCED) beginStaticLoad(cell, performance.now(), 'element created');
      else cell.status = 'idle';
    }
    const live = fn => () => { if (cell.video !== el) return; fn(); };
    const ready = live(() => {
      if (el.readyState >= 2 && !el.seeking) {
        cell.staticReady = true;
        if (REDUCED) { setStatus(cell, 'static', null); cell.consecutive = 0; }
        dirty = true;
      }
    });
    el.addEventListener('loadedmetadata', live(() => {
      if (isFinite(el.duration) && el.duration > 0) {
        try {
          cell.staticReady = false;
          el.currentTime = phase * el.duration * 0.96;
        } catch (e) { cell.lastError = 'phase seek: ' + e.message; }
      }
      ready();
    }));
    el.addEventListener('loadeddata', ready);
    el.addEventListener('seeked', live(() => { cell.seekAt = 0; ready(); }));
    el.addEventListener('seeking', live(() => { if (cell.wanted) cell.seekAt = performance.now(); }));
    el.addEventListener('error', live(() => onMediaError(cell, el)));
    return el;
  }

  function recreateElement(cell) {
    const old = cell.video;
    if (NATIVE) {
      cell.video = null;
      cell.watchEpoch = (cell.watchEpoch || 0) + 1;
      if (old && cell.watchId != null && old.cancelVideoFrameCallback) {
        try { old.cancelVideoFrameCallback(cell.watchId); } catch (e) { /* retired */ }
      }
    }
    invalidate(cell);                  // stale play()/frame callbacks die here
    note(cell, 'recreate', 'level 2');
    if (!cell.recreateWarned) {        // once per failure cycle, not once a minute
      cell.recreateWarned = true;
      console.warn('Hero wall recreating element for:', cell.src, cell.lastError || '');
    }
    if (old) {
      try { old.pause(); } catch (e) { /* discarded */ }
      try { if (typeof old.removeAttribute === 'function') old.removeAttribute('src'); } catch (e) { /* discarded */ }
      try { if (typeof old.load === 'function') old.load(); } catch (e) { /* discarded */ }  // release decoder
    }
    if (NATIVE && old && old.parentNode) old.parentNode.removeChild(old);
    makeVideoElement(cell);
  }

  function onMediaError(cell, el) {
    if (el !== cell.video) return;
    const reason = 'media error ' + (el.error ? el.error.code : 'unknown');
    note(cell, 'media-error', reason);
    if (REDUCED) {
      invalidate(cell);
      cell.staticReady = false;
      cell.lastError = reason;
      cell.consecutive = (cell.consecutive || 0) + 1;
      cell.retryAt = performance.now() + backoff(cell) + jitter(cell);
      setStatus(cell, 'cooldown', reason);
      if (cell.consecutive === 1) console.warn('Hero wall static frame failed:', cell.src, reason);
      dirty = true;
      return;
    }
    if (cell.status === 'starting' || cell.status === 'playing' || cell.status === 'idle') {
      // A plain play() cannot clear an error state — escalate to reload now.
      failAttempt(cell, performance.now(), reason, { minLevel: 1 });
    } else if (cell.status === 'cooldown') {
      cell.consecutive = (cell.consecutive || 0) + 1;
      cell.level = Math.max(cell.level, 1);
      cell.lastError = reason;
    }
  }

  function failAttempt(cell, now, reason, opts = {}) {
    // A repair attempt failed. Escalate the ladder, back off, try again.
    invalidate(cell);
    cell.active = false;
    cell.lastError = reason;
    cell.consecutive = (cell.consecutive || 0) + 1;
    cell.recoveredAt = 0;
    const next = clamp(Math.max(cell.level + 1, opts.minLevel || 0), 0, 2);
    if (next !== cell.level) console.warn('Hero wall clip repair escalated to level ' + next + ':', cell.src, reason);
    else if (cell.consecutive === 1) console.warn('Hero wall clip failed:', cell.src, reason);
    cell.level = next;
    cell.seekAt = 0;                      // seek supervision restarts per attempt
    setStatus(cell, 'cooldown', reason);
    cell.retryAt = now + backoff(cell) + jitter(cell);   // staggered, capped probe deadline
    try { cell.video.pause(); } catch (e) { /* already gone */ }
    note(cell, 'fail', reason);
    dirty = true;
  }

  function stallFromPlaying(cell, now, reason) {
    // The current attempt lost fresh frames while not seeking. Give the cheap
    // restart (pause → play) one chance on a first consecutive failure; if the
    // very next attempt also stalls, escalate the ladder — a pipeline that
    // cannot be restarted needs a reload, then a fresh element.
    invalidate(cell);
    cell.active = false;
    cell.lastError = reason;
    cell.consecutive = (cell.consecutive || 0) + 1;
    cell.recoveredAt = 0;
    cell.seekAt = 0;                      // seek supervision restarts per attempt
    const next = cell.consecutive >= 2 ? clamp(cell.level + 1, 0, 2) : cell.level;
    if (next !== cell.level) console.warn('Hero wall clip repair escalated to level ' + next + ':', cell.src, reason);
    else if (cell.consecutive === 1) console.warn('Hero wall clip stalled, repairing:', cell.src, reason);
    cell.level = next;
    setStatus(cell, 'cooldown', reason);
    cell.retryAt = now + backoff(cell) + jitter(cell) * 0.5;   // staggered probes
    try { cell.video.pause(); } catch (e) { /* already gone */ }
    note(cell, 'stall', reason);
    dirty = true;
  }

  const stallBudget = cell => (cell.level || 0) > 0 ? STALL_TIMEOUT * 2 : STALL_TIMEOUT;

  function beginStaticLoad(cell, now, why) {
    cell.staticAt = now;
    setStatus(cell, 'loading-static', why);
  }

  function beginAttempt(cell, now) {
    if (!cell.wanted || !canRun() || cell.pending || now < (cell.retryAt || 0)) return false;
    const el = cell.video;
    if (!el) return false;
    if (cell.level >= 2) recreateElement(cell);             // ladder step 3: fresh pipeline
    else if (cell.level === 1) {                            // ladder step 2: reload source
      invalidate(cell);
      try { el.pause(); } catch (e) { /* reload continues */ }
      try { if (typeof el.load === 'function') el.load(); } catch (e) { /* play may still work */ }
      rearmFrameWatch(cell);
    }
    const video = cell.video;
    const token = cell.attemptToken = (cell.attemptToken || 0) + 1;
    cell.attempts++;
    cell.pending = true;
    cell.active = false;
    cell.attemptAt = now;
    cell.lastFresh = now;                 // deadline is measured from attempt start
    cell.seekAt = 0;                      // fresh attempt, fresh seek supervision
    cell.lastFrames = decodedFrames(video);
    cell.lastTime = video.currentTime;
    setStatus(cell, 'starting', 'repair level ' + cell.level + (cell.level ? ' (extended first-frame budget)' : ''));
    note(cell, 'attempt', 'level ' + cell.level);
    armFrameWatch(cell);
    const settled = () => {
      if (cell.attemptToken !== token || cell.video !== video) {
        // Stale completion after replacement/failure: never resurrect. If a
        // newer attempt owns this element, leave it alone; otherwise bring
        // the element back in line with the machine (stopped in cooldown).
        if (cell.video !== video || !(cell.wanted && canRun() && (cell.status === 'starting' || cell.status === 'playing'))) { try { video.pause(); } catch (e) { /* discarded */ } }
        return;
      }
      cell.pending = false;
      if (!cell.wanted || !canRun()) { stopCell(cell); return; }
      if (video.paused) { failAttempt(cell, performance.now(), 'play resolved while paused'); return; }
      // Success is declared only on measured fresh frames — never here.
    };
    const rejected = error => {
      if (cell.attemptToken !== token || cell.video !== video) return;
      failAttempt(cell, performance.now(),
                  error && error.name ? error.name + ': ' + error.message : 'play rejected');
    };
    try {
      const result = video.play();
      if (result && result.then) result.then(settled, rejected);
      else settled();
    } catch (error) { rejected(error); }
    return true;
  }

  function stopCell(cell) {
    if (!cell.video) return;
    invalidate(cell);
    cell.active = false;
    cell.video.pause();
    cell.seekAt = 0;
    if (REDUCED) {
      // Pause the static-load deadline too: hidden time must not read as a
      // stall. refreshVisibility re-arms the load on return.
      if (cell.status === 'loading-static') cell.staticAt = null;
    } else if (cell.status !== 'static') setStatus(cell, 'paused', 'off-screen/hidden');
  }

  function checkPlayback(cell, now) {
    if (!cell.video) return;
    if (REDUCED) {                               // genuine static mode: never plays
      if (!cell.wanted || !canRun()) return;
      if (cell.status === 'loading-static' && cell.staticAt != null) {
        // Supervise the metadata/phase-seek load itself, mirroring the
        // animated path: a hung preload cannot wedge a cell forever.
        const ready = cell.staticReady || (cell.video.readyState >= 2 && !cell.video.seeking);
        if (ready) {
          cell.staticAt = null;
          if (!cell.staticReady) {   // events were missed; accept the measured state
            cell.staticReady = true;
            setStatus(cell, 'static', 'ready');
            cell.consecutive = 0;
            dirty = true;
          }
          return;
        }
        if (now - cell.staticAt >= stallBudget(cell)) {
          cell.attempts++;
          cell.lastError = 'static load timed out';
          cell.consecutive = (cell.consecutive || 0) + 1;
          setStatus(cell, 'cooldown', 'static load timed out');
          cell.retryAt = now + backoff(cell) + jitter(cell);
          note(cell, 'stall', 'static load timed out');
          try { if (typeof cell.video.load === 'function') cell.video.load(); } catch (e) { /* retried later */ }
        }
        return;
      }
      if (cell.status === 'cooldown' && now >= (cell.retryAt || 0)) {
        // Bounded reload probes: a transient error loading the phase frame
        // recovers by itself without any visibility toggle.
        cell.attempts++;
        cell.lastError = null;
        beginStaticLoad(cell, now, 'reload probe');
        note(cell, 'attempt', 'static reload');
        cell.retryAt = now + backoff(cell) + jitter(cell);
        try { if (typeof cell.video.load === 'function') cell.video.load(); } catch (e) { /* retried later */ }
      }
      return;
    }
    if (!cell.wanted || !canRun()) return;
    const el = cell.video;
    if (cell.status === 'idle' || cell.status === 'cooldown' || cell.status === 'paused') {
      beginAttempt(cell, now);                   // gated by retryAt: staggered, capped
      return;
    }
    if (cell.status === 'starting') {
      measureFallback(cell, el, now);
      if (cell.status !== 'starting') return;    // fresh frames arrived
      if (el.seeking) {                          // phase seek / loop seek in progress
        if (!cell.seekAt) cell.seekAt = now;
        else if (now - cell.seekAt >= stallBudget(cell)) failAttempt(cell, now, 'stuck seek while starting', { minLevel: 1 });
        return;
      }
      cell.seekAt = 0;
      if (now - cell.attemptAt >= stallBudget(cell)) {
        failAttempt(cell, now, cell.pending ? 'play timed out'
                  : (el.paused ? 'start stalled (paused)' : 'no fresh frames after start'));
      }
      return;
    }
    if (cell.status === 'playing') {
      measureFallback(cell, el, now);
      if (el.seeking) {
        // Loop/phase seeks are supervised by their own seekAt deadline; frame
        // progress legitimately pauses during a seek, so the generic frame
        // stall check must not fire under it.
        if (!cell.seekAt) cell.seekAt = now;
        else if (now - cell.seekAt >= stallBudget(cell)) { failAttempt(cell, now, 'stuck seek', { minLevel: 1 }); return; }
        return;
      }
      cell.seekAt = 0;
      if (now - cell.lastFresh >= STALL_TIMEOUT) {
        stallFromPlaying(cell, now,
                         el.paused ? 'unexpected pause' : 'no fresh frames (' + (cell.signal || 'none') + ')');
      } else if ((cell.consecutive || cell.level || cell.attempts > 1) && cell.recoveredAt &&
                 now - cell.recoveredAt >= HEALTHY_INTERVAL) {
        replenish(cell);                         // measured health restores the budget
      }
    }
  }
  function setInView(value) {
    if (inView === value) return;
    inView = value;
    refreshVisibility();
  }

  function refreshVisibility() {
    const run = canRun();
    cells.forEach(cell => {
      if (!run) {
        stopCell(cell);
        cell.wanted = false;
        return;
      }
      if (!cell.video) return;
      if (REDUCED) {
        // A real return from background/off-screen gives the static frame a
        // fresh bounded cycle; already-ready frames are kept as-is. A load
        // paused mid-flight by stopCell is re-armed with a fresh deadline.
        if (cell.status === 'cooldown') {
          resetCycle(cell, performance.now());
          beginStaticLoad(cell, performance.now(), 'retry on return');
        } else if (cell.status === 'loading-static' && cell.staticAt == null) {
          beginStaticLoad(cell, performance.now(), 'resume load on return');
        }
        return;
      }
      if (cell.status === 'starting') return;                       // attempt owns it
      if (cell.status === 'playing' && !cell.video.paused) return;  // healthy
      resetCycle(cell, performance.now());
      setStatus(cell, 'idle', 'visible again');
    });
    lastCheck = -Infinity;
    dirty = true;
  }

  /* --------------------------------------------------------------- render */

  function hasFrameWatch(cell) {
    return !!(cell.video && typeof cell.video.requestVideoFrameCallback === 'function' && cell.watchGen === cell.gen);
  }

  // Geometry-only safety snapshots; never a per-frame presentation stage.
  function attachNativeVideo(cell, el) {
    if (!nativeLayer) {
      nativeLayer = document.createElement('div');
      nativeLayer.id = 'hero-wall-native-layer';
      nativeLayer.style.cssText = 'position:absolute;inset:0;overflow:hidden;pointer-events:none;z-index:0;background:#080a0e';
      cv.parentNode.style.position = 'relative';
      cv.parentNode.insertBefore(nativeLayer, cv.nextSibling);
      cv.style.visibility = 'hidden';  // keep canvas box geometry, never paint it
    }
    el.style.cssText = 'position:absolute;display:block;visibility:hidden;pointer-events:none;margin:0;max-width:none;max-height:none';
    el.style.objectFit = cell.fit === 'contain' ? 'contain' : 'cover';
    const tone = 16 + Math.round(strHash(cell.src) * 12);
    el.style.background = cell.background || `rgb(${tone},${tone + 2},${tone + 5})`;
    nativeLayer.appendChild(el);        // this IS the owned decoder, not a mirror
  }

  function positionNative(cell, sx, sy, sw, sh, changed) {
    const el = cell.video;
    if (!el) return;
    if (changed || el._nativeLayoutGen !== cell.gen) {
      el.style.left = sx + 'px'; el.style.top = sy + 'px';
      el.style.width = sw + 'px'; el.style.height = sh + 'px';
      el._nativeLayoutGen = cell.gen;
    }
    if (el.style.visibility !== 'visible') el.style.visibility = 'visible';
  }

  function retainDirectFrames(force) {
    if (!ctx || !cw || !cv.width || !cv.height) return;
    const dpr = cv.width / cw;
    for (const cell of cells) {
      const d = cell.directFrame, r = d && d.rect, el = cell.video;
      if (!d || !r || !(r.sw > 0 && r.sh > 0) ||
          (!force && el && el.readyState >= 2 && !el.seeking)) continue;
      try {
        const image = document.createElement('canvas');
        image.width = Math.max(1, Math.round(r.sw * dpr));
        image.height = Math.max(1, Math.round(r.sh * dpr));
        image.getContext('2d').drawImage(cv, r.sx * dpr, r.sy * dpr,
          r.sw * dpr, r.sh * dpr, 0, 0, image.width, image.height);
        d.retained = image;
        d.retainedSource = cell.src;
      } catch (e) { /* retain the previous successful snapshot */ }
    }
  }

  function paintRetained(cell) {
    const d = cell.directFrame, r = cell.paintRect;
    if (!ctx || !r || !d || !d.retained || d.retainedSource !== cell.src) return false;
    ctx.drawImage(d.retained, r.sx, r.sy, r.sw, r.sh);
    return true;                        // pixels only, never fresh-frame proof
  }

  function paintDirect(cell, el, gen, metadata) {
    const r = cell.paintRect;
    if (!ctx || !r || !el || !cell.wanted || !canRun() || cell.video !== el ||
        cell.gen !== gen || el._heroWallSource !== cell.src ||
        el.readyState < 2 || el.seeking) return false;
    const vw = el.videoWidth, vh = el.videoHeight;
    if (!(vw > 0 && vh > 0 && r.sw > 0 && r.sh > 0)) return false;
    ctx.save();
    try {
      ctx.beginPath(); ctx.rect(r.sx, r.sy, r.sw, r.sh); ctx.clip();
      const tone = 16 + Math.round(strHash(cell.src) * 12);
      ctx.fillStyle = cell.background || `rgb(${tone},${tone + 2},${tone + 5})`;
      ctx.fillRect(r.sx, r.sy, r.sw, r.sh);
      if (cell.fit === 'contain') {
        const scale = Math.min(r.sw / vw, r.sh / vh);
        const w = vw * scale, h = vh * scale;
        ctx.drawImage(el, 0, 0, vw, vh,
          r.sx + (r.sw - w) / 2, r.sy + (r.sh - h) / 2, w, h);
      } else {
        const sourceAspect = vw / vh, destAspect = r.sw / r.sh;
        let sx = 0, sy = 0, sw = vw, sh = vh;
        if (sourceAspect > destAspect) { sw = vh * destAspect; sx = (vw - sw) / 2; }
        else { sh = vw / destAspect; sy = (vh - sh) / 2; }
        ctx.drawImage(el, sx, sy, sw, sh, r.sx, r.sy, r.sw, r.sh);
      }
      const d = cell.directFrame || (cell.directFrame = {});
      d.version = (d.version || 0) + 1;
      d.gen = gen; d.clock = el.currentTime; d.qualityFrames = decodedFrames(el);
      d.mediaTime = metadata && Number.isFinite(metadata.mediaTime) ? metadata.mediaTime : null;
      d.nativePresentedFrames = metadata ? metadata.presentedFrames : null;
      d.drawEnd = performance.now();      // submission, not physical screen time
      d.rect = { sx: r.sx, sy: r.sy, sw: r.sw, sh: r.sh };
      d.error = null;
      if (DEBUG) cell.draws = (cell.draws || 0) + 1;
      return true;
    } catch (error) {
      const d = cell.directFrame || (cell.directFrame = {});
      d.error = String(error && error.message || error);
      cell.drawFails = (cell.drawFails || 0) + 1;
      if (cell.drawFails === 1) console.warn('Hero wall direct draw failed:', cell.src, error);
      paintRetained(cell);
      return false;
    } finally { ctx.restore(); }
  }

  function directNeedsPaint(cell, paintRequested, fallbackDue) {
    const el = cell.video, d = cell.directFrame;
    if (!el || el.readyState < 2 || el.seeking) return false;
    if (!d || d.gen !== cell.gen || d.error) return true;
    if (hasFrameWatch(cell) || !fallbackDue) return false;
    const frames = decodedFrames(el);
    return frames !== null ? frames !== d.qualityFrames : el.currentTime !== d.clock;
  }

  function frameCache(cell) {
    if (!cell.frameCache || cell.frameCache.source !== cell.src) {
      cell.frameCache = { source: cell.src, front: null, back: null,
        width: 0, height: 0, version: 0, mediaTime: null, ownerGen: 0,
        capturedAt: 0, paintedVersion: 0, failures: 0, error: null };
    }
    return cell.frameCache;
  }

  function captureCellFrame(cell, el, gen, metadata, requestPaint = true) {
    if (NATIVE) {
      if (cell.video !== el || cell.gen !== gen || el._heroWallSource !== cell.src || el.readyState < 2 || el.seeking) return false;
      cell.nativeFrames = (cell.nativeFrames || 0) + 1;
      return true;                      // browser-video submission, no canvas copy
    }
    if (DIRECT) {
      if (geometryDirty) return false;
      return paintDirect(cell, el, gen, metadata);
    }
    const cache = frameCache(cell);
    if (cell.video !== el || cell.gen !== gen || el._heroWallSource !== cell.src ||
        el.readyState < 2 || el.seeking || !cache.width || !cache.height) return false;
    try {
      const back = cache.back || document.createElement('canvas');
      cache.back = back;
      if (back.width !== cache.width) back.width = cache.width;
      if (back.height !== cache.height) back.height = cache.height;
      const context = back.getContext('2d', { alpha: false });
      if (!context) throw new Error('frame cache context unavailable');
      context.drawImage(el, 0, 0, cache.width, cache.height);
      const old = cache.front;
      cache.front = back;
      cache.back = old;
      cache.version++;
      cache.ownerGen = gen;
      cache.mediaTime = metadata && isFinite(metadata.mediaTime) ? metadata.mediaTime : null;
      cache.clock = el.currentTime;
      cache.qualityFrames = decodedFrames(el);
      cache.capturedAt = performance.now();
      cache.error = null;
      if (requestPaint) dirty = true;
      return true;
    } catch (error) {
      cache.failures++;
      cache.error = String(error && error.message || error);
      // The current front remains valid; only the failed spare is discarded.
      if (cache.back && cache.back !== cache.front) {
        cache.back.width = cache.back.height = 0;
        cache.back = null;
      }
      return false;
    }
  }

  function drawCover(cell, dx, dy, dw, dh) {
    if (DIRECT) {
      if (paintDirect(cell, cell.video, cell.gen, null)) return true;
      return paintRetained(cell);
    }
    const el = cell.video, cache = frameCache(cell);
    const vw = el && el.videoWidth || cell.w || 0;
    const vh = el && el.videoHeight || cell.h || 0;
    if (vw > 0 && vh > 0) {
      const contain = cell.fit === 'contain';
      const scale = contain ? Math.min(dw / vw, dh / vh) : Math.max(dw / vw, dh / vh);
      const dpr = Math.min(1.5, Math.max(1, cv.width / Math.max(cw, 1)));
      const factor = Math.min(1, scale * dpr);
      cache.width = Math.max(1, Math.round(vw * factor));
      cache.height = Math.max(1, Math.round(vh * factor));
    }
    if (el && el.readyState >= 2 && !el.seeking &&
        (!cache.front || (!hasFrameWatch(cell) &&
          (decodedFrames(el) !== null ? cache.qualityFrames !== decodedFrames(el) : cache.clock !== el.currentTime)))) {
      // Initial/unsupported-browser capture is explicitly not an exact rVFC timestamp.
      captureCellFrame(cell, el, cell.gen, null, false);
    }
    if (!cache.front) return false;
    const width = cache.front.width, height = cache.front.height;
    if (cell.fit === 'contain') {
      const scale = Math.min(dw / width, dh / height);
      const w = width * scale, h = height * scale;
      ctx.drawImage(cache.front, 0, 0, width, height,
        dx + (dw - w) / 2, dy + (dh - h) / 2, w, h);
    } else {
      const sAsp = width / height, dAsp = dw / dh;
      let sx = 0, sy = 0, sw = width, sh = height;
      if (sAsp > dAsp) { sw = height * dAsp; sx = (width - sw) / 2; }
      else { sh = width / dAsp; sy = (height - sh) / 2; }
      ctx.drawImage(cache.front, sx, sy, sw, sh, dx, dy, dw, dh);
    }
    cache.paintedVersion = cache.version;
    if (DEBUG) cell.draws = (cell.draws || 0) + 1;
    return true;
  }

  function frame(now) {
    requestAnimationFrame(frame);
    if ((!NATIVE && !ctx) || !manifest || !canRun()) return;
    if (NATIVE && !geometryDirty && !dirty && now - lastCheck < CHECK_INTERVAL) return;
    const loadingStatic = REDUCED && cells.some(c => c.wanted && c.video &&
      c.status === 'loading-static' && c.staticAt != null);
    const checkDue = now - lastCheck >= CHECK_INTERVAL;
    if (checkDue) lastCheck = now;
    if (REDUCED && !dirty && !loadingStatic && !checkDue) return;
    const fallbackDue = !Number.isFinite(lastDraw) || now - lastDraw >= DRAW_INTERVAL - 0.5;
    if (fallbackDue) {
      if (!Number.isFinite(lastDraw)) lastDraw = now;
      else lastDraw += Math.max(1, Math.floor((now - lastDraw + 0.5) / DRAW_INTERVAL)) * DRAW_INTERVAL;
    }
    const paintRequested = dirty;
    const cx = worldW / 2, cy = worldH / 2;
    const x0 = cx - viewW / 2, y0 = cy - viewH / 2;
    const scale = chh / viewH;


    if (testMode && hint) hint.textContent = `a=${(cw / Math.max(chh, 1)).toFixed(2)}`;

    const m = VIEW_MARGIN * UNIT;
    const wanted = [];
    for (const c of cells) {
      const wx = c.col * UNIT, wy = c.row * UNIT;
      const ww = c.cols * UNIT, wh = c.rows * UNIT;
      if (wx + ww < x0 - m || wx > x0 + viewW + m || wy + wh < y0 - m || wy > y0 + viewH + m) continue;
      wanted.push({ c, sx: (wx - x0) * scale, sy: (wy - y0) * scale,
                    sw: ww * scale, sh: wh * scale });
    }
    const visibleCells = new Set(wanted.map(w => w.c));
    for (const c of cells) {
      const should = visibleCells.has(c);
      if (!should) {
        if (NATIVE && c.video && c.video.style.visibility !== 'hidden') c.video.style.visibility = 'hidden';
        if (c.wanted) stopCell(c);
        c.wanted = false;
        continue;
      }
      const entering = !c.wanted;
      c.wanted = true;
      if (!c.video) makeVideoElement(c);
      if (entering) {
        resetCycle(c, now);                       // fresh visit: full repair budget
        if (!REDUCED) {
          setStatus(c, 'idle', 'entered view');
        } else if (!c.staticReady) {
          if (c.video && c.video.error) setStatus(c, 'cooldown', 're-entering with error');
          else beginStaticLoad(c, now, 're-entering');
        }
      }
      if (entering || checkDue) checkPlayback(c, now);
    }

    if (NATIVE) {
      for (const w of wanted) positionNative(w.c, w.sx, w.sy, w.sw, w.sh, geometryDirty);
      dirty = false; geometryDirty = false;
      return;                           // native compositor owns every video frame
    }
    if (DIRECT) {
      if (geometryDirty) retainDirectFrames(false);
      for (const w of wanted) w.c.paintRect = { sx: w.sx, sy: w.sy, sw: w.sw, sh: w.sh };
    }
    const full = geometryDirty;
    const toPaint = wanted.filter(({ c }) => {
      if (full) return true;
      if (DIRECT) return directNeedsPaint(c, paintRequested, fallbackDue);
      const cache = c.frameCache, el = c.video;
      if (cache && cache.front && cache.version !== cache.paintedVersion) return true;
      if (REDUCED && paintRequested) return true;
      if (!el || el.readyState < 2 || el.seeking) return false;
      if (paintRequested && (!cache || !cache.front)) return true;
      if (!hasFrameWatch(c) && fallbackDue) {
        const frames = decodedFrames(el);
        return !cache || !cache.front || (frames !== null ? cache.qualityFrames !== frames : cache.clock !== el.currentTime);
      }
      return false;
    });
    dirty = false;
    if (!full && !toPaint.length) return;
    geometryDirty = false;
    draws++;
    // Clear only for geometry changes; unchanged cell pixels remain intact.
    if (full) {
      ctx.fillStyle = '#080a0e';
      ctx.fillRect(0, 0, cw, chh);
    }
    for (const wcell of toPaint) {
      const { c, sx, sy, sw, sh } = wcell;
      try {
        if (DIRECT) { drawCover(c, sx, sy, sw, sh); continue; }
        // soft per-cell backing fill (visible only before the first decode)
        const tone = 16 + Math.round(strHash(c.src) * 12);
        ctx.fillStyle = c.background || `rgb(${tone},${tone + 2},${tone + 5})`;
        ctx.fillRect(sx, sy, sw, sh);
        drawCover(c, sx, sy, sw, sh);
      } catch (error) {
        // One bad source must never stop the other cells from painting.
        c.drawFails = (c.drawFails || 0) + 1;
        if (c.drawFails === 1) console.warn('Hero wall draw failed for one cell:', c.src, error && error.message);
        note(c, 'draw-fail', error && error.message);
      }
    }
  }

  /* --------------------------------------------------------- scroll-shrink */

  let scrollQueued = false;
  function applyScroll() {
    scrollQueued = false;
    const r = driver.getBoundingClientRect();
    if (!('IntersectionObserver' in window)) setInView(r.bottom > 0 && r.top < window.innerHeight);
    const range = r.height - window.innerHeight;
    const p = range > 0 ? clamp(-r.top / range, 0, 1) : 0;
    const eased = easeInOut(p);
    let s = 1 - (1 - SHRINK) * eased;
    const nextSection = document.getElementById('overview');
    if (nextSection && window.innerHeight > 0) {
      const stickyTop = Math.min(0, r.bottom - window.innerHeight);
      const nextTop = nextSection.getBoundingClientRect().top;
      // clamp keeps the correction within the CSS 1..SHRINK scale range
      const requiredScale = clamp((nextTop - stickyTop) / window.innerHeight, SHRINK, 1);
      s = Math.max(s, requiredScale);
    }
    if (scaleBox) scaleBox.style.transform = `scale(${s.toFixed(4)})`;
    // chrome follows the underlying content: transparent over the dark hero,
    // light-frosted once the hero is half gone (state change only)
    if (navbar) {
      const solid = p > 0.5;
      if (solid !== navbar.classList.contains('hw-solid'))
        navbar.classList.toggle('hw-solid', solid);
    }
    // the title overlay leaves early — it must never linger at tiny scale
    if (overlay) {
      const f = 1 - clamp((p - TITLE_FADE[0]) / (TITLE_FADE[1] - TITLE_FADE[0]), 0, 1);
      overlay.style.opacity = f.toFixed(3);
      overlay.style.pointerEvents = f < 0.15 ? 'none' : '';
    }
    if (hint) hint.style.opacity = p > 0.04 ? 0 : '';
    if (testMode && hint) hint.textContent = `p=${p.toFixed(2)} s=${s.toFixed(2)}`;
  }
  function onScroll() {
    if (!scrollQueued) { scrollQueued = true; requestAnimationFrame(applyScroll); }
  }

  /* --------------------------------------------------------------- navbar
     Toggle handled inside applyScroll (already rAF-throttled): viewport
     progress is the correct signal for a sticky 160vh driver — an
     IntersectionObserver ratio on the driver cannot see the difference
     between "hero filling the screen" and "shrunken wall + light content". */

  /* -------------------------------------------------------------- test hook
     ?hw_shot=<scrollP>,<camSeconds> — deterministic screenshot states.
     (The camera seconds are ignored since the zoom intro was removed; the
     parameter stays so existing screenshot URLs keep working.)           */

  let testMode = false;
  function testHook() {
    const m = location.search.match(/hw_shot=([\d.]+),?([\d.]*)/);
    if (!m) return;
    testMode = true;
    const p = parseFloat(m[1]);
    window.addEventListener('load', () => {
      const r = driver.getBoundingClientRect();
      const range = r.height - window.innerHeight;
      window.scrollTo({ top: Math.max(0, -r.top) + p * range, behavior: 'instant' });
      applyScroll();
    });
    window.addEventListener('error', e => {
      const d = document.createElement('div');
      d.id = 'hw-err'; d.textContent = 'HWERR: ' + e.message;
      d.style.cssText = 'position:fixed;top:0;left:0;z-index:99;background:red;color:#fff;font:14px monospace;padding:4px';
      document.body.appendChild(d);
    });
  }

  /* ------------------------------------------------------------------ boot */

  function boot(m) {
    manifest = m;
    worldW = m.grid_w * UNIT;
    worldH = m.grid_h * UNIT;
    cells = m.cells.map(c => Object.assign({}, c));
    resize();
    applyScroll();
    const r = driver.getBoundingClientRect();
    inView = r.bottom > 0 && r.top < window.innerHeight;
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(es => es.forEach(e => {
        setInView(e.isIntersecting);
      }), { threshold: 0 }).observe(driver);
    }
    requestAnimationFrame(frame);
  }

  fetch(MOBILE() ? './static/videos/wall/full-native-20261020-v1/mobile.json' : './static/videos/wall/full-native-20261020-v1/desktop.json')
    .then(r => { if (!r.ok) throw new Error('manifest ' + r.status); return r.json(); })
    .then(boot)
    .catch(() => {                                   // graceful: page lives on
      driver.style.display = 'none';
      if (navbar) navbar.classList.add('hw-solid');  // nav must come back
    });

  if (DEBUG) {
    // Opt-in diagnostics (?hw_debug): state snapshots plus a bounded ring
    // buffer of machine transitions. The live element reference is part of
    // this debug surface only — production pages never see it.
    window.__heroWallDebug = () => ({
      reducedMotion: !!REDUCED, inView, hidden: document.hidden, pageActive,
      renderer: NATIVE ? 'native-video' : DIRECT ? 'direct-rvfc' : 'cached-raf', draws, canvas: { width: cv.width, height: cv.height },
      ring: ring ? ring.slice(-RING_MAX) : null,
      cells: cells.map(c => ({
        src: c.src, wanted: !!c.wanted, active: !!c.active, pending: !!c.pending,
        status: c.status || 'unloaded', attempts: c.attempts || 0,
        consecutive: c.consecutive || 0, level: c.level || 0,
        signal: c.signal || null, error: c.lastError || null,
        draws: c.draws || 0, drawFails: c.drawFails || 0, gen: c.gen || 0,
        capturedFrames: NATIVE ? (c.nativeFrames || 0) : DIRECT ? (c.directFrame ? c.directFrame.version : 0) : (c.frameCache ? c.frameCache.version : 0),
        nativePresentedFrames: c.nativePresentedFrames,
        cachedMediaTime: DIRECT ? (c.directFrame ? c.directFrame.mediaTime : null) : (c.frameCache ? c.frameCache.mediaTime : null),
        cachedGeneration: c.frameCache ? c.frameCache.ownerGen : null,
        cacheFailures: c.frameCache ? c.frameCache.failures : 0,
        cacheError: DIRECT ? (c.directFrame ? c.directFrame.error : null) : (c.frameCache ? c.frameCache.error : null),
        presented: c.presented || 0, presentedTime: c.presentedTime != null ? c.presentedTime : null,
        currentTime: c.video ? c.video.currentTime : null,
        paused: c.video ? c.video.paused : true,
        seeking: c.video ? !!c.video.seeking : false,
        readyState: c.video ? c.video.readyState : 0,
        frames: c.video ? decodedFrames(c.video) : null
      }))
    });
  }

  testHook();
  document.addEventListener('visibilitychange', refreshVisibility);
  window.addEventListener('pagehide', () => { pageActive = false; refreshVisibility(); });
  window.addEventListener('pageshow', () => { pageActive = true; refreshVisibility(); });
  window.addEventListener('resize', resize);
  window.addEventListener('scroll', onScroll, { passive: true });
})();
