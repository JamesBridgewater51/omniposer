/* ============================================================================
   OmniPoser project website — interaction & data-viz engine
   Human motion uses rendered SMPL / SMPL-X media, never procedural anatomy.
   This file handles static radar data, result bars, media playback, and the
   paired-frame comparison. Static hosting friendly, zero dependencies.
   ========================================================================== */
/* global RADAR_DATA, BARS_DATA, MOE_ROUTING */

(function () {
  'use strict';

  /* ---------------------------------------------------------------- utils */

  const $ = (s, c) => (c || document).querySelector(s);
  const $$ = (s, c) => Array.from((c || document).querySelectorAll(s));
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const easeInOut = t => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const TAU = Math.PI * 2;
  const REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function fitCanvas(cv, w, h) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = Math.max(1, Math.round(w * dpr));
    cv.height = Math.max(1, Math.round(h * dpr));
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  function onVisible(el, fn) {
    if (!('IntersectionObserver' in window)) { fn(true); return; }
    const io = new IntersectionObserver(es => es.forEach(e => fn(e.isIntersecting)), { threshold: 0.05 });
    io.observe(el);
  }

  // pause CSS animations on subtrees that scroll out of view
  function pauseOffscreen(sel) {
    $$(sel).forEach(el => onVisible(el, vis => el.classList.toggle('paused', !vis)));
  }

  function mediaVisibility() {
    $$('.qa-poster video').forEach(video => {
      let visible = false, ticket = 0;
      video.autoplay = false;
      video.pause();
      if (REDUCED) video.controls = true;
      function sync() {
        const current = ++ticket;
        if (!visible || document.hidden || REDUCED) {
          video.pause();
          return;
        }
        video.play().then(() => {
          if (current !== ticket && (!visible || document.hidden)) video.pause();
        }).catch(() => {
          if (current !== ticket) return;
          video.controls = true;
          video.dataset.playback = 'manual';
        });
      }
      onVisible(video, value => { visible = value; sync(); });
      video.addEventListener('loadeddata', sync);
      document.addEventListener('visibilitychange', sync);
    });
  }

  const C = {
    cyan: '#1899ce', cyan2: '#0e7490', violet: '#7c5cbf', violet2: '#6d4fb3',
    ink: '#363636', ink2: '#555555', ink3: '#8a8f98',
  };

  /* ============================================================================
     1. Animated counters
     ========================================================================== */

  function counters() {
    $$('.stat-num[data-count]').forEach(el => {
      const target = parseFloat(el.dataset.count);
      let done = false;
      onVisible(el, vis => {
        if (!vis || done) return;
        done = true;
        if (REDUCED) { el.firstChild.textContent = String(target); el.classList.remove('counting'); return; }
        const t0 = performance.now(), dur = 1600;
        (function tick(now) {
          const t = clamp((now - t0) / dur, 0, 1);
          el.firstChild.textContent = Math.round(target * (1 - Math.pow(1 - t, 3))).toLocaleString('en-US');
          if (t < 1) requestAnimationFrame(tick);
          else el.classList.remove('counting');
        })(t0);
      });
    });
  }

  /* ============================================================================
     3. Radar chart (real data from radar_data.json)
     ========================================================================== */

  function radar() {
    const cv = $('#radar-canvas');
    if (!cv || typeof RADAR_DATA === 'undefined') return;
    const axes = RADAR_DATA.axes;
    const wrap = cv.closest('.chart-wrap');
    const tip = $('#radar-tip');
    const N = axes.length;

    const COLORS = {
      'RoHM-U': '#64748b', 'RoHM++-U': '#dc2626',
      'Ours-U (Dense)': '#b45309', 'Ours-U (MoE)': '#1899ce', 'Ours-U (MoE*)': '#6d4fb3',
    };
    const methods = Object.keys(COLORS);
    const active = new Set(methods);
    let hoverAxis = -1;
    let W = 0, H = 0, ctx = null;

    function resize() {
      const r = wrap.getBoundingClientRect();
      W = Math.max(600, r.width); H = Math.min(520, Math.max(340, W * 0.82));
      ctx = fitCanvas(cv, W, H);
      cv.style.width = W + 'px';
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const cx = W / 2, cy = H / 2 + 6;
      const R = Math.min(W, H) / 2 - 86;

      const ang = i => -Math.PI / 2 + (i / N) * TAU;
      const pt = (i, r) => [cx + Math.cos(ang(i)) * r, cy + Math.sin(ang(i)) * r];
      // Normalized scores use a linear radius: 100 maps to the outer ring.
      const rv = v => R * v;

      for (let ring = 1; ring <= 4; ring++) {
        ctx.beginPath();
        for (let i = 0; i <= N; i++) {
          const [x, y] = pt(i % N, R * ring / 4);
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.strokeStyle = ring === 4 ? 'rgba(70,90,130,0.35)' : 'rgba(70,90,130,0.12)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      ctx.font = '600 12px "Noto Sans", sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (let i = 0; i < N; i++) {
        const [x, y] = pt(i, R);
        ctx.strokeStyle = 'rgba(70,90,130,0.10)';
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
        const [lx, ly] = pt(i, R + 30);
        const parts = (axes[i].benchmark + ' · ' + axes[i].metric).split(' · ');
        ctx.fillStyle = hoverAxis === i ? C.cyan2 : C.ink2;
        ctx.fillText(parts[0], lx, ly - 6);
        ctx.fillStyle = hoverAxis === i ? C.cyan : C.ink2;
        ctx.font = '400 11px "Noto Sans", sans-serif';
        const words = parts[1].split(' ');
        const line1 = words.slice(0, Math.ceil(words.length / 2)).join(' ');
        const line2 = words.slice(Math.ceil(words.length / 2)).join(' ');
        ctx.fillText(line1, lx, ly + 7);
        if (line2) ctx.fillText(line2, lx, ly + 18);
        ctx.font = '600 12px "Noto Sans", sans-serif';
      }

      // specialist scatter dots (gray) on each axis
      axes.forEach((a, i) => {
        Object.keys(a.scores).forEach(m => {
          if (!methods.includes(m)) {
            const v = clamp(a.scores[m] / 100, 0, 1);
            const [x, y] = pt(i, rv(v));
            ctx.beginPath(); ctx.arc(x, y, 1.6, 0, TAU);
            ctx.fillStyle = 'rgba(100,112,135,0.55)'; ctx.fill();
          }
        });
      });

      // universal-model polygons
      methods.forEach(m => {
        if (!active.has(m)) return;
        ctx.beginPath();
        for (let i = 0; i <= N; i++) {
          const v = clamp((axes[i % N].scores[m] || 0) / 100, 0, 1);
          const [x, y] = pt(i % N, rv(v));
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        const col = COLORS[m];
        const isStar = m === 'Ours-U (MoE*)';
        ctx.fillStyle = col + (isStar ? '2e' : '1a');
        ctx.fill();
        ctx.strokeStyle = col;
        ctx.lineWidth = isStar ? 2.2 : 1.6;
        // distinct stroke styles so overlapping series stay separable
        const DASH = { 'RoHM-U': [4, 4], 'RoHM++-U': [8, 4], 'Ours-U (Dense)': [1.5, 3] };
        if (DASH[m]) ctx.setLineDash(DASH[m]);
        ctx.stroke();
        ctx.setLineDash([]);
        for (let i = 0; i < N; i++) {
          const v = clamp((axes[i].scores[m] || 0) / 100, 0, 1);
          const [x, y] = pt(i, rv(v));
          ctx.beginPath(); ctx.arc(x, y, isStar ? 3 : 2.2, 0, TAU);
          ctx.fillStyle = col; ctx.fill();
        }
      });
    }

    cv.addEventListener('mousemove', e => {
      const r = cv.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      const cx = W / 2, cy = H / 2 + 6, R = Math.min(W, H) / 2 - 86;
      let best = -1, bd = 1e9;
      for (let i = 0; i < N; i++) {
        const a = -Math.PI / 2 + (i / N) * TAU;
        const d = Math.hypot(cx + Math.cos(a) * R - mx, cy + Math.sin(a) * R - my);
        if (d < bd) { bd = d; best = i; }
      }
      hoverAxis = bd < 60 ? best : -1;
      if (hoverAxis >= 0 && tip) {
        const a = axes[hoverAxis];
        tip.innerHTML = `<b style="color:${C.cyan2}">${a.benchmark}</b> · ${a.metric}<br>` +
          methods.filter(m => active.has(m)).map(m =>
            `<span style="color:${COLORS[m]}">${m}</span> ${a.raw[m] != null ? a.raw[m] : '—'}`).join('<br>');
        tip.style.left = clamp(mx + 14, 0, W - 170) + 'px';
        tip.style.top = clamp(my + 10, 0, H - 90) + 'px';
        tip.style.opacity = 1;
      } else if (tip) tip.style.opacity = 0;
      draw();
    });
    cv.addEventListener('mouseleave', () => { hoverAxis = -1; if (tip) tip.style.opacity = 0; draw(); });

    $$('#radar-legend .lg').forEach(lg => {
      lg.addEventListener('click', () => {
        const m = lg.dataset.method;
        if (active.has(m)) { if (active.size > 1) active.delete(m); }
        else active.add(m);
        lg.classList.toggle('off', !active.has(m));
        draw();
      });
    });
    resize();
    window.addEventListener('resize', () => { resize(); draw(); });
    draw();
  }

  /* ============================================================================
     4. Results bars (animated, staged Dense → MoE → MoE*)
     ========================================================================== */

  function bars() {
    if (typeof BARS_DATA === 'undefined') return;
    const sel = $('#bars-setting');
    const host = $('#bars-host');
    if (!host) return;
    const groups = BARS_DATA.groups;

    if (sel && !sel.options.length) {
      groups.forEach((g, i) => {
        const o = document.createElement('option');
        o.value = String(i);
        o.textContent = g.label;
        sel.appendChild(o);
      });
    }

    function render() {
      const g = groups[sel ? parseInt(sel.value, 10) || 0 : 0];
      const rows = g.rows;
      const max = Math.max(...rows.map(r => r.v)) * 1.06;
      host.innerHTML = rows.map(r => `
        <div class="bar-row">
          <div class="bar-name ${r.ours ? 'ours' : ''}">${r.name}</div>
          <div class="bar-track"><div class="bar-fill ${r.ours ? (r.best ? 'best' : 'ours') : ''}"
               style="width:${(r.v / max * 100).toFixed(1)}%;transition:none"></div></div>
          <div class="bar-val ${r.ours ? 'ours' : ''}">${r.v.toFixed(1)}<span style="color:var(--ink-3)"> ${g.unit}</span></div>
        </div>`).join('');
      const note = $('#bars-note');
      if (note) note.textContent = g.note;
    }

    if (sel) sel.addEventListener('change', render);
    render();
  }

  /* ============================================================================
     5. MoE routing heatmap (animated reveal)
     ========================================================================== */

  function heatmap() {
    const svg = $('#moe-heat');
    if (!svg || typeof MOE_ROUTING === 'undefined') return;
    const NS = 'http://www.w3.org/2000/svg';
    const pats = MOE_ROUTING.patterns, experts = MOE_ROUTING.experts;
    const rows = MOE_ROUTING.rows;
    const x0 = 96, y0 = 26, cw = 36, ch = 30;

    function el(tag, attrs) {
      const e = document.createElementNS(NS, tag);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      return e;
    }

    const xt = el('text', { class: 'heat-axis-x', x: x0 - 10, y: y0 - 8, 'text-anchor': 'end' });
    xt.textContent = 'expert →';
    svg.appendChild(xt);
    experts.forEach((ex, i) => {
      const t = el('text', { class: 'heat-axis-x', x: x0 + i * cw + cw / 2, y: y0 - 8, 'text-anchor': 'middle' });
      t.textContent = ex;
      svg.appendChild(t);
    });
    pats.forEach((p, i) => {
      const t = el('text', { class: 'heat-axis-y', x: x0 - 10, y: y0 + i * ch + ch / 2 + 3, 'text-anchor': 'end' });
      t.textContent = p;
      svg.appendChild(t);
    });

    const cells = [];
    rows.forEach((row, ri) => {
      row.w.forEach((w, ci) => {
        const alpha = clamp(w / 25, 0, 1);
        const rect = el('rect', {
          class: 'heat-cell', x: x0 + ci * cw + 2, y: y0 + ri * ch + 2,
          width: cw - 4, height: ch - 4, rx: 6,
          fill: w > 0 ? `rgba(24,153,206,${0.08 + alpha * 0.7})` : 'rgba(70,90,130,0.06)',
          stroke: w >= 25 ? 'rgba(14,116,144,0.85)' : 'transparent',
          'stroke-width': 1.2,
        });
        svg.appendChild(rect);
        cells.push({ rect, ri, ci });
        if (w > 0 && row.nums !== false) {
          const t = el('text', {
            x: x0 + ci * cw + cw / 2, y: y0 + ri * ch + ch / 2 + 3.5,
            'text-anchor': 'middle', fill: alpha > 0.5 ? '#0b3a4d' : C.ink2,
            style: 'font:600 11px "JetBrains Mono",monospace; pointer-events:none',
          });
          t.textContent = w.toFixed(0);
          svg.appendChild(t);
        }
      });
    });

    if (REDUCED) { cells.forEach(c => c.rect.classList.add('on')); return; }
    let started = false;
    onVisible(svg, vis => {
      if (!vis || started) return;
      started = true;
      const per = 260;
      cells.forEach(c => {
        setTimeout(() => c.rect.classList.add('on'), 300 + c.ri * per + c.ci * 40);
      });
    });
  }

  /* ============================================================================
     6. Comparison slider
     ========================================================================== */

  function comparators() {
    const initialized = comparators._guards || (comparators._guards = new WeakSet());
    $$('.cmp').forEach(cmp => {
      if (initialized.has(cmp)) return;
      const initial = [$('.cmp-a', cmp), $('.cmp-b', cmp)];
      const output = $('.cmp-output', cmp), handle = $('.cmp-handle', cmp);
      const controls = $(`[data-comparator="${cmp.id}"]`);
      if (!initial.every(v => v && v.tagName === 'VIDEO') || !output || !controls) return;
      initialized.add(cmp);
      const wipe = $('.cmp-wipe', controls), timeline = $('.cmp-time', controls);
      const play = $('.cmp-play', controls), status = $('.cmp-status', controls);
      const selector = $('.cmp-clip', controls);
      const descriptionEl = $('.cmp-description', cmp.parentElement);
      const ctx = output.getContext('2d');
      const originalIds = [initial[0].id, initial[1].id];
      let videos = initial;
      let pairGeneration = 0;
      let frameWatchGeneration = 0;
      const queues = [[], []], pools = [[], []], front = [null, null];
      const callbacks = [null, null];
      let fraction = 0.5, visible = false, wanted = !REDUCED, failed = false;
      let phase = 'loading', epoch = 0, attempts = 0, watchdog = null;
      let phaseSince = performance.now(), lastCommit = phaseSince, target = null;
      let pointer = null, commits = 0, loops = 0, pairSeeking = false;
      let currentPair = null;
      let disposers = [];

      const catalog = [];
      if (Array.isArray(window.COMPARISON_CLIPS)) {
        window.COMPARISON_CLIPS.forEach(e => {
          if (e && e.id && e.pred && e.gt) catalog.push(e);
        });
      }
      if (!catalog.length) {
        const legacyStep = Number(timeline.step);
        catalog.push({
          id: 'legacy',
          label: 'Comparison clips',
          dataset: 'Comparisons',
          condition: '',
          description: '',
          pred: cmp.dataset.pred || '',
          gt: cmp.dataset.gt || '',
          fps: legacyStep > 0 ? 1 / legacyStep : 30
        });
      }

      if (selector) {
        selector.textContent = '';
        const groups = new Map();
        catalog.forEach(e => {
          const key = e.dataset || 'Comparisons';
          if (!groups.has(key)) groups.set(key, []);
          groups.get(key).push(e);
        });
        groups.forEach((entries, name) => {
          const group = document.createElement('optgroup');
          group.label = name;
          entries.forEach(e => {
            const opt = document.createElement('option');
            opt.value = e.id;
            opt.textContent = e.label || e.id;
            group.appendChild(opt);
          });
          selector.appendChild(group);
        });
        selector.value = catalog[0].id;
        selector.disabled = catalog.length < 2;
        selector.addEventListener('change', () => {
          const entry = catalog.find(e => e.id === selector.value);
          if (entry) setPair(entry);
        });
      }

      function state(next, message) {
        if (phase !== next) phaseSince = performance.now();
        phase = next;
        cmp.dataset.state = next;
        status.textContent = message;
        play.textContent = failed ? 'Retry' : wanted ? 'Pause' : 'Play';
        play.setAttribute('aria-label', failed ? 'Retry comparison playback' : wanted ? 'Pause comparison playback' : 'Play comparison playback');
      }
      function pauseBoth() {
        epoch++;
        videos.forEach(v => v.pause());
      }
      function clearQueues() {
        queues.forEach((q, i) => { while (q.length) pools[i].push(q.shift().canvas); });
      }
      function supportsFrameAPI() {
        return videos.length === 2 && videos.every(v => typeof v.requestVideoFrameCallback === 'function');
      }
      function paint() {
        if (!front.every(Boolean)) return;
        const width = output.width, height = output.height;
        ctx.drawImage(front[0], 0, 0, width, height);
        ctx.save();
        ctx.beginPath(); ctx.rect(width * fraction, 0, width * (1 - fraction), height); ctx.clip();
        ctx.drawImage(front[1], 0, 0, width, height);
        ctx.restore();
      }
      function setWipe(value) {
        fraction = clamp(value, 0, 1);
        wipe.value = String(Math.round(fraction * 100));
        wipe.setAttribute('aria-valuetext', `${wipe.value}% prediction, ${100 - Number(wipe.value)}% ground truth`);
        handle.style.left = fraction * 100 + '%';
        paint();
      }
      function capture(i, time, pairGen, owner) {
        if (pairGen !== pairGeneration || owner !== videos) return;
        if (target !== null && Math.abs(time - target) > 0.04) return;
        const v = videos[i], q = queues[i];
        if (q.length === 3) pools[i].push(q.shift().canvas);
        const canvas = pools[i].pop() || document.createElement('canvas');
        if (canvas.width !== v.videoWidth || canvas.height !== v.videoHeight) {
          canvas.width = v.videoWidth; canvas.height = v.videoHeight;
        }
        canvas.getContext('2d').drawImage(v, 0, 0);
        q.push({ time, canvas, gen: pairGen, slot: i });
        const other = queues[1 - i].find(f => f.gen === pairGen && Math.abs(f.time - time) < 0.001);
        if (!other) return;
        if (pairGen !== pairGeneration || owner !== videos) return;
        const pair = i === 0 ? [q[q.length - 1], other] : [other, q[q.length - 1]];
        pair.forEach((f, k) => {
          if (front[k]) pools[k].push(front[k]);
          front[k] = f.canvas;
          queues[k].splice(queues[k].indexOf(f), 1);
        });
        cmp.dataset.frameA = String(pair[0].time);
        cmp.dataset.frameB = String(pair[1].time);
        cmp.dataset.commits = String(++commits);
        cmp.dataset.commitPair = currentPair.id;
        lastCommit = performance.now();
        timeline.value = String(pair[0].time);
        timeline.setAttribute('aria-valuetext', `${pair[0].time.toFixed(2)} of ${Number(timeline.max).toFixed(2)} seconds`);
        paint();
      }
      function armFrames() {
        if (!supportsFrameAPI() || !visible || document.hidden || failed) return;
        videos.forEach((v, i) => {
          if (callbacks[i] !== null) return;
          const pairGen = pairGeneration, watchGen = frameWatchGeneration, owner = videos, slot = i;
          const id = v.requestVideoFrameCallback((now, meta) => {
            if (pairGen !== pairGeneration || owner !== videos || watchGen !== frameWatchGeneration || callbacks[slot] !== id) return;
            callbacks[slot] = null;
            capture(slot, meta.mediaTime, pairGen, owner);
            armFrames();
          });
          callbacks[i] = id;
        });
      }
      function cancelFrames() {
        frameWatchGeneration++;
        videos.forEach((v, i) => {
          if (callbacks[i] !== null) {
            try { v.cancelVideoFrameCallback(callbacks[i]); } catch (_) {}
          }
          callbacks[i] = null;
        });
      }
      function fail(message) {
        failed = true; pauseBoth(); cancelFrames(); clearInterval(watchdog); watchdog = null;
        state('error', message);
      }
      function seekPair(time) {
        if (!videos.every(v => v.readyState >= 1)) return;
        pauseBoth(); cancelFrames(); clearQueues();
        target = clamp(time, 0, Math.max(0, Math.min(...videos.map(v => v.duration)) - (Number(timeline.step) || 1 / 30)));
        cmp.dataset.seekTarget = String(target);
        pairSeeking = true;
        state('seeking', 'Aligning comparison…');
        phaseSince = performance.now();
        armFrames();
        videos.forEach(v => { v.currentTime = target; });
      }
      function reconcile() {
        if (failed) return;
        if (!visible || document.hidden) {
          // Temporary suspension only: mark the paused phase before pausing so the
          // videos' own pause events are not mistaken for a reader pause, and
          // playback resumes automatically when the comparison returns to view.
          state('paused', 'Paused while out of view');
          pauseBoth();
          return;
        }
        if (!wanted) {
          if (!front.every(Boolean) && !pairSeeking && videos.every(v => v.readyState >= 2)) {
            seekPair(videos[0].currentTime + (Number(timeline.step) || 1 / 30));
          }
          state('paused', 'Paused');
          pauseBoth();
          return;
        }
        if (!videos.every(v => v.readyState >= 3 && !v.seeking)) {
          if (phase !== 'seeking') state('loading', 'Loading both comparison clips…');
          return;
        }
        if (!supportsFrameAPI()) {
          wanted = false;
          state('paused', 'Synchronized playback is unavailable in this browser.');
          play.disabled = true;
          return;
        }
        if (phase === 'playing' || phase === 'starting') return;
        if (phase === 'buffering') { seekPair(videos[0].currentTime); return; }
        target = null;
        const ticket = ++epoch;
        state('starting', 'Starting comparison…');
        armFrames();
        const owner = videos;
        const playPairGen = pairGeneration;
        Promise.all(owner.map(v => v.play())).then(() => {
          if (playPairGen !== pairGeneration || owner !== videos) {
            owner.forEach(v => { try { v.pause(); } catch (_) {} });
            return;
          }
          const healthy = wanted && visible && !document.hidden && (phase === 'starting' || phase === 'playing');
          if (ticket !== epoch) {
            if (!healthy) owner.forEach(v => { try { v.pause(); } catch (_) {} });
            return;
          }
          lastCommit = performance.now();
          state('playing', 'Playing · matched camera frames');
        }).catch(() => {
          if (playPairGen !== pairGeneration || owner !== videos) {
            owner.forEach(v => { try { v.pause(); } catch (_) {} });
            return;
          }
          if (ticket !== epoch) return;
          pauseBoth(); wanted = false;
          state('paused', 'Playback was blocked. Press Play to retry.');
        });
      }
      function recover() {
        if (++attempts > 3) { fail('Comparison stalled. Press Retry to reload both clips.'); return; }
        cmp.dataset.recoveries = String(attempts);
        if (!videos.every(v => v.readyState >= 1)) {
          pauseBoth(); clearQueues();
          state('loading', 'Reloading both comparison clips…');
          phaseSince = performance.now();
          videos.forEach(v => v.load());
          return;
        }
        seekPair(videos[0].ended ? 0 : videos[0].currentTime);
      }
      function lifecycle() {
        if (!visible || document.hidden) {
          cancelFrames(); clearInterval(watchdog); watchdog = null;
        } else if (!failed) {
          armFrames();
          if (!watchdog) watchdog = setInterval(() => {
            if (!wanted || failed) return;
            const now = performance.now();
            if (phase === 'playing' && now - lastCommit > 1500) recover();
            else if (['loading', 'buffering', 'seeking', 'starting'].includes(phase) && now - phaseSince > 8000) recover();
          }, 250);
        }
        reconcile();
      }
      function attachPairListeners(gen, owner) {
        owner.forEach((v, i) => {
          const add = (event, fn) => {
            const handler = (...args) => {
              if (gen !== pairGeneration || owner !== videos) return;
              fn(...args);
            };
            v.addEventListener(event, handler);
            disposers.push(() => v.removeEventListener(event, handler));
          };
          add('loadedmetadata', () => {
            if (!owner.every(s => s.readyState >= 1)) return;
            if (owner.some(s => s.videoWidth <= 0 || s.videoHeight <= 0 || !isFinite(s.duration) || s.duration <= 0)) {
              fail('Comparison clips have invalid dimensions or durations.'); return;
            }
            if (owner[0].videoWidth !== owner[1].videoWidth || owner[0].videoHeight !== owner[1].videoHeight ||
                Math.abs(owner[0].duration - owner[1].duration) > 0.001) {
              fail('Comparison clips have incompatible dimensions or durations.'); return;
            }
            if (currentPair.width && (owner[0].videoWidth !== currentPair.width || owner[0].videoHeight !== currentPair.height)) {
              fail('Comparison clips have incompatible dimensions or durations.'); return;
            }
            const fps = currentPair.fps > 0 ? currentPair.fps : 30;
            const step = 1 / fps;
            output.width = owner[0].videoWidth; output.height = owner[0].videoHeight;
            cmp.style.setProperty('--cmp-aspect', String(owner[0].videoWidth / owner[0].videoHeight));
            timeline.max = String(Math.max(0, Math.min(...owner.map(s => s.duration)) - step));
            timeline.step = String(step);
            timeline.disabled = false;
            play.disabled = false;
            reconcile();
          });
          ['loadeddata', 'canplay', 'seeked'].forEach(event => add(event, () => {
            if (!supportsFrameAPI() && owner.every(s => s.readyState >= 2)) {
              capture(0, owner[0].currentTime, gen, owner);
              capture(1, owner[1].currentTime, gen, owner);
            }
            reconcile();
          }));
          add('waiting', () => {
            if (phase !== 'playing' && phase !== 'starting') return;
            pauseBoth(); state('buffering', 'Buffering both clips…');
          });
          add('seeking', () => {
            if (!pairSeeking && v.readyState >= 1) seekPair(v.currentTime);
          });
          add('seeked', () => {
            if (owner.every(s => !s.seeking)) pairSeeking = false;
          });
          add('pause', () => {
            if (phase !== 'playing' || v.ended) return;
            if (!visible || document.hidden) return; // suspension, not a reader pause
            wanted = false; pauseBoth(); state('paused', 'Paused');
          });
          add('ended', () => {
            if (phase !== 'playing' && phase !== 'starting') return;
            cmp.dataset.loops = String(++loops); seekPair(0);
          });
          add('error', () => fail('Comparison could not load. Press Retry to reload both clips.'));
        });
      }
      function setPair(entry) {
        pairGeneration++;
        cancelFrames();
        clearInterval(watchdog); watchdog = null;
        clearQueues();
        front[0] = null; front[1] = null;
        disposers.forEach(d => d());
        disposers = [];
        videos.forEach(v => {
          try { v.pause(); } catch (_) {}
          try { v.removeAttribute('src'); v.load(); } catch (_) {}
          v.remove();
        });
        videos = [];
        ctx.clearRect(0, 0, output.width, output.height);
        delete cmp.dataset.frameA;
        delete cmp.dataset.frameB;
        delete cmp.dataset.commitPair;
        delete cmp.dataset.recoveries;
        delete cmp.dataset.seekTarget;
        delete cmp.dataset.loops;
        currentPair = entry;
        cmp.dataset.pair = entry.id;
        cmp.dataset.pairGeneration = String(pairGeneration);
        cmp.dataset.pred = entry.pred;
        cmp.dataset.gt = entry.gt;
        if (descriptionEl) descriptionEl.textContent = entry.description || '';
        attempts = 0; loops = 0; target = null; pairSeeking = false; failed = false;
        epoch++;
        play.disabled = false;
        timeline.value = '0'; timeline.max = '0'; timeline.disabled = true;
        const fps = entry.fps > 0 ? entry.fps : 30;
        timeline.step = String(1 / fps);
        if (entry.width > 0 && entry.height > 0) {
          cmp.style.setProperty('--cmp-aspect', String(entry.width / entry.height));
        } else {
          cmp.style.removeProperty('--cmp-aspect');
        }
        state('loading', 'Loading comparison…');
        phaseSince = performance.now();
        lastCommit = performance.now();
        const pair = [];
        for (let i = 0; i < 2; i++) {
          const v = document.createElement('video');
          if (originalIds[i]) v.id = originalIds[i];
          v.className = i === 0 ? 'cmp-a' : 'cmp-b';
          v.muted = true;
          v.playsInline = true;
          v.setAttribute('aria-hidden', 'true');
          v.preload = 'auto';
          v.loop = false;
          v.autoplay = false;
          const poster = i === 0 ? entry.posterPred : entry.posterGt;
          v.poster = poster || (entry.id === 'legacy'
            ? (i === 0 ? entry.pred : entry.gt).replace(/\.mp4$/, '_poster.jpg')
            : '');
          v.src = i === 0 ? entry.pred : entry.gt;
          pair.push(v);
        }
        videos = pair;
        attachPairListeners(pairGeneration, pair);
        pair.forEach(v => cmp.appendChild(v));
        setWipe(fraction);
        lifecycle();
      }
      wipe.addEventListener('input', () => setWipe(Number(wipe.value) / 100));
      timeline.addEventListener('input', () => { seekPair(Number(timeline.value)); reconcile(); });
      play.addEventListener('click', () => {
        if (failed) {
          failed = false; attempts = 0; target = null; wanted = true;
          clearQueues(); state('loading', 'Reloading comparison…');
          videos.forEach(v => v.load()); lifecycle(); return;
        }
        wanted = !wanted;
        if (wanted && videos.every(v => v.readyState >= 1)) seekPair(videos[0].currentTime);
        reconcile();
      });
      function move(e) {
        const r = cmp.getBoundingClientRect();
        if (r.width) setWipe((e.clientX - r.left) / r.width);
      }
      cmp.addEventListener('pointerdown', e => {
        if (!e.isPrimary || e.button !== 0) return;
        pointer = e.pointerId; cmp.setPointerCapture(pointer); move(e);
      });
      cmp.addEventListener('pointermove', e => { if (e.pointerId === pointer) move(e); });
      const release = e => { if (e.pointerId === pointer) pointer = null; };
      ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(event => cmp.addEventListener(event, release));
      document.addEventListener('visibilitychange', lifecycle);
      onVisible(cmp, inView => { visible = inView; lifecycle(); });
      setWipe(0.5);
      setPair(catalog[0]);
    });
  }
  /* ============================================================================
     7. Reveal-on-scroll, nav, bibtex copy
     ========================================================================== */

  function reveals() {
    const els = $$('.rv');
    if (!('IntersectionObserver' in window) || REDUCED) { els.forEach(e => e.classList.add('in')); return; }
    const io = new IntersectionObserver(es => es.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
    }), { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
    els.forEach(e => io.observe(e));
  }

  function navbar() {
    const burger = $('.navbar-burger'), menu = $('.navbar-menu');
    if (!burger || !menu) return;
    const mq = window.matchMedia('(min-width: 1024px)');
    const isDesktop = () => mq.matches;
    const isExpanded = () => burger.classList.contains('is-active');
    function sync() {
      const desktop = isDesktop(), expanded = isExpanded();
      burger.classList.toggle('is-active', desktop ? false : expanded);
      menu.classList.toggle('is-active', desktop ? false : expanded);
      burger.setAttribute('aria-expanded', String(desktop ? true : expanded));
      menu.inert = !desktop && !expanded;
      if (menu.inert && menu.contains(document.activeElement)) burger.focus();
    }
    burger.addEventListener('click', () => {
      burger.classList.toggle('is-active');
      sync();
    });
        document.addEventListener('keydown', (e) => {
          if (e.key === 'Escape' && !isDesktop() && isExpanded()) {
        burger.classList.remove('is-active');
        burger.setAttribute('aria-expanded', 'false');
        menu.classList.remove('is-active');
        menu.inert = true;
        burger.focus();
      }
    });
    menu.addEventListener('click', (e) => {
      if (e.target.closest('.navbar-item') && !isDesktop()) {
        burger.classList.remove('is-active');
        sync();
      }
    });
    let desktop = isDesktop();
    sync();
    mq.addEventListener('change', () => {
      const nowDesktop = isDesktop();
      if (nowDesktop !== desktop) {
        desktop = nowDesktop;
        if (!nowDesktop) burger.classList.remove('is-active');
        sync();
      }
    });
  }

  /* ------------------------------------------------------------------ boot */

  document.addEventListener('DOMContentLoaded', () => {
    mediaVisibility();
    bars();
    comparators();
    reveals();
    navbar();
    pauseOffscreen('.pan-four');
    pauseOffscreen('.video-soon');
    pauseOffscreen('.qa-poster .ph');
    pauseOffscreen('.flow-dots');
  });
})();
