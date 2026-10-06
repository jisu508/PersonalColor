/* =====================================================================
 *  draping.js — 실시간 디지털 드레이핑
 *  ---------------------------------------------------------------------
 *  결과 화면에서 "디지털 드레이핑" 버튼을 누르면 카메라가 켜지고,
 *  시즌 컬러 팬 배경 가운데 타원에 내 얼굴이 실시간으로 들어간다.
 *  (스노우 필터와 같은 방식 — 매 프레임 캔버스에 배경 → 얼굴 순서로 그린다)
 *
 *  백엔드를 거치지 않는다. 전부 브라우저 안에서 돈다.
 *
 *  쓰는 법
 *      openDraping('가을 웜');     // 결과의 best_group 을 그대로 넘기면 된다
 *
 *  나중에 '인물 분리(진짜 드레이핑)' 로 바꾸려면 drawPerson() 하나만 고치면 된다.
 * ===================================================================== */
(function () {
  'use strict';

  // 배경 이미지 (static/images/ 안에 있는 파일)
  const SEASONS = [
    { key: '봄',   label: 'SPRING WARM', img: '봄_웜_확인.png' },
    { key: '여름', label: 'SUMMER COOL', img: '여름_쿨_확인.png' },
    { key: '가을', label: 'AUTUMN WARM', img: '가을_웜_확인.png' },
    { key: '겨울', label: 'WINTER COOL', img: '겨울_쿨_확인.png' },
  ];

  // 배경 이미지 안에서 타원(얼굴 들어갈 구멍)이 차지하는 위치 — 0~1 비율.
  // 실제 이미지(1254×1254)를 측정해서 넣은 값이다. 이미지를 바꾸면 다시 재야 한다.
  const OVAL = { cx: 0.4996, cy: 0.5012, rx: 0.2572 * 0.955, ry: 0.3250 * 0.965 };

  let stream = null, raf = null, video = null, canvas = null, ctx = null;
  let current = 2;                       // 지금 보고 있는 시즌 (기본: 가을)
  const images = {};                     // 미리 받아둔 배경 이미지

  // ── 배경 이미지 미리 받기 ───────────────────────────────────────
  function loadImage(file) {
    return new Promise((resolve, reject) => {
      if (images[file]) return resolve(images[file]);
      const im = new Image();
      im.onload = () => { images[file] = im; resolve(im); };
      im.onerror = () => reject(new Error(file + ' 를 불러오지 못했습니다'));
      im.src = '/static/images/' + encodeURIComponent(file);
    });
  }

  // ── 화면 만들기 (한 번만) ──────────────────────────────────────
  function buildUI() {
    if (document.getElementById('draping-modal')) return;
    const wrap = document.createElement('div');
    wrap.id = 'draping-modal';
    wrap.className = 'draping-modal hidden';
    wrap.innerHTML = `
      <div class="draping-inner">
        <canvas id="draping-canvas"></canvas>
        <p class="draping-hint">타원 안에 얼굴을 맞추고, 아래 버튼으로 시즌을 바꿔보세요</p>
        <div class="draping-seasons">
          ${SEASONS.map((s, i) =>
            `<button data-i="${i}" class="draping-season">${s.key}</button>`).join('')}
        </div>
        <div class="draping-actions">
          <button id="draping-shot" class="btn-primary">사진 저장</button>
          <button id="draping-close" class="btn-secondary">닫기</button>
        </div>
      </div>`;
    document.body.appendChild(wrap);

    wrap.querySelectorAll('.draping-season').forEach(b => {
      b.addEventListener('click', () => select(Number(b.dataset.i)));
    });
    document.getElementById('draping-close').addEventListener('click', closeDraping);
    document.getElementById('draping-shot').addEventListener('click', snapshot);
    wrap.addEventListener('click', e => { if (e.target === wrap) closeDraping(); });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && !wrap.classList.contains('hidden')) closeDraping();
    });

    canvas = document.getElementById('draping-canvas');
    ctx = canvas.getContext('2d');
    video = document.createElement('video');
    video.autoplay = true; video.playsInline = true; video.muted = true;
  }

  function select(i) {
    current = i;
    document.querySelectorAll('.draping-season').forEach((b, k) =>
      b.classList.toggle('on', k === i));
    loadImage(SEASONS[i].img).catch(err => console.warn(err.message));
  }

  // ── 매 프레임 그리기 ───────────────────────────────────────────
  function draw() {
    raf = requestAnimationFrame(draw);
    const bg = images[SEASONS[current].img];
    if (!bg || !video.videoWidth) return;

    const S = canvas.width;                       // 정사각형 캔버스
    ctx.clearRect(0, 0, S, S);
    ctx.drawImage(bg, 0, 0, S, S);                // ① 배경(컬러 팬)

    const rect = {                                // ② 얼굴 들어갈 타원
      cx: OVAL.cx * S, cy: OVAL.cy * S,
      rx: OVAL.rx * S, ry: OVAL.ry * S,
    };
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(rect.cx, rect.cy, rect.rx, rect.ry, 0, 0, Math.PI * 2);
    ctx.clip();                                   // 타원 밖은 안 그려진다
    drawPerson(ctx, rect);
    ctx.restore();
  }

  /* 타원 안에 사람을 그린다.
     지금은 영상을 통째로 'cover' 로 채운다(타원 클립 방식).
     나중에 인물 분리를 붙일 때는 이 함수만 바꾸면 된다 —
     MediaPipe ImageSegmenter 로 사람 마스크를 만들어 같은 자리에 올리면 된다. */
  function drawPerson(c, rect) {
    const vw = video.videoWidth, vh = video.videoHeight;
    const bw = rect.rx * 2, bh = rect.ry * 2;
    const scale = Math.max(bw / vw, bh / vh);      // 타원을 덮도록 확대
    const dw = vw * scale, dh = vh * scale;
    const dx = rect.cx - dw / 2, dy = rect.cy - dh / 2;
    c.save();
    c.translate(dx + dw, dy);                      // 좌우 반전 (거울처럼)
    c.scale(-1, 1);
    c.drawImage(video, 0, 0, dw, dh);
    c.restore();
  }

  // ── 열기 / 닫기 / 저장 ─────────────────────────────────────────
  async function openDraping(bestGroup) {
    buildUI();
    const i = SEASONS.findIndex(s => (bestGroup || '').includes(s.key));
    select(i >= 0 ? i : 2);
    SEASONS.forEach(s => loadImage(s.img).catch(() => {}));   // 나머지도 미리 받아둔다

    const modal = document.getElementById('draping-modal');
    modal.classList.remove('hidden');
    const S = Math.min(window.innerWidth * 0.9, window.innerHeight * 0.68, 720);
    canvas.width = canvas.height = Math.round(S);
    canvas.style.width = canvas.style.height = Math.round(S) + 'px';

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      video.srcObject = stream;
      await video.play();
    } catch (e) {
      alert('카메라를 켤 수 없습니다: ' + e.message);
      closeDraping();
      return;
    }
    if (!raf) draw();
  }

  function closeDraping() {
    const modal = document.getElementById('draping-modal');
    if (modal) modal.classList.add('hidden');
    if (raf) { cancelAnimationFrame(raf); raf = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  }

  function snapshot() {
    const a = document.createElement('a');
    a.download = `드레이핑_${SEASONS[current].key}.png`;
    a.href = canvas.toDataURL('image/png');
    a.click();
  }

  window.openDraping = openDraping;
  window.closeDraping = closeDraping;
})();
