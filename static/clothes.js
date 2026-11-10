/* =====================================================================
 *  clothes.js — 실시간 옷 색 바꾸기 (디지털 드레이핑 · 부가기능)
 *  ---------------------------------------------------------------------
 *  카메라를 켠 상태에서 색상 버튼을 누르면 내가 입고 있는 옷의 색만
 *  실시간으로 바뀐다. 주름·그림자·질감은 그대로 남는다.
 *
 *  동작 방식 (전부 브라우저 안에서, 서버·외부 API 안 씀)
 *    ① MediaPipe ImageSegmenter(selfie_multiclass) 로 '옷' 영역만 분리
 *    ② 그 영역의 밝기(명암)는 유지한 채 색만 목표색으로 갈아끼움
 *       └ out = 목표색 × (픽셀밝기 / 영역평균밝기)   ← 주름이 살아남는 이유
 *    ③ 마스크 가장자리는 부드럽게, 프레임 간에는 평균내서 떨림 제거
 *
 *  쓰는 법
 *      openClothes('가을 웜');   // 결과의 best_group 을 넘기면 그 시즌 탭이 먼저 열림
 *
 *  ※ 인터넷이 필요합니다 (모델 파일을 CDN 에서 받음).
 *    오프라인으로 쓰려면 맨 아래 '오프라인 설치' 주석 참고.
 * ===================================================================== */
(function () {
  'use strict';

  // ── 모델 주소 ──────────────────────────────────────────────────
  // 오프라인으로 돌리려면 이 둘을 '/static/vendor/...' 로 바꾸세요 (맨 아래 설명)
  const VISION_MJS = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs';
  const WASM_BASE  = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
  const MODEL_URL  = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/'
                   + 'selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite';

  // selfie_multiclass 가 내놓는 분류 번호
  //   0 배경 / 1 머리카락 / 2 몸 피부 / 3 얼굴 피부 / 4 옷 / 5 기타(액세서리)
  const CLOTHES = 4;

  // ── 시즌별 색 팔레트 ───────────────────────────────────────────
  const PALETTE = [
    { key: '봄', label: '봄 웜', colors: [
      ['코랄',      '#FF9E7A'], ['피치핑크', '#FFB3C1'], ['골드옐로우', '#FFD166'],
      ['라이트그린', '#A8D672'], ['오렌지레드', '#FF6B4A'],
    ]},
    { key: '여름', label: '여름 쿨', colors: [
      ['스카이블루', '#A8DCE7'], ['라벤더',   '#C9B6E4'], ['로즈핑크', '#F4A6C0'],
      ['파우더블루', '#8FB8DE'], ['실버그레이', '#C7CED4'],
    ]},
    { key: '가을', label: '가을 웜', colors: [
      ['카멜',   '#B88354'], ['브라운',   '#8C6239'], ['올리브', '#6B7A3A'],
      ['테라코타', '#C1573B'], ['머스터드', '#D9A441'],
    ]},
    { key: '겨울', label: '겨울 쿨', colors: [
      ['로열블루', '#2B5299'], ['마젠타',   '#C2185B'], ['에메랄드', '#00897B'],
      ['퓨어화이트', '#F2F4F7'], ['블랙',    '#1C1C1E'],
    ]},
  ];

  // ── 처리 해상도 ────────────────────────────────────────────────
  // 픽셀을 하나씩 계산하므로 너무 크면 느려진다. 640×360 이면 30fps 나온다.
  const W = 640, H = 360;

  let segmenter = null, loading = null;
  let stream = null, raf = null, video = null;
  let view = null, vctx = null;          // 화면에 보이는 캔버스
  let work = null, wctx = null;          // 실제 계산하는 캔버스
  let maskC = null, maskX = null;        // 모델이 준 작은 마스크
  let softC = null, softX = null;        // 그걸 W×H 로 키운 부드러운 마스크

  let target = null;                     // 지금 고른 색 [r,g,b] · null 이면 원본
  let strength = 1.0;                    // 1.0 = 완전히 갈아끼움
  let showMask = false;                  // 디버그: 옷 인식 영역 보기
  let alphaEMA = null;                   // 프레임 간 마스크 평균 (떨림 제거)
  let lumEMA = null;                     // 옷 영역 평균 밝기 (급변 방지)

  const hex2rgb = h => [
    parseInt(h.slice(1, 3), 16),
    parseInt(h.slice(3, 5), 16),
    parseInt(h.slice(5, 7), 16),
  ];

  // ── 모델 불러오기 (처음 한 번만, 10~20초 걸릴 수 있음) ──────────
  function loadSegmenter() {
    if (segmenter) return Promise.resolve(segmenter);
    if (loading) return loading;
    loading = (async () => {
      const { FilesetResolver, ImageSegmenter } = await import(VISION_MJS);
      const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
      segmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      });
      return segmenter;
    })().catch(err => {
      loading = null;                    // 실패하면 다음에 다시 시도할 수 있게
      throw err;
    });
    return loading;
  }

  // ── 화면 만들기 (한 번만) ──────────────────────────────────────
  function buildUI() {
    if (document.getElementById('clothes-modal')) return;

    const swatches = PALETTE.map(g => `
      <div class="cl-group">
        <span class="cl-group-name">${g.label}</span>
        <div class="cl-swatches">
          ${g.colors.map(([name, hex]) => `
            <button class="cl-swatch" data-hex="${hex}" title="${name}"
                    style="background:${hex}"><span>${name}</span></button>`).join('')}
        </div>
      </div>`).join('');

    const wrap = document.createElement('div');
    wrap.id = 'clothes-modal';
    wrap.className = 'cl-modal hidden';
    wrap.innerHTML = `
      <div class="cl-inner">
        <div class="cl-stage">
          <canvas id="cl-canvas"></canvas>
          <div id="cl-status" class="cl-status">옷 인식 모델을 불러오는 중입니다…</div>
        </div>

        <p class="cl-hint">색을 누르면 지금 입고 있는 옷의 색만 바뀝니다 · 어깨와 가슴이 보이게 앉아주세요</p>

        <div class="cl-palette">${swatches}</div>

        <div class="cl-controls">
          <label class="cl-range">진하기
            <input id="cl-strength" type="range" min="0.3" max="1" step="0.05" value="1">
          </label>
          <label class="cl-check">
            <input id="cl-maskview" type="checkbox"> 인식 영역 보기
          </label>
        </div>

        <div class="cl-actions">
          <button id="cl-reset" class="btn-secondary" type="button">원본</button>
          <button id="cl-shot" class="btn-primary" type="button">사진 저장</button>
          <button id="cl-close" class="btn-secondary" type="button">닫기</button>
        </div>
      </div>`;
    document.body.appendChild(wrap);

    wrap.querySelectorAll('.cl-swatch').forEach(b => {
      b.addEventListener('click', () => {
        wrap.querySelectorAll('.cl-swatch').forEach(x => x.classList.remove('on'));
        b.classList.add('on');
        target = hex2rgb(b.dataset.hex);
      });
    });
    document.getElementById('cl-reset').addEventListener('click', () => {
      target = null;
      wrap.querySelectorAll('.cl-swatch').forEach(x => x.classList.remove('on'));
    });
    document.getElementById('cl-strength').addEventListener('input', e => {
      strength = Number(e.target.value);
    });
    document.getElementById('cl-maskview').addEventListener('change', e => {
      showMask = e.target.checked;
    });
    document.getElementById('cl-close').addEventListener('click', closeClothes);
    document.getElementById('cl-shot').addEventListener('click', snapshot);
    wrap.addEventListener('click', e => { if (e.target === wrap) closeClothes(); });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && !wrap.classList.contains('hidden')) closeClothes();
    });

    view = document.getElementById('cl-canvas');
    vctx = view.getContext('2d');
    view.width = W; view.height = H;

    work  = document.createElement('canvas'); work.width = W;  work.height = H;
    wctx  = work.getContext('2d', { willReadFrequently: true });
    maskC = document.createElement('canvas');
    maskX = maskC.getContext('2d', { willReadFrequently: true });
    softC = document.createElement('canvas'); softC.width = W; softC.height = H;
    softX = softC.getContext('2d', { willReadFrequently: true });

    video = document.createElement('video');
    video.autoplay = true; video.playsInline = true; video.muted = true;
  }

  function status(msg) {
    const el = document.getElementById('cl-status');
    if (!el) return;
    if (msg) { el.textContent = msg; el.classList.remove('hidden'); }
    else el.classList.add('hidden');
  }

  // ── 매 프레임 ──────────────────────────────────────────────────
  function tick() {
    raf = requestAnimationFrame(tick);
    if (!video.videoWidth) return;

    // ① 원본 프레임을 작업 캔버스에 (좌우반전은 맨 마지막에 한다 —
    //    여기서 뒤집으면 모델이 준 마스크와 위치가 어긋난다)
    wctx.drawImage(video, 0, 0, W, H);

    // ② 옷 영역 분리
    let alpha = null;
    if (segmenter) {
      try {
        alpha = getClothesAlpha();
      } catch (e) {
        console.warn('[옷 색 바꾸기] 분리 실패:', e);
      }
    }

    // ③ 색 갈아끼우기
    if (alpha && (target || showMask)) recolor(alpha);

    // ④ 화면에 거울처럼 좌우반전해서 그리기
    vctx.save();
    vctx.translate(W, 0);
    vctx.scale(-1, 1);
    vctx.drawImage(work, 0, 0, W, H);
    vctx.restore();
  }

  /* 모델이 준 분류 마스크 → W×H 짜리 부드러운 알파(0~1) 배열 */
  function getClothesAlpha() {
    let result = null;
    const ret = segmenter.segmentForVideo(video, performance.now(), r => { result = r; });
    if (!result) result = ret;                     // 버전에 따라 반환형이 다르다
    if (!result || !result.categoryMask) return null;

    const m = result.categoryMask;
    const mw = m.width, mh = m.height;
    const cat = m.getAsUint8Array();

    // 옷(4) 인 곳만 불투명하게 칠한 작은 이미지를 만든다
    if (maskC.width !== mw || maskC.height !== mh) { maskC.width = mw; maskC.height = mh; }
    const img = maskX.createImageData(mw, mh);
    const d = img.data;
    for (let i = 0, j = 0; i < cat.length; i++, j += 4) {
      d[j] = d[j + 1] = d[j + 2] = 255;
      d[j + 3] = cat[i] === CLOTHES ? 255 : 0;
    }
    maskX.putImageData(img, 0, 0);
    result.close();                                // 안 하면 메모리가 샌다

    // 작은 마스크를 W×H 로 확대 — 보간 덕분에 가장자리가 부드러워진다
    softX.clearRect(0, 0, W, H);
    softX.imageSmoothingEnabled = true;
    softX.imageSmoothingQuality = 'high';
    softX.drawImage(maskC, 0, 0, W, H);
    const sd = softX.getImageData(0, 0, W, H).data;

    // 프레임끼리 평균내서 떨림을 없앤다
    const n = W * H;
    if (!alphaEMA || alphaEMA.length !== n) alphaEMA = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const cur = sd[i * 4 + 3] / 255;
      alphaEMA[i] = alphaEMA[i] * 0.45 + cur * 0.55;
    }
    return alphaEMA;
  }

  /* 밝기는 그대로 두고 색만 바꾼다 — 주름·그림자가 살아 있는 비결 */
  function recolor(alpha) {
    const frame = wctx.getImageData(0, 0, W, H);
    const p = frame.data;
    const n = W * H;

    if (showMask) {                                 // 디버그: 인식 영역을 초록으로
      for (let i = 0, j = 0; i < n; i++, j += 4) {
        const a = alpha[i];
        if (a > 0.05) {
          p[j]     = p[j]     * (1 - a * 0.6);
          p[j + 1] = p[j + 1] * (1 - a * 0.6) + 255 * a * 0.6;
          p[j + 2] = p[j + 2] * (1 - a * 0.6);
        }
      }
      wctx.putImageData(frame, 0, 0);
      return;
    }

    // 옷 영역의 평균 밝기 — 이걸 기준으로 '여기보다 밝다/어둡다' 를 판단한다
    let sum = 0, cnt = 0;
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      if (alpha[i] > 0.5) {
        sum += 0.299 * p[j] + 0.587 * p[j + 1] + 0.114 * p[j + 2];
        cnt++;
      }
    }
    if (cnt < 300) return;                          // 옷이 거의 안 보이면 건너뛴다
    const cur = sum / cnt;
    lumEMA = lumEMA == null ? cur : lumEMA * 0.8 + cur * 0.2;
    const base = Math.max(30, lumEMA);              // 0 으로 나누기 방지

    const [tr, tg, tb] = target;
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      const a = alpha[i] * strength;
      if (a <= 0.02) continue;

      const r = p[j], g = p[j + 1], b = p[j + 2];
      const k = (0.299 * r + 0.587 * g + 0.114 * b) / base;   // 이 픽셀의 상대 밝기

      let nr, ng, nb;
      if (k <= 1) {                                 // 그늘·주름 → 목표색을 어둡게
        nr = tr * k; ng = tg * k; nb = tb * k;
      } else {                                      // 하이라이트 → 흰색 쪽으로 (광택 유지)
        const t = Math.min(k - 1, 1) * 0.75;
        nr = tr + (255 - tr) * t;
        ng = tg + (255 - tg) * t;
        nb = tb + (255 - tb) * t;
      }

      p[j]     = r + (nr - r) * a;
      p[j + 1] = g + (ng - g) * a;
      p[j + 2] = b + (nb - b) * a;
    }
    wctx.putImageData(frame, 0, 0);
  }

  // ── 열기 / 닫기 / 저장 ─────────────────────────────────────────
  async function openClothes(bestGroup) {
    buildUI();

    // 진단 결과 시즌의 첫 번째 색을 기본으로 골라둔다
    const gi = PALETTE.findIndex(g => (bestGroup || '').includes(g.key));
    const first = document.querySelectorAll('.cl-swatch')[gi >= 0 ? gi * 5 : 0];
    if (first) first.click();

    document.getElementById('clothes-modal').classList.remove('hidden');
    status('카메라를 켜는 중입니다…');

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      video.srcObject = stream;
      await video.play();
    } catch (e) {
      status('카메라를 켤 수 없습니다: ' + e.message);
      return;
    }

    if (!raf) tick();                               // 모델이 없어도 영상은 먼저 보여준다
    status('옷 인식 모델을 불러오는 중입니다… (처음 한 번만, 10초쯤 걸려요)');

    try {
      await loadSegmenter();
      status('');
    } catch (e) {
      console.error(e);
      status('옷 인식 모델을 불러오지 못했습니다. 인터넷 연결을 확인해주세요.');
    }
  }

  function closeClothes() {
    const modal = document.getElementById('clothes-modal');
    if (modal) modal.classList.add('hidden');
    if (raf) { cancelAnimationFrame(raf); raf = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    alphaEMA = null; lumEMA = null;
  }

  function snapshot() {
    const a = document.createElement('a');
    a.download = '옷색바꾸기.png';
    a.href = view.toDataURL('image/png');
    a.click();
  }

  window.openClothes = openClothes;
  window.closeClothes = closeClothes;
})();

/* =====================================================================
 *  오프라인 설치 (발표장 인터넷이 불안할 때)
 *  ---------------------------------------------------------------------
 *  1) 아래 두 가지를 미리 받아서 static/vendor/ 에 넣는다
 *     · https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/   → vendor/tasks-vision/
 *       (vision_bundle.mjs 와 wasm/ 폴더 전체)
 *     · 위 MODEL_URL 의 .tflite 파일                                   → vendor/selfie_multiclass.tflite
 *  2) 이 파일 맨 위 세 상수를 바꾼다
 *       VISION_MJS = '/static/vendor/tasks-vision/vision_bundle.mjs';
 *       WASM_BASE  = '/static/vendor/tasks-vision/wasm';
 *       MODEL_URL  = '/static/vendor/selfie_multiclass.tflite';
 * ===================================================================== */
