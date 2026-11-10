/* =====================================================================
 *  makeup.js — 시즌 메이크업 (입술 · 볼터치)
 *  ---------------------------------------------------------------------
 *  결과 화면의 "시즌 메이크업 입어보기" 버튼을 누르면 카메라가 켜지고,
 *  진단된 시즌의 입술·볼터치 색이 내 얼굴 위에 실시간으로 얹힌다.
 *
 *  동작 방식 (전부 브라우저 안에서 — 서버·외부 API·인터넷 연결 모두 필요 없음)
 *    ① MediaPipe FaceLandmarker 로 얼굴 점 478개를 찾는다
 *    ② 입술 모양대로 색을 입히되, 본인 입술색과 섞고 가장자리는 번지게 한다
 *    ③ 볼터치는 본인 피부색과 섞어 광대 쪽으로 넓게 번지게 한다
 *    ④ 프레임마다 점 위치를 평균내서 떨림을 줄인다
 *
 *  쓰는 법
 *      openMakeup('가을 웜');   // 결과의 best_group 을 넘기면 그 시즌이 먼저 선택됨
 *
 *  필요한 파일: static/makeup/ 안의 vendor/ (라이브러리) 와 face_landmarker.task (모델)
 * ===================================================================== */
(function () {
  'use strict';

  const BASE = '/static/makeup/';
  const VISION_JS = BASE + 'vendor/vision_bundle.js';
  const WASM_BASE = BASE + 'vendor/wasm';
  const MODEL_URL = BASE + 'face_landmarker.task';

  // ───────────── 시즌별 색 (static/images/*_화장품.png 의 색 견본에서 뽑음) ─────────────
  const SEASONS = {
    '봄 웜':   { lips:['#f86054','#fd7a7c'],                       blush:['#fe9c91','#fca5a0','#fcb99d','#f97579'], note:'코랄 · 피치 계열' },
    '여름 쿨': { lips:['#fb9ab3','#f0839b','#e5728d','#cf798b'],   blush:['#fdd4e0','#f7def7','#f8b3cc','#f5d8ea'], note:'핑크 · 로즈 · 라벤더 계열' },
    '가을 웜': { lips:['#dca994','#b87761','#c2726c','#d07566'],   blush:['#f9cbb4','#f9b1a6','#f8c1aa','#d48f89'], note:'누드 · 브라운 · 로즈브라운 계열' },
    '겨울 쿨': { lips:['#ef287f','#de043c','#c1134f','#8a1848'],   blush:['#fb2f96','#dd9bf5','#b64982','#d23a6f'], note:'푸시아 · 체리 · 와인 계열' },
  };
  
  // 얼굴 점 번호 (MediaPipe Face Landmarker)
  const LIP_OUT = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185];
  const LIP_IN  = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191];
  const CHEEK_A = [116,117,118,119,100,36,205,187,123];
  const CHEEK_B = [345,346,347,348,329,266,425,411,352];
  const FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
  
  // 합성 방식 (주소 뒤에 ?lipblend=... 로 바꿔 시험할 수 있음)
  // 입술은 'color'(입술 결은 살리고 색만 바꿈), 볼터치는 'source-over'(견본 색 그대로 번지게)가 가장 자연스러웠다.
  const BLEND = { lip: 'color', blush: 'source-over' };
  let LIP_LUMA = 0.3;       // 입술 밝기를 견본 색 밝기 쪽으로 맞추는 정도 (0~1). 1이면 견본과 거의 같고, 낮추면 입술 결이 더 살아남
  
  
  
  // ───────────── 상태 ─────────────
  const S = { season:'봄 웜', lip:0, blush:0, lipAmt:.6, blushAmt:.4, grid:false, showOrig:false };
  let cv = null, ctx = null, msgEl = null;
  const setMsg = t => { if (!msgEl) return; msgEl.textContent = t || ''; msgEl.classList.toggle('hidden', !t); };
  const frameCv = document.createElement('canvas');   // 한 칸을 그리는 임시 캔버스
  const layerCv = document.createElement('canvas');   // 입술/볼터치 색을 칠하는 임시 캔버스
  const video = document.createElement('video'); video.playsInline = true; video.muted = true;
  let landmarker = null, loading = null, stream = null, raf = 0, smooth = null, opened = false;

  // ───────────── 그리기 ─────────────
  const hexA = (hex, a) => { const n = parseInt(hex.slice(1),16); return `rgba(${n>>16},${(n>>8)&255},${n&255},${a})`; };
  const P = (lm,i,w,h) => ({ x: lm[i].x*w, y: lm[i].y*h });
  const centroid = (lm, idx, w, h) => { let x=0,y=0; idx.forEach(i => { x+=lm[i].x*w; y+=lm[i].y*h; }); return { x:x/idx.length, y:y/idx.length }; };
  function loop(lm, idx, w, h){ const p = new Path2D(); idx.forEach((i,k) => { const x=lm[i].x*w, y=lm[i].y*h; k ? p.lineTo(x,y) : p.moveTo(x,y); }); p.closePath(); return p; }
  
  const sampleCv = document.createElement('canvas'); sampleCv.width = sampleCv.height = 1;
  const sampleCtx = sampleCv.getContext('2d', { willReadFrequently:true });
  /** 원본 화면의 (x,y) 주변 평균 색 */
  function sampleColor(src, w, h, x, y, r){
    r = Math.max(2, r|0);
    sampleCtx.clearRect(0,0,1,1);
    sampleCtx.drawImage(src, Math.max(0,x-r), Math.max(0,y-r), r*2, r*2, 0, 0, 1, 1);
    const d = sampleCtx.getImageData(0,0,1,1).data; return [d[0],d[1],d[2]];
  }
  function rgb2hsl(r,g,b){ r/=255;g/=255;b/=255; const mx=Math.max(r,g,b), mn=Math.min(r,g,b); let h=0,s=0,l=(mx+mn)/2;
    if (mx!==mn){ const d=mx-mn; s = l>.5 ? d/(2-mx-mn) : d/(mx+mn);
      h = mx===r ? (g-b)/d+(g<b?6:0) : mx===g ? (b-r)/d+2 : (r-g)/d+4; h/=6; } return [h,s,l]; }
  function hsl2rgb(h,s,l){ const f=(p,q,t)=>{ if(t<0)t+=1; if(t>1)t-=1; return t<1/6?p+(q-p)*6*t : t<1/2?q : t<2/3?p+(q-p)*(2/3-t)*6 : p; };
    if (s===0){ const v=Math.round(l*255); return [v,v,v]; }
    const q = l<.5 ? l*(1+s) : l+s-l*s, p = 2*l-q;
    return [f(p,q,h+1/3),f(p,q,h),f(p,q,h-1/3)].map(v=>Math.round(v*255)); }
  const hexRgb = hex => { const n=parseInt(hex.slice(1),16); return [n>>16,(n>>8)&255,n&255]; };
  /** 견본 색을 본인 색과 섞고, 형광처럼 튀지 않게 채도·밝기를 눌러 준다 */
  function naturalize(hex, base, mix, maxS, lMin, lMax){
    const t = hexRgb(hex), m = base ? t.map((v,i)=> v*(1-mix) + base[i]*mix) : t;
    let [h,s,l] = rgb2hsl(m[0],m[1],m[2]);
    s = Math.min(s, maxS); l = Math.min(lMax, Math.max(lMin, l));
    const o = hsl2rgb(h,s,l); return `rgb(${o[0]},${o[1]},${o[2]})`;
  }
  
  /** src(영상/사진) 위에 화장을 얹어서 target 에 w×h 로 그린다 */
  function composeFrame(target, src, w, h, lm, seasonName, lipIdx, blushIdx, withMakeup){
    target.clearRect(0,0,w,h);
    target.drawImage(src, 0, 0, w, h);
    const pal = SEASONS[seasonName];
    if (!withMakeup || !lm || !pal) return;
  
    layerCv.width = w; layerCv.height = h;
    const L = layerCv.getContext('2d');
    const a = P(lm,234,w,h), b = P(lm,454,w,h);
    const faceW = Math.hypot(b.x-a.x, b.y-a.y);
    const tilt = Math.atan2(b.y-a.y, b.x-a.x);
  
    // 1) 입술 — 본인 입술색에 견본 색을 25% 섞고, 가장자리는 번지게, 가운데는 조금 더 진하게
    if (S.lipAmt > 0){
      const lc = P(lm,14,w,h), lb = P(lm,17,w,h);
      const natural = sampleColor(src, w, h, (lc.x+lb.x)/2, (lc.y+lb.y)/2, faceW*.012);
      const col = naturalize(pal.lips[Math.min(lipIdx, pal.lips.length-1)], natural, .25, .68, .26, .62);
  
      L.clearRect(0,0,w,h);
      const path = new Path2D(); path.addPath(loop(lm,LIP_OUT,w,h)); path.addPath(loop(lm,LIP_IN,w,h));
      L.fillStyle = col; L.fill(path, 'evenodd');
  
      // 입술 윗선·아랫선 쪽은 옅게, 입술 가운데는 진하게 (실제 립 제품의 그러데이션)
      const mx = (P(lm,61,w,h).x + P(lm,291,w,h).x)/2, my = (P(lm,13,w,h).y + P(lm,17,w,h).y)/2;
      L.globalCompositeOperation = 'destination-in';
      const lw = Math.hypot(P(lm,291,w,h).x-P(lm,61,w,h).x, P(lm,291,w,h).y-P(lm,61,w,h).y);
      const gr = L.createRadialGradient(mx,my,lw*.04, mx,my,lw*.62);
      gr.addColorStop(0,'rgba(0,0,0,1)'); gr.addColorStop(.7,'rgba(0,0,0,.85)'); gr.addColorStop(1,'rgba(0,0,0,.55)');
      L.fillStyle = gr; L.fillRect(0,0,w,h);
      L.globalCompositeOperation = 'source-over';
  
      target.save();
      target.filter = `blur(${Math.max(1.2, faceW*.01)}px)`;
      target.globalAlpha = S.lipAmt; target.globalCompositeOperation = BLEND.lip;      // 색상
      target.drawImage(layerCv, 0, 0);
      target.globalAlpha = S.lipAmt * LIP_LUMA; target.globalCompositeOperation = 'luminosity'; // 밝기는 살짝만
      target.drawImage(layerCv, 0, 0);
      target.restore();
  
      // 은은한 윤기: 아랫입술 가운데에 작은 하이라이트
      const gl = P(lm,17,w,h), gu = P(lm,14,w,h);
      const gx = gl.x, gy = gu.y*.4 + gl.y*.6;
      target.save();
      target.translate(gx, gy); target.rotate(tilt); target.scale(1, .32);
      const gg = target.createRadialGradient(0,0,0, 0,0,lw*.2);
      gg.addColorStop(0,'rgba(255,255,255,'+(.22*S.lipAmt)+')'); gg.addColorStop(1,'rgba(255,255,255,0)');
      target.fillStyle = gg; target.beginPath(); target.arc(0,0,lw*.2,0,Math.PI*2); target.fill();
      target.restore();
    }
  
    // 2) 볼터치 — 본인 피부색과 섞어 살구빛으로, 광대 쪽으로 길고 넓게 번지게
    if (S.blushAmt > 0){
      L.clearRect(0,0,w,h);
      L.save();
      L.clip(loop(lm,FACE_OVAL,w,h));
      const R = faceW*.24;
      [[CHEEK_A,-1,454],[CHEEK_B,1,234]].forEach(([idx,side,earIdx]) => {
        const c = centroid(lm, idx, w, h);
        const ear = P(lm, earIdx, w, h);
        c.x += (ear.x-c.x)*.12; c.y += faceW*.045;     // 광대 바깥쪽으로 살짝
        const skin = sampleColor(src, w, h, c.x, c.y, faceW*.03);
        const col = naturalize(pal.blush[Math.min(blushIdx, pal.blush.length-1)], skin, .3, .72, .45, .78);
        const [cr,cg,cb] = col.match(/\d+/g).map(Number);
        const rgba = al => `rgba(${cr},${cg},${cb},${al})`;
        L.save(); L.translate(c.x, c.y); L.rotate(tilt + side*0.0 - (side*0.22));
        L.scale(1.45, .78);
        const g = L.createRadialGradient(0,0,0, 0,0,R);
        g.addColorStop(0, rgba(.9)); g.addColorStop(.35, rgba(.55)); g.addColorStop(.7, rgba(.18)); g.addColorStop(1, rgba(0));
        L.fillStyle = g; L.beginPath(); L.arc(0,0,R,0,Math.PI*2); L.fill(); L.restore();
      });
      L.restore();
      target.save();
      target.filter = `blur(${Math.max(2, faceW*.03)}px)`;
      target.globalAlpha = S.blushAmt; target.globalCompositeOperation = 'multiply';    // 피부 질감 유지
      target.drawImage(layerCv, 0, 0);
      target.globalAlpha = S.blushAmt * .55; target.globalCompositeOperation = 'soft-light'; // 혈색 느낌
      target.drawImage(layerCv, 0, 0);
      target.restore();
    }
  }
  
  /** 지금 화면에 보이는 것 전체(한 장 또는 4계절)를 메인 캔버스에 그린다 */
  function render(src, w, h, lm, flip){
    if (cv.width !== w || cv.height !== h){ cv.width = w; cv.height = h; }
    frameCv.width = w; frameCv.height = h;
    const F = frameCv.getContext('2d');
    const showMakeup = !S.showOrig && S.season !== '__off';
  
    const put = (dx, dy, dw, dh) => {
      ctx.save();
      if (flip){ ctx.translate(dx+dw, dy); ctx.scale(-1,1); ctx.drawImage(frameCv, 0, 0, dw, dh); }
      else ctx.drawImage(frameCv, dx, dy, dw, dh);
      ctx.restore();
    };
  
    if (!S.grid){
      composeFrame(F, src, w, h, lm, S.season, S.lip, S.blush, showMakeup);
      put(0, 0, w, h);
    } else {
      const names = Object.keys(SEASONS);
      names.forEach((n, i) => {
        composeFrame(F, src, w, h, lm, n, 0, 0, !S.showOrig);
        const dx = (i%2) * w/2, dy = Math.floor(i/2) * h/2;
        put(dx, dy, w/2, h/2);
        ctx.save(); ctx.font = `700 ${Math.round(h/24)}px system-ui,"Malgun Gothic",sans-serif`; ctx.textBaseline='top';
        ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(59,47,74,.75)'; ctx.fillStyle = '#fff';
        ctx.strokeText(n, dx+10, dy+8); ctx.fillText(n, dx+10, dy+8); ctx.restore();
      });
    }
  }
  
  
  // ───────────── 얼굴 인식 모델 (처음 한 번만) ─────────────
  function loadModel() {
    if (landmarker) return Promise.resolve(landmarker);
    if (loading) return loading;
    loading = (async () => {
      const { FaceLandmarker, FilesetResolver } = await import(VISION_JS);
      const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
      const make = delegate => FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate },
        runningMode: 'VIDEO', numFaces: 1,
      });
      try { landmarker = await make('GPU'); }
      catch (e) { console.warn('GPU로 못 열어서 CPU로 다시 시도', e); landmarker = await make('CPU'); }
      return landmarker;
    })().catch(err => { loading = null; throw err; });
    return loading;
  }

  // ───────────── 카메라 ─────────────
  async function startCam() {
    smooth = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
    } catch (e) { setMsg('카메라를 열 수 없어요. 브라우저의 카메라 권한을 확인해 주세요.'); return; }
    video.srcObject = stream; await video.play();
    setMsg('');
    cancelAnimationFrame(raf); raf = requestAnimationFrame(camLoop);
  }
  function stopCam() {
    cancelAnimationFrame(raf); raf = 0;
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  }
  function camLoop() {
    if (!stream || !opened) return;
    if (video.readyState >= 2) {
      const w = video.videoWidth, h = video.videoHeight;
      let lm = null;
      const r = landmarker.detectForVideo(video, performance.now());
      if (r.faceLandmarks && r.faceLandmarks[0]) {
        const cur = r.faceLandmarks[0];
        smooth = smooth ? smooth.map((p, i) => ({ x: p.x + (cur[i].x - p.x) * .6, y: p.y + (cur[i].y - p.y) * .6 }))
                        : cur.map(p => ({ x: p.x, y: p.y }));
        lm = smooth; setMsg('');
      } else { smooth = null; setMsg('얼굴이 보이도록 카메라를 봐주세요'); }
      render(video, w, h, lm, true);
    }
    raf = requestAnimationFrame(camLoop);
  }

  // ───────────── 화면 만들기 (한 번만) ─────────────
  const q = id => document.getElementById(id);

  function chips(containerId, colors, key) {
    const el = q(containerId); el.innerHTML = '';
    colors.forEach((c, i) => {
      const b = document.createElement('button'); b.type = 'button';
      b.className = 'mk-chip' + (S[key] === i ? ' on' : '');
      b.style.background = c; b.title = c; b.setAttribute('aria-label', c);
      b.onclick = () => { S[key] = i; refreshUI(); };
      el.appendChild(b);
    });
  }
  function refreshUI() {
    q('mk-seasons').querySelectorAll('button').forEach(b => b.classList.toggle('on', !S.grid && b.dataset.s === S.season));
    q('mk-grid').classList.toggle('on', S.grid);
    const pal = SEASONS[S.season];
    q('mk-note').textContent = S.grid ? '네 시즌의 첫 번째 색으로 비교해요' : (pal ? pal.note : '원본 그대로 보여줘요');
    if (pal) { chips('mk-lip-chips', pal.lips, 'lip'); chips('mk-blush-chips', pal.blush, 'blush'); }
    else { q('mk-lip-chips').innerHTML = ''; q('mk-blush-chips').innerHTML = ''; }
  }

  function buildUI() {
    if (q('makeup-modal')) return;
    const wrap = document.createElement('div');
    wrap.id = 'makeup-modal';
    wrap.className = 'mk-modal hidden';
    wrap.innerHTML = `
      <div class="mk-inner">
        <div class="mk-stage">
          <canvas id="mk-canvas" width="640" height="480"></canvas>
          <div id="mk-msg" class="mk-msg">얼굴 인식 모델을 불러오는 중입니다…</div>
        </div>
        <p class="mk-hint">시즌을 누르면 그 시즌의 입술·볼터치 색이 입혀져요 · 얼굴을 정면으로, 밝은 곳에서 보면 더 자연스러워요</p>

        <div class="mk-block">
          <span class="mk-label">시즌</span>
          <div class="mk-row" id="mk-seasons"></div>
          <p class="mk-note" id="mk-note"></p>
          <div class="mk-row"><button id="mk-grid" class="mk-btn" type="button">4계절 한꺼번에 보기</button></div>
        </div>

        <div class="mk-two">
          <div class="mk-block"><span class="mk-label">입술 색</span><div class="mk-row" id="mk-lip-chips"></div></div>
          <div class="mk-block"><span class="mk-label">볼터치 색</span><div class="mk-row" id="mk-blush-chips"></div></div>
        </div>

        <div class="mk-controls">
          <label class="mk-range">입술 <input id="mk-lip-amt" type="range" min="0" max="100" value="60"></label>
          <label class="mk-range">볼터치 <input id="mk-blush-amt" type="range" min="0" max="100" value="40"></label>
        </div>

        <div class="mk-actions">
          <button id="mk-orig" class="btn-secondary" type="button">누르고 있으면 원본</button>
          <button id="mk-save" class="btn-primary" type="button">사진 저장</button>
          <button id="mk-close" class="btn-secondary" type="button">닫기</button>
        </div>
      </div>`;
    document.body.appendChild(wrap);

    cv = q('mk-canvas'); ctx = cv.getContext('2d'); msgEl = q('mk-msg');

    const seasonsEl = q('mk-seasons');
    Object.keys(SEASONS).forEach(name => {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'mk-btn'; b.textContent = name; b.dataset.s = name;
      b.onclick = () => { S.season = name; S.lip = 0; S.blush = 0; S.grid = false; refreshUI(); };
      seasonsEl.appendChild(b);
    });
    const off = document.createElement('button'); off.type = 'button'; off.className = 'mk-btn'; off.textContent = '화장 안 함'; off.dataset.s = '__off';
    off.onclick = () => { S.season = '__off'; S.grid = false; refreshUI(); };
    seasonsEl.appendChild(off);

    q('mk-grid').onclick = () => { S.grid = !S.grid; if (S.grid && S.season === '__off') S.season = '봄 웜'; refreshUI(); };
    q('mk-lip-amt').oninput = e => { S.lipAmt = e.target.value / 100; };
    q('mk-blush-amt').oninput = e => { S.blushAmt = e.target.value / 100; };

    const orig = q('mk-orig');
    orig.addEventListener('pointerdown', () => { S.showOrig = true; });
    ['pointerup', 'pointerleave', 'pointercancel'].forEach(e => orig.addEventListener(e, () => { S.showOrig = false; }));

    q('mk-save').onclick = () => {
      const FILE = { '봄 웜': 'spring', '여름 쿨': 'summer', '가을 웜': 'autumn', '겨울 쿨': 'winter' };
      cv.toBlob(b => {
        if (!b) return;
        const a = document.createElement('a');
        a.href = URL.createObjectURL(b);
        a.download = 'makeup_' + (S.grid ? '4seasons' : (FILE[S.season] || 'original')) + '.png';
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 3000);
      }, 'image/png');
    };

    q('mk-close').onclick = closeMakeup;
    wrap.addEventListener('click', e => { if (e.target === wrap) closeMakeup(); });
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && opened) closeMakeup(); });
  }

  // ───────────── 열기 / 닫기 ─────────────
  function seasonFrom(group) {
    const g = String(group || '');
    return Object.keys(SEASONS).find(n => g.includes(n.charAt(0))) || '봄 웜';
  }

  async function openMakeup(bestGroup) {
    buildUI();
    S.season = seasonFrom(bestGroup); S.lip = 0; S.blush = 0; S.grid = false; S.showOrig = false;
    refreshUI();
    q('makeup-modal').classList.remove('hidden');
    document.body.style.overflow = 'hidden';
    opened = true;
    setMsg('얼굴 인식 모델을 불러오는 중입니다…');
    try {
      await loadModel();
      if (!opened) return;                       // 불러오는 사이에 닫았으면 카메라 켜지 않음
      await landmarker.setOptions({ runningMode: 'VIDEO' });
      await startCam();
    } catch (err) {
      console.error(err);
      setMsg('모델을 불러오지 못했어요: ' + err.message);
    }
  }

  function closeMakeup() {
    opened = false; stopCam();
    const m = q('makeup-modal'); if (m) m.classList.add('hidden');
    document.body.style.overflow = '';
  }

  window.openMakeup = openMakeup;

  // 결과 화면의 버튼 연결 (진단된 시즌은 화면에 표시된 이름에서 읽는다)
  document.addEventListener('DOMContentLoaded', () => {
    const btn = q('makeup-btn');
    if (btn) btn.addEventListener('click', () => {
      const el = q('best-season-name');
      openMakeup(el ? el.textContent : '');
    });
  });
})();
