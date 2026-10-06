document.addEventListener('DOMContentLoaded', async () => {
    const video = document.getElementById('webcam-video');
    const captureBtn = document.getElementById('capture-btn');
    const canvas = document.getElementById('snapshot-canvas');
    const resetBtn = document.getElementById('reset-btn');
    
    const webcamSection = document.getElementById('webcam-section');
    const loadingSection = document.getElementById('loading-section');
    const resultSection = document.getElementById('result-section');
    const previewSection = document.getElementById('preview-section');
    const previewCanvas = document.getElementById('preview-canvas');

    let stream = null;
    let faceCanvas = null;   // 얼굴 가이드 영역만 잘라낸 캔버스
    let currentSeason = null;

    // 시즌별 가이드 이미지 (static/images/ 폴더에 넣어두세요)
    const SEASON_IMAGES = {
        '봄':   { style: '최종_봄.png',   makeup: '봄_화장품.png', clothes: '봄_코디.png',  title: '봄 웜톤 가이드' },
        '여름': { style: '최종_여름.png', makeup: '여름_화장품.png', clothes: '여름_코디.png', title: '여름 쿨톤 가이드' },
        '가을': { style: '최종_가을.png', makeup: '가을_화장품.png', clothes: '가을_코디.png', title: '가을 웜톤 가이드' },
        '겨울': { style: '최종_겨울.png', makeup: '겨울_화장품.png', clothes: '겨울_코디.png', title: '겨울 쿨톤 가이드' }
    };

    // 얼굴 합성용 시즌 이미지 (static/images/ 폴더)
    const PREVIEW_IMAGES = {
        spring: { label: '봄 웜',   file: '봄_웜_확인.png' },
        summer: { label: '여름 쿨', file: '여름_쿨_확인.png' },
        autumn: { label: '가을 웜', file: '가을_웜_확인.png' },
        winter: { label: '겨울 쿨', file: '겨울_쿨_확인.png' }
    };
    // 시즌 이미지(1254x1254 기준)의 회색 타원 위치. 어긋나면 이 숫자만 조절하세요.
    const OVAL = { size: 1254, cx: 626, cy: 628, rx: 311, ry: 396 };

    function renderSeasonGuide(bestGroup) {
        const card = document.getElementById('season-guide-card');
        const key = Object.keys(SEASON_IMAGES).find(k => String(bestGroup).includes(k));
        if (!key) { card.classList.add('hidden'); return; }
        const info = SEASON_IMAGES[key];
        const base = '/static/images/';
        document.getElementById('season-guide-title').textContent = info.title;
        document.getElementById('season-guide-style').src = base + encodeURIComponent(info.style);
        document.getElementById('season-guide-makeup').src = base + encodeURIComponent(info.makeup);
        document.getElementById('season-guide-clothes').src = base + encodeURIComponent(info.clothes);
        card.classList.remove('hidden');
    }

    // 1. 웹캠 켜기
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } } 
        });
        video.srcObject = stream;
    } catch (err) {
        alert('카메라 권한을 허용해야 퍼스널컬러 진단이 가능합니다.');
        console.error(err);
    }

    // 2. 촬영 및 서버 전송
    captureBtn.addEventListener('click', async () => {
        // 카메라가 까맣거나 준비가 덜 된 상태면 튕겨냄
        if (video.videoWidth === 0 || video.videoHeight === 0) {
            alert('카메라 화면이 켜질 때까지 조금만 기다려주세요!');
            return; 
        }

        // 가이드 박스 좌표는 화면을 숨기기 '전에' 재야 한다.
        // 숨긴 요소는 크기가 0 이라 계산이 NaN 이 되고, JSON 으로는 null 이 되어 서버가 오류를 낸다.
        const boxCoords = getGuideCoords('guide-box');
        const faceCoords = getGuideCoords('face-guide');

        webcamSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        try {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // 얼굴 가이드 영역만 잘라서 보관 (흰 종이 부분 제외)
            faceCanvas = cropToCanvas(canvas, faceCoords);

            canvas.toBlob(async (blob) => {
                try {
                    if (!blob) throw new Error("이미지 캡처에 실패했습니다.");

                    const formData = new FormData();
                    formData.append('image', blob, 'capture.jpg');
                    
                    // 좌표가 정상일 때만 보낸다. 안 보내면 백엔드가 흰 종이를 알아서 찾는다.
                    if (boxCoords) formData.append('coords', JSON.stringify(boxCoords));

                    const response = await fetch('/api/diagnose', { method: 'POST', body: formData });

                    // 서버가 JSON 이 아닌 걸 돌려줄 수도 있다(파이썬 오류 페이지 등).
                    // 그냥 response.json() 하면 "Unexpected token '<'" 로 터지므로 먼저 글자로 받는다.
                    const raw = await response.text();
                    let data;
                    try {
                        data = JSON.parse(raw);
                    } catch (e) {
                        throw new Error(
                            `서버가 오류를 냈습니다 (HTTP ${response.status}). ` +
                            `python app.py 를 실행한 터미널에 찍힌 빨간 글씨를 확인해주세요.`);
                    }

                    // 촬영 조건 미달(400)은 오류가 아니라 '다시 찍어주세요' 안내다.
                    // 백엔드가 사람이 읽을 수 있는 문장을 data.error 로 보내므로 그대로 보여준다.
                    if (!response.ok) {
                        showRetake(data.error || '진단할 수 없는 사진입니다.', data.reasons || []);
                        return;
                    }
                    const tempImageUrl = URL.createObjectURL(blob);
                    renderResults(data, tempImageUrl); 

                } catch (error) {
                    handleError(error); // 내부 에러 발생 시 처리
                }
            }, 'image/jpeg', 0.95);
        } catch (error) {
            handleError(error); // 외부 에러 발생 시 처리
        }
    });

    // 가이드 박스가 실제 영상(object-fit: cover)의 어느 부분인지 0~1 비율로 계산
    function getGuideCoords(elId = 'guide-box') {
        const container = video.parentElement.getBoundingClientRect();
        const guide = document.getElementById(elId).getBoundingClientRect();
        const vw = video.videoWidth, vh = video.videoHeight;
        const scale = Math.max(container.width / vw, container.height / vh);
        const dw = vw * scale, dh = vh * scale;
        const offX = (container.width - dw) / 2, offY = (container.height - dh) / 2;
        const clamp = v => Math.max(0, Math.min(1, v));
        const x = clamp((guide.left - container.left - offX) / dw);
        const y = clamp((guide.top - container.top - offY) / dh);
        const w = Math.min(1 - x, guide.width / dw);
        const h = Math.min(1 - y, guide.height / dh);
        const box = { x, y, width: w, height: h };
        // NaN·Infinity 가 섞이면 JSON 에서 null 이 되어 서버가 오류를 낸다 → 아예 안 보낸다
        const ok = Object.values(box).every(v => Number.isFinite(v)) && w > 0.01 && h > 0.01;
        if (!ok) { console.warn('가이드 박스 좌표를 계산하지 못했습니다.', elId, box); return null; }
        return box;
    }

    // 얼굴 가이드 영역만 잘라내기 (좌표 계산 실패 시 중앙 정사각형으로 대체)
    function cropToCanvas(src, c) {
        let sx, sy, sw, sh;
        if (c) {
            sx = c.x * src.width;  sy = c.y * src.height;
            sw = c.width * src.width; sh = c.height * src.height;
        } else {
            sw = sh = Math.min(src.width, src.height);
            sx = (src.width - sw) / 2; sy = (src.height - sh) / 2;
        }
        const out = document.createElement('canvas');
        out.width = Math.round(sw); out.height = Math.round(sh);
        out.getContext('2d').drawImage(src, sx, sy, sw, sh, 0, 0, out.width, out.height);
        return out;
    }

    // 2-1. 촬영 조건 미달 → 촬영 화면으로 돌아가며 이유를 알려준다
    function showRetake(message, reasons) {
        loadingSection.classList.add('hidden');
        webcamSection.classList.remove('hidden');
        alert('다시 촬영해주세요\n\n' + message +
              (reasons.length > 1 ? '\n\n- ' + reasons.slice(1).join('\n- ') : ''));
    }

    // 3. 에러 발생 시 로딩 화면 끄고 복귀하는 함수 (무한 로딩 방지)
    function handleError(error) {
        console.error(error);
        alert('진단 중 오류가 발생했습니다: ' + error.message);
        loadingSection.classList.add('hidden');
        webcamSection.classList.remove('hidden');
    }

    // 4. 다시 검사하기
    resetBtn.addEventListener('click', () => {
        resultSection.classList.add('hidden');
        previewSection.classList.add('hidden');
        webcamSection.classList.remove('hidden');
    });

    // 5. 결과 렌더링 함수
    function renderResults(data, imageUrl) {
        loadingSection.classList.add('hidden');
        resultSection.classList.remove('hidden');

        document.getElementById('result-profile-img').src = imageUrl;
        document.getElementById('reliability-text').textContent = `신뢰도: ${data.reliability}%`;
        document.getElementById('reliability-desc').textContent = data.reliability_msg;

        const listContainer = document.getElementById('percentage-list');
        listContainer.innerHTML = ''; 
        
        data.percentages.forEach(item => {
            const li = document.createElement('li');
            li.className = 'percentage-item';
            li.innerHTML = `
                <div class="percentage-head">
                    <div class="season-name-wrapper">
                        <span class="color-dot" style="background-color: ${item.color};"></span>
                        <span>${item.name}</span>
                    </div>
                    <span class="percentage-value">${item.value}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="background-color: ${item.color};" data-value="${item.value}"></div>
                </div>
            `;
            listContainer.appendChild(li);
        });

        // 바 애니메이션 (0% -> 값)
        requestAnimationFrame(() => requestAnimationFrame(() => {
            listContainer.querySelectorAll('.bar-fill').forEach(bar => {
                bar.style.width = Math.max(0, Math.min(100, bar.dataset.value)) + '%';
            });
        }));

        document.getElementById('best-season-name').textContent = data.best_group;

        renderSeasonGuide(data.best_group);

        // 웜/쿨 판정 — 4계절보다 신뢰도가 높은 주 결과
        const tone = data.tone || {};
        document.getElementById('tone-line').innerHTML =
            `${tone.label ?? '-'} 톤 · 웜 ${tone.warm ?? '-'}% / 쿨 ${tone.cool ?? '-'}%` +
            `<br><span class="tone-sub">${data.message ?? ''}</span>`;

        // 조명 보정 비교 — CSS 효과가 아니라 백엔드가 실제로 보정한 사진을 쓴다
        document.getElementById('img-original').src = imageUrl;
        const corrected = document.getElementById('img-corrected');
        corrected.style.filter = '';
        corrected.src = data.corrected_image || imageUrl;
    }

    // ===== 6. 시즌 컬러 휠에 내 얼굴 합성 =====
    const imgCache = {};
    function loadImage(src) {
        if (imgCache[src]) return Promise.resolve(imgCache[src]);
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => { imgCache[src] = img; resolve(img); };
            img.onerror = () => reject(new Error('시즌 이미지를 불러오지 못했습니다: ' + src));
            img.src = src;
        });
    }

    // 시즌 이미지 위 회색 타원에 얼굴을 꽉 차게 합성
    async function renderPreview(key) {
        const info = PREVIEW_IMAGES[key];
        if (!info || !faceCanvas) return;
        const bg = await loadImage('/static/images/' + encodeURIComponent(info.file));

        previewCanvas.width = bg.naturalWidth;
        previewCanvas.height = bg.naturalHeight;
        const ctx = previewCanvas.getContext('2d');
        ctx.drawImage(bg, 0, 0);

        const k = bg.naturalWidth / OVAL.size;
        const cx = OVAL.cx * k, cy = OVAL.cy * k, rx = OVAL.rx * k, ry = OVAL.ry * k;

        ctx.save();
        ctx.beginPath();
        ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
        ctx.clip();
        // cover 방식: 타원을 빈틈없이 채우고 넘치는 부분은 잘림
        const s = Math.max((rx * 2) / faceCanvas.width, (ry * 2) / faceCanvas.height);
        const dw = faceCanvas.width * s, dh = faceCanvas.height * s;
        ctx.drawImage(faceCanvas, cx - dw / 2, cy - dh / 2, dw, dh);
        ctx.restore();

        currentSeason = key;
        document.getElementById('preview-title').textContent = info.label + ' 컬러 확인';
        document.querySelectorAll('#preview-section .season-pick-btn').forEach(b =>
            b.classList.toggle('active', b.dataset.season === key));
    }

    async function openPreview(key) {
        try {
            await renderPreview(key);
            resultSection.classList.add('hidden');
            previewSection.classList.remove('hidden');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        } catch (e) {
            alert(e.message);
        }
    }

    // 시즌 버튼 (결과 화면 + 합성 화면 공용)
    document.querySelectorAll('.season-pick-btn').forEach(btn => {
        btn.addEventListener('click', () => openPreview(btn.dataset.season));
    });

    document.getElementById('preview-back-btn').addEventListener('click', () => {
        previewSection.classList.add('hidden');
        resultSection.classList.remove('hidden');
    });

    document.getElementById('preview-save-btn').addEventListener('click', () => {
        previewCanvas.toBlob(blob => {
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `personal-color-${currentSeason}.png`;
            a.click();
            URL.revokeObjectURL(a.href);
        }, 'image/png');
    });
});