document.addEventListener('DOMContentLoaded', async () => {
    const video = document.getElementById('webcam-video');
    const captureBtn = document.getElementById('capture-btn');
    const canvas = document.getElementById('snapshot-canvas');
    const resetBtn = document.getElementById('reset-btn');
    
    const webcamSection = document.getElementById('webcam-section');
    const loadingSection = document.getElementById('loading-section');
    const resultSection = document.getElementById('result-section');

    let stream = null;

    // 시즌별 가이드 이미지 (static/images/ 폴더에 넣어두세요)
    const SEASON_IMAGES = {
        '봄':   { style: '최종_봄.png',   makeup: '봄_화장품.png', clothes: '여름_코디.png',  title: '봄 웜톤 가이드' },
        '여름': { style: '최종_여름.png', makeup: '여름_화장품.png', clothes: '여름_코디.png', title: '여름 쿨톤 가이드' },
        '가을': { style: '최종_가을.png', makeup: '가을_화장품.png', clothes: '가을_코디.png', title: '가을 웜톤 가이드' },
        '겨울': { style: '최종_겨울.png', makeup: '겨울_화장품.png', clothes: '여름_코디.png', title: '겨울 쿨톤 가이드' }
    };

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

        webcamSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        try {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            canvas.toBlob(async (blob) => {
                try {
                    if (!blob) throw new Error("이미지 캡처에 실패했습니다.");

                    const formData = new FormData();
                    formData.append('image', blob, 'capture.jpg');
                    
                    const boxCoords = getGuideCoords();
                    formData.append('coords', JSON.stringify(boxCoords));

                    const response = await fetch('/api/diagnose', { method: 'POST', body: formData });
                    if (!response.ok) throw new Error("서버 응답 오류 (Flask 백엔드를 확인하세요)");
                    
                    const data = await response.json();
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
    function getGuideCoords() {
        const container = video.parentElement.getBoundingClientRect();
        const guide = document.getElementById('guide-box').getBoundingClientRect();
        const vw = video.videoWidth, vh = video.videoHeight;
        const scale = Math.max(container.width / vw, container.height / vh);
        const dw = vw * scale, dh = vh * scale;
        const offX = (container.width - dw) / 2, offY = (container.height - dh) / 2;
        const clamp = v => Math.max(0, Math.min(1, v));
        const x = clamp((guide.left - container.left - offX) / dw);
        const y = clamp((guide.top - container.top - offY) / dh);
        const w = Math.min(1 - x, guide.width / dw);
        const h = Math.min(1 - y, guide.height / dh);
        return { x, y, width: w, height: h };
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
        document.getElementById('img-original').src = imageUrl;
        document.getElementById('img-corrected').src = imageUrl; 
        document.getElementById('img-corrected').style.filter = "contrast(1.1) brightness(1.05)";
    }
});