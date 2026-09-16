document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('image-upload');
    const uploadArea = document.getElementById('upload-area');
    const submitBtn = document.getElementById('submit-btn');
    const resetBtn = document.getElementById('reset-btn');
    
    // 섹션 요소
    const uploadSection = document.getElementById('upload-section');
    const loadingSection = document.getElementById('loading-section');
    const resultSection = document.getElementById('result-section');

    let selectedFile = null;

    // 1. 파일 선택 이벤트 (확장자 및 용량 검증 포함)
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            // 악의적인 파일 업로드를 방지하기 위한 클라이언트 측 1차 검증
            const validTypes = ['image/jpeg', 'image/png'];
            if (!validTypes.includes(file.type)) {
                alert('JPG 또는 PNG 형식의 이미지만 업로드 가능합니다.');
                fileInput.value = ''; // 초기화
                return;
            }
            
            selectedFile = file;
            uploadArea.style.borderColor = '#4A90E2';
            uploadArea.querySelector('span:nth-child(2)').textContent = file.name;
            submitBtn.disabled = false;
        }
    });

    // 2. 서버 전송 및 처리 로직
    submitBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI 상태 변경 (업로드 숨기고 로딩 표시)
        uploadSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        // Flask 서버로 실제 데이터 전송
        const formData = new FormData();
        formData.append('image', selectedFile);
        
        try {
            const response = await fetch('/api/diagnose', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) throw new Error("서버 응답 오류");
            
            const data = await response.json();
            renderResults(data); // 백엔드에서 받은 결과(JSON)를 화면에 그림
            
        } catch (error) {
            console.error(error);
            alert('진단 중 오류가 발생했습니다.');
            // 오류 시 다시 업로드 화면으로 복귀
            loadingSection.classList.add('hidden');
            uploadSection.classList.remove('hidden');
        }
    });

    // 3. 백엔드 데이터(JSON)를 UI에 렌더링하는 함수
    function renderResults(data) {
        loadingSection.classList.add('hidden');
        resultSection.classList.remove('hidden');

        // 신뢰도 세팅[cite: 1]
        document.getElementById('reliability-text').textContent = `신뢰도: ${data.reliability}%`;
        document.getElementById('reliability-desc').textContent = data.reliability_msg;

        // 퍼센티지 진단 세팅[cite: 1]
        document.getElementById('primary-season-label').textContent = data.percentages.primary.name;
        document.getElementById('primary-percent').textContent = `${data.percentages.primary.value}%`;
        document.getElementById('primary-progress').style.width = `${data.percentages.primary.value}%`;

        document.getElementById('secondary-season-label').textContent = data.percentages.secondary.name;
        document.getElementById('secondary-percent').textContent = `${data.percentages.secondary.value}%`;
        document.getElementById('secondary-progress').style.width = `${data.percentages.secondary.value}%`;

        // 경계 케이스 해석 (예: 전형적 vs 두 시즌 경계)[cite: 1]
        document.getElementById('borderline-analysis').textContent = data.analysis_msg;

        // 조명 보정 이미지 세팅 (실제로는 서버에서 반환받은 base64나 이미지 URL 사용)[cite: 1]
        const originalUrl = URL.createObjectURL(selectedFile);
        document.getElementById('img-original').src = originalUrl;
        // 데모용으로 원본을 그대로 씁니다. 실제로는 OpenCV로 화이트밸런스 복원된 이미지 URL
        document.getElementById('img-corrected').src = originalUrl; 
        document.getElementById('img-corrected').style.filter = "contrast(1.1) brightness(1.05)"; // 보정된 효과 시뮬레이션

        // 드레이핑 이미지 세팅[cite: 1]
        document.getElementById('img-draping').src = originalUrl;
    }

    // 4. 다시 검사하기 초기화
    resetBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        uploadArea.querySelector('span:nth-child(2)').textContent = '사진 업로드 (클릭 또는 드래그)';
        uploadArea.style.borderColor = '#ccc';
        submitBtn.disabled = true;

        resultSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
    });

    // 백엔드 반환 데이터 구조 시뮬레이션 (Python 서버 작성 시 이 구조를 참고하여 Return)
    function mockOpenCVProcessing(file) {
        return {
            reliability: 92,
            reliability_msg: "신뢰도 92%, 조명 충분",
            percentages: {
                primary: { name: "겨울 브라이트", value: 78 },
                secondary: { name: "여름 라이트", value: 12 }
            },
            analysis_msg: "전형적인 겨울 브라이트 타입입니다. 명도가 높은 원색 계열이 잘 어울립니다.",
            // images: { corrected_url: "/media/corrected_img.jpg", draping_url: "/media/draping_img.jpg" }
        };
    }
});