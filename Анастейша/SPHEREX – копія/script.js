document.addEventListener('DOMContentLoaded', () => {
    const API_BASE = 'http://127.0.0.1:8000';

    let currentLimit = 1;
    let totalFrames = 1;
    let epochsData = [];

    const frameImg = document.getElementById('current-frame-img');
    const colorToggleCheckbox = document.getElementById('color-toggle-checkbox');
    const valViewportData = document.getElementById('val-viewport-data');
    const slider = document.getElementById('orbit-slider');
    const frameCounter = document.getElementById('frame-counter');

    function parseTargetInput(raw) {
        const cleaned = (raw || '').trim();
        if (!cleaned || cleaned.startsWith('M101/RA')) return 'M101';
        return cleaned;
    }

    function parseBandInput(raw) {
        const cleaned = (raw || '').replace(/['"]/g, '').trim();
        if (!cleaned || cleaned.includes('D2 D3,D4 all')) return 'D3';
        return cleaned;
    }

    function loadFrameImage(frameIndex) {
        if (!frameImg) return;
        const isColor = colorToggleCheckbox ? colorToggleCheckbox.checked : true;
        const url = `${API_BASE}/api/frame/${frameIndex}?is_color=${isColor}&t=${Date.now()}`;

        if (valViewportData && epochsData[frameIndex]) {
            const ep = epochsData[frameIndex];
            valViewportData.textContent = ep.date_start === ep.date_end
                ? ep.date_start
                : `${ep.date_start} — ${ep.date_end}`;
        }

        frameImg.onerror = () => {
            if (!isColor) {
                if (colorToggleCheckbox) colorToggleCheckbox.checked = true;
                frameImg.onerror = null;
                frameImg.src = `${API_BASE}/api/frame/${frameIndex}?is_color=true&t=${Date.now()}`;
            }
        };

        frameImg.src = url;
    }

    // ==========================================
    // 1. ЛОГІКА ПЕРЕХОДУ НА РОБОЧУ ЗОНУ
    // ==========================================
    const detectBtn = document.getElementById('detect-btn');
    const loader = document.getElementById('space-loader');
    const sidebar = document.getElementById('sidebar');
    const mainViewport = document.getElementById('main-viewport');

    if (detectBtn) {
        detectBtn.addEventListener('click', async function () {
            if (loader) loader.classList.add('active');
            detectBtn.style.opacity = '0.5';
            detectBtn.style.pointerEvents = 'none';

            const targetEl = document.getElementById('val-target');
            const bandEl = document.getElementById('val-band');

            const target = parseTargetInput(targetEl ? targetEl.textContent : 'M101');
            const band = parseBandInput(bandEl ? bandEl.textContent : 'D3');

            try {
                const params = new URLSearchParams({
                    target: target,
                    band: band,
                    limit: String(currentLimit)
                });

                const minAnimationDelay = new Promise(resolve => setTimeout(resolve, 2000));
                const apiRequest = fetch(`${API_BASE}/api/get-anomaly?${params.toString()}`).then(async res => {
                    if (!res.ok) {
                        const errText = await res.text();
                        throw new Error(`HTTP ${res.status}: ${errText}`);
                    }
                    return res.json();
                });

                const [data] = await Promise.all([apiRequest, minAnimationDelay]);

                epochsData = Array.isArray(data.epochs) ? data.epochs : [];
                totalFrames = Math.max(1, data.total_frames || epochsData.length || 1);

                if (slider) {
                    slider.min = '1';
                    slider.max = String(totalFrames);
                    slider.value = '1';
                }

                if (frameCounter) {
                    frameCounter.textContent = `Кадр: 1 / ${totalFrames}`;
                }

                if (colorToggleCheckbox) {
                    colorToggleCheckbox.checked = true;
                }

                loadFrameImage(0);

                if (loader) loader.classList.remove('active');

                const detectContainer = document.querySelector('.detection-container');
                if (detectContainer) detectContainer.style.display = 'none';

                if (sidebar) {
                    sidebar.classList.remove('expanded');
                    sidebar.classList.add('collapsed');
                }

                if (mainViewport) {
                    mainViewport.classList.remove('hidden');
                }

                setTimeout(() => {
                    window.dispatchEvent(new Event('resize'));
                }, 800);

            } catch (error) {
                console.error('Помилка запиту до бекенду:', error);
                if (loader) loader.classList.remove('active');
                alert(`Не вдалося отримати дані з сервера (${API_BASE}). Перевір, чи запущений uvicorn.\n\nДеталі: ${error.message}`);
            } finally {
                detectBtn.style.opacity = '1';
                detectBtn.style.pointerEvents = 'auto';
            }
        });
    }

    // ==========================================
    // 2. ЛОГІКА ТАЙМЛАЙНУ ТА ПЛЕЄРА
    // ==========================================
    const sunThumb = document.getElementById('sun-thumb');
    const orbitPath = document.getElementById('orbit-path');
    const trackContainer = document.querySelector('.orbit-track-container');

    const playPauseBtn = document.getElementById('btn-play-pause');
    const playPauseImg = document.getElementById('play-pause-img');
    const btnPrev = document.getElementById('btn-rewind');
    const btnNext = document.getElementById('btn-forward');

    if (slider && sunThumb && orbitPath) {
        const pathLength = orbitPath.getTotalLength();
        const svgViewBoxWidth = 1000;
        const svgViewBoxHeight = 100;

        let lastReportedFrame = 1;
        let isPlaying = false;
        let playInterval;

        function updateSunPosition() {
            const rawVal = parseFloat(slider.value);
            const max = Math.max(1, parseInt(slider.max, 10));
            const min = parseInt(slider.min, 10);
            const currentFrame = Math.round(rawVal);

            if (frameCounter) {
                frameCounter.textContent = `Кадр: ${currentFrame} / ${max}`;
            }

            const progress = max === min ? 0.5 : (rawVal - min) / (max - min);
            const point = orbitPath.getPointAtLength(progress * pathLength);

            const containerRect = trackContainer.getBoundingClientRect();
            if (containerRect.width === 0) return;

            const pixelX = (point.x / svgViewBoxWidth) * containerRect.width;
            const pixelY = (point.y / svgViewBoxHeight) * containerRect.height;

            sunThumb.style.left = `${pixelX}px`;
            sunThumb.style.top = `${pixelY}px`;

            if (currentFrame !== lastReportedFrame) {
                lastReportedFrame = currentFrame;
                loadFrameImage(currentFrame - 1);
            }
        }

        function startSlideshow() {
            isPlaying = true;
            if (playPauseImg) playPauseImg.src = 'pause.png';

            playInterval = setInterval(() => {
                let currentVal = parseFloat(slider.value);
                let maxVal = parseFloat(slider.max);

                if (currentVal >= maxVal) {
                    slider.value = slider.min;
                } else {
                    slider.value = currentVal + 1;
                }
                updateSunPosition();
            }, 500);
        }

        function pauseSlideshow() {
            if (isPlaying) {
                isPlaying = false;
                if (playPauseImg) playPauseImg.src = 'play.png';
                clearInterval(playInterval);
            }
        }

        if (playPauseBtn) {
            playPauseBtn.addEventListener('click', () => {
                if (isPlaying) pauseSlideshow();
                else startSlideshow();
            });
        }

        slider.addEventListener('input', () => {
            pauseSlideshow();
            updateSunPosition();
        });

        if (btnPrev) {
            btnPrev.addEventListener('click', () => {
                pauseSlideshow();
                let currentFrame = Math.round(parseFloat(slider.value));
                if (currentFrame > parseInt(slider.min, 10)) {
                    slider.value = currentFrame - 1;
                    updateSunPosition();
                }
            });
        }

        if (btnNext) {
            btnNext.addEventListener('click', () => {
                pauseSlideshow();
                let currentFrame = Math.round(parseFloat(slider.value));
                if (currentFrame < parseInt(slider.max, 10)) {
                    slider.value = currentFrame + 1;
                    updateSunPosition();
                }
            });
        }

        window.addEventListener('resize', updateSunPosition);
        updateSunPosition();
    }

    // ==========================================
    // 3. ЛОГІКА КНОПОК "НАЗАД" ТА "ВПЕРЕД"
    // ==========================================
    const btnGoBack = document.getElementById('btn-go-back');
    const btnGoForward = document.getElementById('btn-go-forward');

    if (btnGoBack) {
        btnGoBack.addEventListener('click', () => {
            if (mainViewport) mainViewport.classList.add('hidden');

            const detectContainer = document.querySelector('.detection-container');
            if (detectContainer) detectContainer.style.display = '';

            if (sidebar) {
                sidebar.classList.remove('collapsed');
                sidebar.classList.add('expanded');
            }

            if (btnGoForward) btnGoForward.classList.remove('hidden');
        });
    }

    if (btnGoForward) {
        btnGoForward.addEventListener('click', () => {
            const detectContainer = document.querySelector('.detection-container');
            if (detectContainer) detectContainer.style.display = 'none';

            if (sidebar) {
                sidebar.classList.remove('expanded');
                sidebar.classList.add('collapsed');
            }

            if (mainViewport) mainViewport.classList.remove('hidden');

            btnGoForward.classList.add('hidden');
        });
    }

    // ==========================================
    // 4. ЛОГІКА ЧЕКБОКСА (КОЛІР)
    // ==========================================
    if (colorToggleCheckbox) {
        colorToggleCheckbox.addEventListener('change', function () {
            const currentIdx = slider ? Math.max(0, Math.round(parseFloat(slider.value)) - 1) : 0;
            loadFrameImage(currentIdx);
        });
    }

    // ==========================================
    // 5. ЛОГІКА ВИБОРУ ШВИДКОСТІ
    // ==========================================
    const speedBtns = document.querySelectorAll('.speed-btn');

    if (speedBtns.length > 0) {
        speedBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                speedBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                const selectedSpeed = parseInt(btn.getAttribute('data-speed'), 10);
                currentLimit = selectedSpeed === 30 ? 1 : selectedSpeed;
            });
        });
    }
});