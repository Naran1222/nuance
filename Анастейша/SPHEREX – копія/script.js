document.addEventListener('DOMContentLoaded', () => {
    
    // ==========================================
    // 1. ЛОГІКА ПЕРЕХОДУ НА РОБОЧУ ЗОНУ
    // ==========================================
    const detectBtn = document.getElementById('detect-btn');
    const loader = document.getElementById('space-loader');
    const sidebar = document.getElementById('sidebar'); // Або document.querySelector('.sidebar')
    const mainViewport = document.getElementById('main-viewport');

    if (detectBtn) {
        detectBtn.addEventListener('click', function() {
            if (loader) loader.classList.add('active');
            detectBtn.style.opacity = '0.5';
            detectBtn.style.pointerEvents = 'none';

            setTimeout(() => {
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

                // ВИПРАВЛЕНО: Використовуємо правильні ID з HTML
                const valTarget = document.getElementById('val-coords'); // Перевірте, який ID у вас в HTML для координат
                if (valTarget) valTarget.textContent = "M101/RA, DEC = 210,8, 54,35"; 

                const valBand = document.getElementById('val-band');
                if (valBand) valBand.textContent = "'D2' 'D3,D4' 'all'";  

                setTimeout(() => {
                    window.dispatchEvent(new Event('resize')); 
                }, 800); 
                
            }, 2000); 
        });
    }

    // ==========================================
    // 2. ЛОГІКА ТАЙМЛАЙНУ ТА ПЛЕЄРА
    // ==========================================
    const slider = document.getElementById('orbit-slider');
    const sunThumb = document.getElementById('sun-thumb');
    const orbitPath = document.getElementById('orbit-path');
    const trackContainer = document.querySelector('.orbit-track-container');
    const frameCounter = document.getElementById('frame-counter');

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
            const max = parseInt(slider.max, 10);
            const min = parseInt(slider.min, 10);
            const currentFrame = Math.round(rawVal);

            if (frameCounter) {
                frameCounter.textContent = `Кадр: ${currentFrame} / ${max}`;
            }

            const progress = (rawVal - min) / (max - min); 
            const point = orbitPath.getPointAtLength(progress * pathLength);
            
            const containerRect = trackContainer.getBoundingClientRect();
            
            if (containerRect.width === 0) return; 

            const pixelX = (point.x / svgViewBoxWidth) * containerRect.width;
            const pixelY = (point.y / svgViewBoxHeight) * containerRect.height;

            sunThumb.style.left = `${pixelX}px`;
            sunThumb.style.top = `${pixelY}px`;

            if (currentFrame !== lastReportedFrame) {
                lastReportedFrame = currentFrame;
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
    // 3. ЛОГІКА КНОПОК "НАЗАД" ТА "ВПЕРЕД" (Без дублювання)
    // ==========================================
    const btnGoBack = document.getElementById('btn-go-back');
    const btnGoForward = document.getElementById('btn-go-forward');

    if (btnGoBack) {
        btnGoBack.addEventListener('click', () => {
            if (mainViewport) mainViewport.classList.add('hidden');
            
            if (sidebar) {
                sidebar.classList.remove('collapsed');
                sidebar.classList.add('expanded');
            }
            
            if (btnGoForward) btnGoForward.classList.remove('hidden');
        });
    }

    if (btnGoForward) {
        btnGoForward.addEventListener('click', () => {
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
    const colorToggleCheckbox = document.getElementById('color-toggle-checkbox');
    
    if (colorToggleCheckbox) {
        colorToggleCheckbox.addEventListener('change', function() {
            if (this.checked) {
                console.log("Запит на бекенд: завантажити КОЛЬОРОВУ картинку...");
            } else {
                console.log("Запит на бекенд: завантажити ЧОРНО-БІЛУ картинку...");
            }
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
                
                const selectedSpeed = btn.getAttribute('data-speed');
                console.log(`Обрана швидкість: ${selectedSpeed} fps`);
            });
        });
    }
});
