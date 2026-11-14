/*тимчасово, щоб глянути, як працює курсор*/
const viewport = document.querySelector('.viewport');
const crosshair = document.getElementById('crosshair');

viewport.addEventListener('mousemove', (e) => {
    const rect = viewport.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    crosshair.style.left = `${x}px`;
    crosshair.style.top = `${y}px`;
});

viewport.addEventListener('mouseenter', () => {
    crosshair.style.display = 'block';
});

viewport.addEventListener('mouseleave', () => {
    crosshair.style.display = 'none';
});
/*ебучая орбита*/
document.addEventListener('DOMContentLoaded', () => {
    const slider = document.getElementById('orbit-slider');
    const sunThumb = document.getElementById('sun-thumb');
    const orbitPath = document.getElementById('orbit-path');
    const trackContainer = document.querySelector('.orbit-track-container');

    // Кнопки (використовуємо ВАШІ ID з HTML)
    const playPauseBtn = document.getElementById('btn-play-pause');
    const playPauseImg = document.getElementById('play-pause-img');
    const btnPrev = document.getElementById('btn-rewind');   // Виправлено!
    const btnNext = document.getElementById('btn-forward');  // Виправлено!
    
    // Лічильник кадрів (якщо ви його додасте пізніше, він запрацює, а поки код не ламатиметься)
    const frameCounter = document.getElementById('frame-counter');

    const pathLength = orbitPath.getTotalLength();
    const svgViewBoxWidth = 1000;
    const svgViewBoxHeight = 100;

    let lastReportedFrame = 1;
    let isPlaying = false;
    let playInterval;

    // Головна функція оновлення позиції
    function updateSunPosition() {
        const rawVal = parseFloat(slider.value);
        const max = parseInt(slider.max, 10);
        const min = parseInt(slider.min, 10);
        const currentFrame = Math.round(rawVal);

        // Оновлюємо лічильник тільки якщо він існує в HTML
        if (frameCounter) {
            frameCounter.textContent = `Кадр: ${currentFrame} / ${max}`;
        }

        const progress = (rawVal - min) / (max - min); 
        const point = orbitPath.getPointAtLength(progress * pathLength);
        
        const containerRect = trackContainer.getBoundingClientRect();
        const pixelX = (point.x / svgViewBoxWidth) * containerRect.width;
        const pixelY = (point.y / svgViewBoxHeight) * containerRect.height;

        sunThumb.style.left = `${pixelX}px`;
        sunThumb.style.top = `${pixelY}px`;

        if (currentFrame !== lastReportedFrame) {
            lastReportedFrame = currentFrame;
            // Місце для відправки кадру на бекенд
        }
    }

    // ==========================================
    // ЛОГІКА СЛАЙДШОУ (Автоматичне відтворення)
    // ==========================================
    function startSlideshow() {
        isPlaying = true;
        if (playPauseImg) playPauseImg.src = 'pause.png';
        
        playInterval = setInterval(() => {
            let currentVal = parseFloat(slider.value);
            let maxVal = parseFloat(slider.max);
            
            if (currentVal >= maxVal) {
                slider.value = slider.min; // Зациклюємо слайдшоу
            } else {
                slider.value = currentVal + 1; // Крок на 1 кадр
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

    // ==========================================
    // ЛОГІКА РУЧНОГО КЕРУВАННЯ
    // ==========================================
    
    // Перетягування мишкою
    slider.addEventListener('input', () => {
        pauseSlideshow();
        updateSunPosition();
    });

    // Стрілочка НАЗАД
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

    // Стрілочка ВПЕРЕД
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

    // Ініціалізація
    window.addEventListener('resize', updateSunPosition);
    updateSunPosition();
});

// Кнопка детектора
document.getElementById('detect-btn')?.addEventListener('click', function() {
  const loader = document.getElementById('space-loader');
  if (loader) {
      loader.classList.add('active');
      setTimeout(() => loader.classList.remove('active'), 3000);
  }
});