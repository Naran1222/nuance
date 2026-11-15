document.addEventListener('DOMContentLoaded', () => {
    
    // ==========================================
    // 1. ЛОГІКА ПЕРЕХОДУ НА РОБОЧУ ЗОНУ (Наступна сторінка)
    // ==========================================
    const detectBtn = document.getElementById('detect-btn');
    const loader = document.getElementById('space-loader');
    const sidebar = document.getElementById('sidebar');
    const mainViewport = document.getElementById('main-viewport');

    // Перевіряємо, чи є кнопка на сторінці, щоб код не ламався
    if (detectBtn) {
        detectBtn.addEventListener('click', function() {
            
            // 1. Запускаємо лоадер-зірку
            if (loader) loader.classList.add('active');
            detectBtn.style.opacity = '0.5';
            detectBtn.style.pointerEvents = 'none'; // Блокуємо повторні кліки

            // 2. Імітація завантаження (2 секунди), після чого з'являється робоча зона
            setTimeout(() => {
                if (loader) loader.classList.remove('active');
                
                // Ховаємо блок із зіркою та кнопкою назавжди
                const detectContainer = document.querySelector('.detection-container');
                if (detectContainer) detectContainer.style.display = 'none';

                // Запускаємо CSS-анімацію переходу (звужуємо сайдбар)
                if (sidebar) {
                    sidebar.classList.remove('expanded');
                    sidebar.classList.add('collapsed');
                }
                
                // Показуємо головну зону з фото та таймлайном
                if (mainViewport) {
                    mainViewport.classList.remove('hidden');
                }

                // Імітуємо дані, що прийшли з бекенду
                const valTarget = document.getElementById('val-target');
                if (valTarget) valTarget.textContent = "e.g. Orion Nebula"; 

                const valBand = document.getElementById('val-band');
                if (valBand) valBand.textContent = "e.g. Near-IR";  

                // ВАЖЛИВО: Кажемо сонечку перерахувати свої координати, бо екран щойно розширився
                setTimeout(() => {
                    window.dispatchEvent(new Event('resize')); 
                }, 800); 
                
            }, 2000); // 2000 мілісекунд = 2 секунди
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

    // Якщо на сторінці немає повзунка - не виконуємо цей код далі
    if (!slider || !sunThumb || !orbitPath) return;

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
        
        // ЗАПОБІЖНИК: Якщо блок схований, не намагаємося рахувати позицію
        if (containerRect.width === 0) return; 

        const pixelX = (point.x / svgViewBoxWidth) * containerRect.width;
        const pixelY = (point.y / svgViewBoxHeight) * containerRect.height;

        sunThumb.style.left = `${pixelX}px`;
        sunThumb.style.top = `${pixelY}px`;

        if (currentFrame !== lastReportedFrame) {
            lastReportedFrame = currentFrame;
            // console.log("Кадр для бекенду:", currentFrame);
        }
    }

    function startSlideshow() {
        isPlaying = true;
        if (playPauseImg) playPauseImg.src = 'pause.png'; // Змінює картинку на паузу
        
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
            if (playPauseImg) playPauseImg.src = 'play.png'; // Змінює картинку на плей
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
});

const btnGoBack = document.getElementById('btn-go-back');
const btnGoForward = document.getElementById('btn-go-forward');
const sidebar = document.querySelector('.sidebar');
const mainViewport = document.querySelector('.main-viewport');

// Логіка натискання на "НАЗАД" (повертаємось на головну)
btnGoBack.addEventListener('click', () => {
    // Ховаємо робочу зону з сонцем
    mainViewport.classList.add('hidden');
    
    // Розгортаємо сайдбар назад на весь екран
    sidebar.classList.remove('collapsed');
    sidebar.classList.add('expanded');
    
    // Показуємо кнопку "Вперед"
    btnGoForward.classList.remove('hidden');
});

// Логіка натискання на кнопку "ВПЕРЕД" (повертаємось до сонця)
btnGoForward.addEventListener('click', () => {
    // Згортаємо сайдбар
    sidebar.classList.remove('expanded');
    sidebar.classList.add('collapsed');
    
    // Показуємо робочу зону з сонцем
    mainViewport.classList.remove('hidden');
    
    // ХОВАЄМО КНОПКУ "ВПЕРЕД", щоб вона не висіла на екрані з космосом!
    btnGoForward.classList.add('hidden');
});


// Логика нажатия на "НАЗАД" (находимся у солнца, возвращаемся на главную)
btnGoBack.addEventListener('click', () => {
    // Ховаємо робочу зону з сонцем
    mainViewport.classList.add('hidden');
    
    // Розгортаємо сайдбар назад на весь екран
    sidebar.classList.remove('collapsed');
    sidebar.classList.add('expanded');
    
    // ТЕПЕРЬ ПОКАЗЫВАЕМ НОВУЮ КНОПКУ, так как мы вернулись с рабочей зоны
    btnGoForward.classList.remove('hidden');
});

// Логика нажатия на новую кнопку "ВПЕРЕД" (возвращаемся к солнцу)
btnGoForward.addEventListener('click', () => {
    // Згортаємо сайдбар
    sidebar.classList.remove('expanded');
    sidebar.classList.add('collapsed');
    
    // Показуємо робочу зону з сонцем
    mainViewport.classList.remove('hidden');
});

// Знаходимо кнопки Так і Ні
const btnColorYes = document.getElementById('btn-color-yes');
const btnColorNo = document.getElementById('btn-color-no');
const currentFrameImg = document.getElementById('current-frame-img'); // Твоя фотографія космосу

// Логіка для кнопки ТАК (Галочка)
btnColorYes.addEventListener('click', () => {
    console.log("Запит на бекенд: завантажити КОЛЬОРОВУ картинку...");
    
    // В МАЙБУТНЬОМУ ТУТ БУДЕ КОД БЕКЕНДУ:
    // fetch('/api/get-color-image') ...
    // currentFrameImg.src = 'шлях_до_кольорової_картинки_від_бекенду.jpg';
    
    // Візуальний відгук (робимо кнопку активнішою)
    btnColorYes.style.filter = 'drop-shadow(0 0 15px rgba(0, 255, 0, 0.8))'; // Зелене світіння
    setTimeout(() => btnColorYes.style.filter = '', 300); // Повертаємо назад
});

// Логіка для кнопки НІ (Хрестик)
btnColorNo.addEventListener('click', () => {
    console.log("Запит на бекенд: завантажити ЧОРНО-БІЛУ картинку...");
    
    // В МАЙБУТНЬОМУ ТУТ БУДЕ КОД БЕКЕНДУ:
    // currentFrameImg.src = 'шлях_до_звичайної_картинки.jpg';
    
    // Візуальний відгук
    btnColorNo.style.filter = 'drop-shadow(0 0 15px rgba(255, 0, 0, 0.8))'; // Червоне світіння
    setTimeout(() => btnColorNo.style.filter = '', 300);
});

// ==========================================
// ЛОГІКА ВИБОРУ ШВИДКОСТІ
// ==========================================
const speedBtns = document.querySelectorAll('.speed-btn');

speedBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // 1. Знімаємо підсвітку (клас active) з усіх кнопок
        speedBtns.forEach(b => b.classList.remove('active'));
        
        // 2. Додаємо підсвітку тільки на ту, яку зараз натиснули
        btn.classList.add('active');
        
        // 3. Отримуємо значення кадрів (для майбутнього бекенду)
        const selectedSpeed = btn.getAttribute('data-speed');
        console.log(`Обрана швидкість: ${selectedSpeed} fps`);
    });
});
