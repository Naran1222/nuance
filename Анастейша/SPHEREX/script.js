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
    const frameCounter = document.getElementById('frame-counter');
    const trackContainer = document.querySelector('.orbit-track-container');

    // Отримуємо фізичну довжину кривої SVG
    const pathLength = orbitPath.getTotalLength();
    
    // Внутрішні розміри viewBox (як ми задали в HTML: 0 0 1000 100)
    const svgViewBoxWidth = 1000;
    const svgViewBoxHeight = 100;

    function updateSunPosition() {
        const val = parseInt(slider.value, 10);
        const max = parseInt(slider.max, 10);

        // 1. Оновлюємо текст лічильника
        frameCounter.textContent = `Кадр: ${val} / ${max}`;

        // 2. Обчислюємо, на скільки відсотків пройдено шлях (від 0 до 1)
        const progress = (val - 1) / (max - 1); 

        // 3. Знаходимо координати точки на кривій
        const point = orbitPath.getPointAtLength(progress * pathLength);

        // 4. Оскільки SVG розтягується під ширину екрану (через width: 100%), 
        // нам треба перевести координати viewBox у реальні пікселі контейнера
        const containerRect = trackContainer.getBoundingClientRect();
        
        const pixelX = (point.x / svgViewBoxWidth) * containerRect.width;
        const pixelY = (point.y / svgViewBoxHeight) * containerRect.height;

        // 5. Рухаємо сонечко
        sunThumb.style.left = `${pixelX}px`;
        sunThumb.style.top = `${pixelY}px`;
    }

    // Слухаємо зміни повзунка
    slider.addEventListener('input', updateSunPosition);

    // Оновлюємо позицію при зміні розміру вікна (якщо інтерфейс адаптивний)
    window.addEventListener('resize', updateSunPosition);

    // Встановлюємо сонечко на старт при завантаженні сторінки
    updateSunPosition();
    // Находим центральную кнопку и картинку внутри нее
const playPauseBtn = document.getElementById('btn-play-pause');
const playPauseImg = document.getElementById('play-pause-img');

// Переменная состояния плеера
let isPlaying = false;

playPauseBtn.addEventListener('click', () => {
    // Меняем статус на противоположный
    isPlaying = !isPlaying;
    
    if (isPlaying) {
        // Если играет -> показываем паузу
        playPauseImg.src = 'pause.png';
    } else {
        // Если на паузе -> показываем плей
        playPauseImg.src = 'play.png';
    }
});
});
