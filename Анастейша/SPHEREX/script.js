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
const solarRange = document.getElementById('solarRange');
const sunThumb = document.getElementById('sunThumb');
const frameNumber = document.getElementById('frameNumber');

solarRange.addEventListener('input', (e) => {
    const val = e.target.value; // від 1 до 100
    const percent = (val - 1) / 99; // від 0 до 1
    
    // Рухаємо сонечко по горизонталі
    const leftPos = 5 + (percent * 90);
    
    // Математичне зміщення по дузі, щоб сонечко йшло точно по твоїй намальованій траєкторії
    const verticalOffset = Math.sin(percent * Math.PI) * -15; 
    
    sunThumb.style.left = `${leftPos}%`;
    sunThumb.style.top = `calc(50% + ${verticalOffset}px)`;
    
    // Оновлюємо лічильник кадрів від 1 до 100
    frameNumber.textContent = val;
});