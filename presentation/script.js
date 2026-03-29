const slides = document.querySelectorAll('.slide');
const totalSlides = slides.length;
let current = 0;
let isAnimating = false;

document.getElementById('total-num').textContent = totalSlides;

// Build dot indicators
const dotsContainer = document.getElementById('dots');
slides.forEach((_, i) => {
  const dot = document.createElement('div');
  dot.className = 'dot' + (i === 0 ? ' active' : '');
  dot.addEventListener('click', () => goTo(i));
  dotsContainer.appendChild(dot);
});

function goTo(index) {
  if (isAnimating || index === current || index < 0 || index >= totalSlides) return;
  isAnimating = true;

  const prev = slides[current];
  const next = slides[index];

  prev.classList.add('exit-left');
  next.classList.add('active');

  setTimeout(() => {
    prev.classList.remove('active', 'exit-left');
    isAnimating = false;
  }, 300);

  current = index;
  updateUI();
}

function changeSlide(dir) {
  goTo(current + dir);
}

function updateUI() {
  document.getElementById('current-num').textContent = current + 1;
  document.getElementById('prev-btn').disabled = current === 0;
  document.getElementById('next-btn').disabled = current === totalSlides - 1;

  document.querySelectorAll('.dot').forEach((dot, i) => {
    dot.classList.toggle('active', i === current);
  });
}

// Keyboard navigation
document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === ' ') {
    e.preventDefault();
    changeSlide(1);
  } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
    e.preventDefault();
    changeSlide(-1);
  } else if (e.key === 'Home') {
    goTo(0);
  } else if (e.key === 'End') {
    goTo(totalSlides - 1);
  }
});

// Touch/swipe support
let touchStartX = 0;
let touchStartY = 0;

document.addEventListener('touchstart', (e) => {
  touchStartX = e.touches[0].clientX;
  touchStartY = e.touches[0].clientY;
}, { passive: true });

document.addEventListener('touchend', (e) => {
  const dx = e.changedTouches[0].clientX - touchStartX;
  const dy = e.changedTouches[0].clientY - touchStartY;
  if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 50) {
    changeSlide(dx < 0 ? 1 : -1);
  }
}, { passive: true });

// Image tabs
document.querySelectorAll('.tab-btns').forEach((btnGroup) => {
  btnGroup.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.target;
      const panel = btnGroup.parentElement;

      btnGroup.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      panel.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      panel.querySelector('#' + target).classList.add('active');
    });
  });
});

// Init
updateUI();
