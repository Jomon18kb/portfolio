// Fade in animation using Intersection Observer
document.addEventListener('DOMContentLoaded', () => {
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.15
    };

    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target); // Stop observing once it's visible
            }
        });
    }, observerOptions);

    const fadeElements = document.querySelectorAll('.fade-in');
    fadeElements.forEach(element => {
        observer.observe(element);
    });

    // Subtly parallax the glow orbs on mousemove in hero section
    const heroSection = document.querySelector('.hero');
    const orbs = document.querySelectorAll('.glow-orb');

    if (heroSection && orbs.length > 0) {
        heroSection.addEventListener('mousemove', (e) => {
            const x = (window.innerWidth / 2 - e.pageX) / 25;
            const y = (window.innerHeight / 2 - e.pageY) / 25;

            orbs[0].style.transform = `translate(${x}px, ${y}px)`;
            if (orbs[1]) {
                orbs[1].style.transform = `translate(${-x}px, ${-y}px) scale(0.9)`;
            }
        });

        // Reset transform on mouse leave
        heroSection.addEventListener('mouseleave', () => {
            orbs[0].style.transform = 'translate(0, 0)';
            if (orbs[1]) {
                orbs[1].style.transform = 'translate(0, 0) scale(0.9)';
            }
        });
    }

    // --- Background Particle Animation ---
    const canvas = document.getElementById('background-canvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        let width, height;
        let particles = [];

        // Configuration
        const particleCount = 60; // Adjust for density
        const colors = ['#58a6ff', '#bc8cff', '#e6edf3', '#8b949e']; // Theme colors

        function resizeCanvas() {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        }

        class Particle {
            constructor() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.vx = (Math.random() - 0.5) * 0.5; // Very slow horizontal movement
                this.vy = (Math.random() - 0.5) * 0.5; // Very slow vertical movement
                this.radius = Math.random() * 2 + 1; // 1 to 3px radius
                this.color = colors[Math.floor(Math.random() * colors.length)];
                this.opacity = Math.random() * 0.5 + 0.1;
                this.pulseSpeed = Math.random() * 0.02 + 0.005;
                this.pulseOffset = Math.random() * Math.PI * 2;
            }

            update(time) {
                this.x += this.vx;
                this.y += this.vy;

                // Wrap around edges seamlessly
                if (this.x < 0) this.x = width;
                if (this.x > width) this.x = 0;
                if (this.y < 0) this.y = height;
                if (this.y > height) this.y = 0;

                // Pulsate opacity slightly
                this.currentOpacity = this.opacity + Math.sin(time * this.pulseSpeed + this.pulseOffset) * 0.1;
                if (this.currentOpacity < 0) this.currentOpacity = 0;
            }

            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fillStyle = this.color;
                ctx.globalAlpha = this.currentOpacity;
                ctx.fill();
            }
        }

        function initParticles() {
            particles = [];
            for (let i = 0; i < particleCount; i++) {
                particles.push(new Particle());
            }
        }

        // Draw connecting lines between close particles
        function drawConnections() {
            ctx.lineWidth = 0.5;
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const distance = Math.sqrt(dx * dx + dy * dy);

                    if (distance < 120) { // Connection threshold
                        const opacity = 1 - (distance / 120);
                        ctx.beginPath();
                        ctx.strokeStyle = `rgba(139, 148, 158, ${opacity * 0.15})`; // subtle line color
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.stroke();
                    }
                }
            }
        }

        function animate(time) {
            ctx.clearRect(0, 0, width, height);
            ctx.globalCompositeOperation = 'lighter'; // give it a glowing effect

            drawConnections();

            particles.forEach(p => {
                p.update(time);
                p.draw();
            });

            ctx.globalAlpha = 1; // reset alpha
            requestAnimationFrame(animate);
        }

        window.addEventListener('resize', () => {
            resizeCanvas();
            initParticles(); // Re-initialize to spread evenly on resize
        });

        resizeCanvas();
        initParticles();
        requestAnimationFrame(animate);
    }
});
