/**
 * DIGITALSCALE AI - COMPONENTE HUD Y PUNTOS DE MOVIMIENTO (hud.component.js)
 * MediaPipe Pose Web: visualización de los 33 puntos anatómicos y control de cuenta regresiva.
 */

class HudComponent {
  constructor(canvasElement, videoElement, uiElements) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.video = videoElement;
    this.ui = uiElements;

    this.pose = null;
    this.isMediaPipeReady = false;
    this.landmarks = null;

    // Conexiones anatómicas oficiales de MediaPipe Pose
    this.POSE_CONNECTIONS = [
      // Torso superior
      [11, 12], // Hombro a hombro
      [11, 23], // Hombro izq a cadera izq
      [12, 24], // Hombro der a cadera der
      [23, 24], // Cadera a cadera
      // Brazos
      [11, 13], [13, 15], // Brazo izq
      [12, 14], [14, 16], // Brazo der
      // Rostro
      [0, 1], [1, 2], [2, 3], [3, 7],
      [0, 4], [4, 5], [5, 6], [6, 8],
      [9, 10], [11, 0], [12, 0],
      // Piernas superiores
      [23, 25], [24, 26], [25, 27], [26, 28]
    ];

    // Estados de cuenta regresiva
    this.countdownTimer = null;
    this.countdownSeconds = 3;
    this.totalSeconds = 3;
    this.isCountingDown = false;
    this.stableFramesCount = 0;
    this.onCaptureCallback = null;
    this.isSendingFrame = false;

    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());

    if (this.video) {
      this.video.addEventListener('loadedmetadata', () => this.resizeCanvas());
      this.video.addEventListener('play', () => this.resizeCanvas());
    }
  }

  resizeCanvas() {
    if (!this.canvas) return;
    const parent = this.canvas.parentElement;
    if (!parent) return;
    const rect = parent.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      this.canvas.width = rect.width;
      this.canvas.height = rect.height;
    }
  }

  /**
   * Inicializa MediaPipe Pose en el navegador con los pesos de CDN jsDelivr.
   */
  async initMediaPipe() {
    if (typeof window.Pose === 'undefined') {
      console.warn('[WARN] MediaPipe Pose no está disponible en window. Usando modo silueta estática.');
      this.isMediaPipeReady = false;
      return false;
    }

    try {
      const CDN_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404';
      this.pose = new window.Pose({
        locateFile: (file) => `${CDN_BASE}/${file}`
      });

      this.pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      this.pose.onResults((results) => this.onPoseResults(results));

      // Esperar explícitamente la descarga y compilación del WebAssembly
      await this.pose.initialize();
      this.isMediaPipeReady = true;
      console.log('[INFO] MediaPipe Pose Web inicializado con éxito para detección en vivo.');
      return true;
    } catch (e) {
      console.error('[ERROR] Error al inicializar MediaPipe Pose Web:', e);
      this.isMediaPipeReady = false;
      return false;
    }
  }

  /**
   * Envía un fotograma de video a MediaPipe Pose respetando la finalización del anterior.
   */
  async processVideoFrame() {
    if (!this.isMediaPipeReady || !this.pose || !this.video || this.video.readyState < 2) {
      this.drawFallbackSilhouette();
      return;
    }

    // Evitar llamadas concurrentes al motor WebAssembly
    if (this.isSendingFrame) return;

    this.isSendingFrame = true;
    try {
      await this.pose.send({ image: this.video });
    } catch (err) {
      console.warn('[WARN] Error al procesar frame en MediaPipe Pose:', err);
    } finally {
      this.isSendingFrame = false;
    }
  }

  /**
   * Callback invocado cada vez que MediaPipe detecta landmarks corporales.
   */
  onPoseResults(results) {
    this.landmarks = results.poseLandmarks || null;
    this.render();
    this.analyzePoseAndTrigger();
  }

  /**
   * Renderiza el esqueleto y los puntos de movimiento en tiempo real sobre el visor.
   */
  render() {
    const { ctx, canvas } = this;
    if (!ctx || !canvas || canvas.width === 0) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!this.landmarks) {
      this.drawFallbackSilhouette();
      return;
    }

    const w = canvas.width;
    const h = canvas.height;
    const lm = this.landmarks;

    ctx.save();

    // NOTA: El video tiene transform: scaleX(-1) en CSS.
    // Para que los puntos coincidan exactamente con la imagen en espejo:
    const getPoint = (idx) => {
      if (!lm[idx]) return null;
      if (lm[idx].visibility !== undefined && lm[idx].visibility < 0.25) return null;
      return {
        x: (1 - lm[idx].x) * w,
        y: lm[idx].y * h
      };
    };

    // 1. Dibujar Líneas Anatómicas del Esqueleto
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.strokeStyle = this.isCountingDown ? 'rgba(0, 105, 72, 0.9)' : 'rgba(0, 99, 152, 0.75)';

    for (const [i, j] of this.POSE_CONNECTIONS) {
      const p1 = getPoint(i);
      const p2 = getPoint(j);
      if (p1 && p2) {
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
    }

    // 2. Dibujar Puntos Clave de Movimiento (Keypoints)
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    let validPts = 0;

    for (let i = 0; i < lm.length; i++) {
      const pt = getPoint(i);
      if (!pt) continue;

      minX = Math.min(minX, pt.x);
      maxX = Math.max(maxX, pt.x);
      minY = Math.min(minY, pt.y);
      maxY = Math.max(maxY, pt.y);
      validPts++;

      // Nodos principales del torso (hombros: 11, 12 | caderas: 23, 24)
      const isTorsoNode = (i === 11 || i === 12 || i === 23 || i === 24);
      const radius = isTorsoNode ? 6.5 : 4.5;
      const borderColor = isTorsoNode ? '#006948' : '#006398';

      // Círculo exterior con contraste
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = borderColor;
      ctx.stroke();
    }

    // 3. Dibujar Bounding Box de encuadre corporal (similar a webcam_demo.py)
    if (validPts >= 6 && minX < maxX && minY < maxY) {
      const padX = (maxX - minX) * 0.08;
      const padY = (maxY - minY) * 0.05;
      const boxX = Math.max(0, minX - padX);
      const boxY = Math.max(0, minY - padY);
      const boxW = Math.min(w - boxX, (maxX - minX) + padX * 2);
      const boxH = Math.min(h - boxY, (maxY - minY) + padY * 2);

      ctx.strokeStyle = this.isCountingDown ? 'rgba(0, 105, 72, 0.85)' : 'rgba(0, 99, 152, 0.6)';
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 8]);
      ctx.strokeRect(boxX, boxY, boxW, boxH);
      ctx.setLineDash([]);
    }

    ctx.restore();
  }

  /**
   * Dibuja la silueta estática orientativa si aún no hay landmarks detectados.
   */
  drawFallbackSilhouette() {
    const { ctx, canvas } = this;
    if (!ctx || !canvas) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;

    ctx.save();
    ctx.strokeStyle = 'rgba(0, 105, 72, 0.4)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 6]);

    // Cabeza
    ctx.beginPath();
    ctx.ellipse(cx, h * 0.2, w * 0.08, h * 0.11, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Hombros y Torso
    ctx.beginPath();
    ctx.moveTo(cx - w * 0.22, h * 0.36);
    ctx.lineTo(cx + w * 0.22, h * 0.36);
    ctx.lineTo(cx + w * 0.18, h * 0.8);
    ctx.lineTo(cx - w * 0.18, h * 0.8);
    ctx.closePath();
    ctx.stroke();

    ctx.restore();
  }

  /**
   * Analiza la geometría corporal detectada para controlar la cuenta regresiva.
   */
  analyzePoseAndTrigger() {
    if (!this.landmarks) {
      this.updatePill('idle', 'Ubíquese frente a la cámara enfocando su torso');
      this.stableFramesCount = 0;
      this.cancelCountdown();
      return;
    }

    const lm = this.landmarks;
    const lSh = lm[11]; // Hombro izq
    const rSh = lm[12]; // Hombro der
    const lHip = lm[23]; // Cadera izq
    const rHip = lm[24]; // Cadera der

    // Comprobar visibilidad de hombros y caderas
    const hasShoulders = lSh && rSh && (lSh.visibility === undefined || lSh.visibility > 0.35) && (rSh.visibility === undefined || rSh.visibility > 0.35);
    const hasHips = lHip && rHip && (lHip.visibility === undefined || lHip.visibility > 0.3) && (rHip.visibility === undefined || rHip.visibility > 0.3);

    if (!hasShoulders) {
      this.updatePill('idle', 'Buscando postura... ubique sus hombros en el encuadre');
      this.stableFramesCount = 0;
      this.cancelCountdown();
      return;
    }

    // Ancho de hombros normalizado
    const shoulderWidth = Math.abs(lSh.x - rSh.x);

    // Detección de proximidad excesiva a la lente
    const isTooClose = shoulderWidth > 0.48 || lSh.x < 0.05 || rSh.x > 0.95;

    if (isTooClose) {
      this.updatePill('alert', 'Demasiado Cerca: Retroceda un paso para enfocar su torso');
      this.stableFramesCount = 0;
      this.cancelCountdown();
      return;
    }

    // Si tiene hombros y caderas en encuadre
    if (hasShoulders && hasHips) {
      this.stableFramesCount += 1;
      this.updatePill('ready', 'Postura Correcta: Manténgase erguido');

      // Tras 5 frames estables, iniciar cuenta regresiva de 3 segundos
      if (this.stableFramesCount >= 5 && !this.isCountingDown) {
        this.startCountdown();
      }
    } else {
      this.updatePill('idle', 'Por favor enfoque sus hombros y caderas');
      this.stableFramesCount = 0;
      this.cancelCountdown();
    }
  }

  updatePill(status, message) {
    if (this.ui.hudMessage) {
      this.ui.hudMessage.textContent = message;
    }
    if (this.ui.hudPill) {
      this.ui.hudPill.className = 'hud-floating-pill';
      if (status === 'alert') {
        this.ui.hudPill.classList.add('alert');
      }
    }
  }

  startCountdown() {
    if (this.isCountingDown) return;
    this.isCountingDown = true;
    this.countdownSeconds = this.totalSeconds;

    if (this.ui.countdownOverlay) {
      this.ui.countdownOverlay.classList.add('visible');
    }
    this.updateCountdownRing();

    this.countdownTimer = setInterval(() => {
      this.countdownSeconds -= 1;
      this.updateCountdownRing();

      if (this.countdownSeconds <= 0) {
        this.finishCountdown();
      }
    }, 1000);
  }

  cancelCountdown() {
    if (this.isCountingDown) {
      this.isCountingDown = false;
      if (this.countdownTimer) {
        clearInterval(this.countdownTimer);
        this.countdownTimer = null;
      }
      if (this.ui.countdownOverlay) {
        this.ui.countdownOverlay.classList.remove('visible');
      }
    }
  }

  updateCountdownRing() {
    if (this.ui.countdownNumber) {
      this.ui.countdownNumber.textContent = this.countdownSeconds;
    }
    if (this.ui.countdownProgress) {
      const circ = 314.16; // 2 * PI * 50
      const offset = circ - (this.countdownSeconds / this.totalSeconds) * circ;
      this.ui.countdownProgress.style.strokeDashoffset = offset;
    }
  }

  finishCountdown() {
    this.cancelCountdown();
    this.triggerFlash();
    if (typeof this.onCaptureCallback === 'function') {
      this.onCaptureCallback();
    }
  }

  triggerFlash() {
    if (!this.ui.captureFlash) return;
    this.ui.captureFlash.classList.add('active');
    setTimeout(() => {
      this.ui.captureFlash.classList.remove('active');
    }, 250);
  }

  setOnCapture(cb) {
    this.onCaptureCallback = cb;
  }
}

window.HudComponent = HudComponent;
