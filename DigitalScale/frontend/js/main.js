/**
 * DIGITALSCALE AI - CONTROLADOR PRINCIPAL DE ESCANER (main.js)
 * Orquesta la captura WebRTC, MediaPipe Pose en vivo y la llamada al backend.
 */

document.addEventListener('DOMContentLoaded', async () => {
  // 1. Elementos del DOM
  const dom = {
    videoStream: document.getElementById('video-stream'),
    hudCanvas: document.getElementById('hud-canvas'),
    captureFlash: document.getElementById('capture-flash'),
    countdownOverlay: document.getElementById('countdown-overlay'),
    countdownNumber: document.getElementById('countdown-number'),
    countdownProgress: document.getElementById('countdown-progress'),
    hudPill: document.getElementById('hud-pill'),
    hudMessage: document.getElementById('hud-message'),
    btnManualCapture: document.getElementById('btn-manual-capture'),
    btnSwitchCamera: document.getElementById('btn-switch-camera'),
    telemetryBadge: document.getElementById('telemetry-badge'),
    telemetryText: document.getElementById('telemetry-text'),
    loadingModal: document.getElementById('loading-modal'),
    navBtnResults: document.getElementById('nav-btn-results')
  };

  // Si ya existia un resultado en memoria, habilitar la pestana de resultados
  const existingPrediction = StorageUtil.getPrediction();
  if (existingPrediction && existingPrediction.bmi) {
    if (dom.navBtnResults) {
      dom.navBtnResults.classList.remove('disabled');
    }
  }

  // 2. Instanciacion de Servicios y Componentes
  const cameraService = new CameraService(dom.videoStream);
  const hudComponent = new HudComponent(dom.hudCanvas, dom.videoStream, {
    countdownOverlay: dom.countdownOverlay,
    countdownNumber: dom.countdownNumber,
    countdownProgress: dom.countdownProgress,
    hudPill: dom.hudPill,
    hudMessage: dom.hudMessage,
    captureFlash: dom.captureFlash
  });

  let isCapturing = false;
  let animationFrameId = null;
  let isProcessingFrame = false;
  let lastFrameTime = 0;
  const FRAME_INTERVAL_MS = 40; // ~25 FPS para equilibrio entre fluidez y consumo de CPU

  // 3. Monitoreo de Conexion del Backend (/api/health)
  let consecutiveHealthFailures = 0;

  async function pollBackendHealth() {
    const health = await ApiService.checkHealth();
    if (health.available) {
      consecutiveHealthFailures = 0;
      dom.telemetryBadge.className = 'telemetry-badge online';
      const gpuLabel = health.gpu_available ? 'GPU Online' : 'Online (CPU)';
      dom.telemetryText.textContent = gpuLabel;
      dom.telemetryBadge.title = `Servidor activo en: ${health.device} ${health.gpu_name || ''}`;
    } else {
      consecutiveHealthFailures += 1;
      // Tolerar 1 fallo transitorio (ej. inicio de camara o carga de frames) antes de marcar offline
      if (consecutiveHealthFailures >= 2) {
        dom.telemetryBadge.className = 'telemetry-badge offline';
        dom.telemetryText.textContent = 'Servidor Desconectado';
        dom.telemetryBadge.title = 'Inicie el backend ejecutando iniciar_servidor.bat';
      }
    }
  }

  pollBackendHealth();
  setInterval(pollBackendHealth, 8000);

  // 4. Sistema de Notificaciones Toast Flotantes no bloqueantes
  function showToast(message, type = 'warning', title = null, duration = 4500) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast-item toast-${type}`;

    let iconName = 'warning';
    let defaultTitle = 'Aviso de IA';
    if (type === 'error') {
      iconName = 'error';
      defaultTitle = 'Error';
    } else if (type === 'info') {
      iconName = 'info';
      defaultTitle = 'Información';
    }

    toast.innerHTML = `
      <span class="material-symbols-outlined toast-icon">${iconName}</span>
      <div class="toast-content">
        <div class="toast-title">${title || defaultTitle}</div>
        <div class="toast-message">${message}</div>
      </div>
      <button class="toast-close-btn" type="button" aria-label="Cerrar notificación">
        <span class="material-symbols-outlined" style="font-size: 18px;">close</span>
      </button>
    `;

    const closeBtn = toast.querySelector('.toast-close-btn');
    const dismiss = () => {
      toast.classList.add('fade-out');
      setTimeout(() => {
        if (toast.parentElement) toast.parentElement.removeChild(toast);
      }, 300);
    };

    if (closeBtn) {
      closeBtn.addEventListener('click', dismiss);
    }

    container.appendChild(toast);

    if (duration > 0) {
      setTimeout(dismiss, duration);
    }
  }

  // 5. Iniciar Flujo de Video y MediaPipe Pose en Vivo
  function stopVisionLoop() {
    if (animationFrameId !== null) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
  }

  function resumeVisionLoop() {
    if (animationFrameId === null) {
      lastFrameTime = performance.now();
      animationFrameId = requestAnimationFrame(runVisionLoop);
    }
  }

  async function runVisionLoop(timestamp) {
    if (!isCapturing && (timestamp - lastFrameTime >= FRAME_INTERVAL_MS)) {
      if (!isProcessingFrame) {
        isProcessingFrame = true;
        try {
          await hudComponent.processVideoFrame();
        } catch (e) {
          console.warn('[WARN] Error en loop de vision:', e);
        } finally {
          isProcessingFrame = false;
        }
        lastFrameTime = timestamp;
      }
    }
    if (!isCapturing) {
      animationFrameId = requestAnimationFrame(runVisionLoop);
    } else {
      animationFrameId = null;
    }
  }

  async function startVisionPipeline() {
    hudComponent.updatePill('idle', 'Iniciando cámara y cargando MediaPipe...');

    const camResult = await cameraService.start();
    if (!camResult.success) {
      hudComponent.updatePill('alert', camResult.error || 'Cámara no accesible');
      return;
    }

    // Inicializar MediaPipe Pose Web esperando su inicialización completa
    hudComponent.updatePill('idle', 'Descargando modelo de postura MediaPipe...');
    const mpReady = await hudComponent.initMediaPipe();
    if (mpReady) {
      hudComponent.updatePill('idle', 'Ubíquese frente a la cámara enfocando su torso');
    } else {
      hudComponent.updatePill('alert', 'MediaPipe Web no disponible. Usando silueta.');
    }

    // Redimensionar canvas a las dimensiones del visor activo
    hudComponent.resizeCanvas();

    // Iniciar bucle de procesamiento optimizado con requestAnimationFrame
    stopVisionLoop();
    animationFrameId = requestAnimationFrame(runVisionLoop);
  }

  // 6. Configurar Disparo al Finalizar la Cuenta Regresiva de 3s
  hudComponent.setOnCapture(async () => {
    await executeCaptureAndPredict();
  });

  // 7. Proceso de Captura y Llamada a la IA
  async function executeCaptureAndPredict() {
    if (isCapturing) return;
    isCapturing = true;
    hudComponent.cancelCountdown();

    try {
      // Capturar fotograma de alta calidad
      const snapshotB64 = cameraService.captureSnapshot();

      // Mostrar modal de procesamiento
      if (dom.loadingModal) {
        dom.loadingModal.classList.add('active');
      }

      // Enviar al backend FastAPI
      const result = await ApiService.predictBMI(snapshotB64);

      if (dom.loadingModal) {
        dom.loadingModal.classList.remove('active');
      }

      if (result.success && result.bmi) {
        // Guardar resultado y capturas en sessionStorage
        StorageUtil.savePrediction({
          ...result,
          original_image_base64: snapshotB64,
          timestamp: new Date().toISOString()
        });

        // Detener la camara y el bucle visual antes de navegar
        stopVisionLoop();
        cameraService.stop();

        // Redirigir a la pantalla de resultados clinicos
        window.location.href = 'pages/results.html';
      } else {
        const errorMsg = result.message || result.posture_status || 'No se detectó un torso erguido en el encuadre.';
        showToast(errorMsg, 'warning', 'Aviso de Postura y Distancia');
        hudComponent.updatePill('alert', errorMsg);
        hudComponent.cancelCountdown();
        hudComponent.stableFramesCount = 0;
      }
    } catch (e) {
      if (dom.loadingModal) {
        dom.loadingModal.classList.remove('active');
      }
      showToast(`Error al procesar: ${e.message}`, 'error', 'Error de Comunicación');
      hudComponent.updatePill('alert', 'Error de procesamiento. Reintentando...');
      hudComponent.cancelCountdown();
      hudComponent.stableFramesCount = 0;
    } finally {
      isCapturing = false;
      isProcessingFrame = false;
      if (hudComponent) {
        hudComponent.isSendingFrame = false;
      }
      // Reanudar inmediatamente el bucle de visión en vivo
      resumeVisionLoop();
    }
  }

  // 7. Eventos de Botones
  if (dom.btnManualCapture) {
    dom.btnManualCapture.addEventListener('click', () => {
      hudComponent.triggerFlash();
      executeCaptureAndPredict();
    });
  }

  if (dom.btnSwitchCamera) {
    dom.btnSwitchCamera.addEventListener('click', async () => {
      hudComponent.updatePill('idle', 'Cambiando dispositivo de cámara...');
      const switched = await cameraService.switchCamera();
      if (switched.success) {
        hudComponent.updatePill('ready', 'Cámara cambiada. Quédese quieto');
      } else {
        hudComponent.updatePill('alert', switched.error || 'No se pudo cambiar de cámara');
      }
    });
  }

  // 8. Arrancar la vision artificial
  await startVisionPipeline();
});
