/**
 * DIGITALSCALE AI - SERVICIO DE CAMARA WEBRTC (camera.service.js)
 * Manejo robusto del flujo de video, cambio de camaras y captura de fotogramas.
 */

class CameraService {
  constructor(videoElement) {
    this.video = videoElement;
    this.stream = null;
    this.currentFacingMode = 'user'; // 'user' (frontal) o 'environment' (trasera)
    this.devices = [];
    this.currentDeviceIndex = 0;
    this.isStreaming = false;
  }

  /**
   * Inicializa el flujo de la camara web con fallback elastico.
   * @returns {Promise<{success: boolean, error?: string}>}
   */
  async start() {
    this.stop();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.error('[ERROR] WebRTC getUserMedia no esta disponible en este navegador o contexto.');
      return {
        success: false,
        error: 'El navegador no soporta captura de video o la conexion no es segura (use http://localhost:8000).'
      };
    }

    try {
      // Enumerar camaras disponibles
      await this.enumerateVideoDevices();

      // Intento 1: Restricciones ideales (HD 720p)
      let constraints = {
        audio: false,
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: this.currentFacingMode
        }
      };

      if (this.devices.length > 0 && this.devices[this.currentDeviceIndex]?.deviceId) {
        constraints.video = {
          deviceId: { exact: this.devices[this.currentDeviceIndex].deviceId },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        };
      }

      try {
        this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (firstErr) {
        console.warn('[WARN] Restricciones HD fallaron, intentando fallback elastico generico { video: true }...', firstErr.message);
        // Intento 2: Fallback elastico sin restricciones estrictas
        this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }

      this.video.srcObject = this.stream;

      return new Promise((resolve) => {
        this.video.onloadedmetadata = () => {
          this.video.play()
            .then(() => {
              this.isStreaming = true;
              resolve({ success: true });
            })
            .catch((playErr) => {
              console.error('[ERROR] Error al reproducir stream de video:', playErr);
              resolve({ success: false, error: 'No se pudo reproducir el video de la camara.' });
            });
        };
      });
    } catch (err) {
      console.error('[ERROR] Error al acceder a la camara:', err);
      this.isStreaming = false;
      let errorMsg = 'No se pudo acceder a la camara.';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        errorMsg = 'Permiso denegado. Active el acceso a la camara en los permisos del navegador.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        errorMsg = 'No se encontro ninguna camara conectada al sistema.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        errorMsg = 'La camara esta siendo usada por otra aplicacion.';
      }
      return { success: false, error: errorMsg };
    }
  }

  /**
   * Detiene el flujo de la camara para liberar el hardware de captura.
   */
  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.video) {
      this.video.srcObject = null;
    }
    this.isStreaming = false;
  }

  /**
   * Enumera todas las camaras de video disponibles en el equipo.
   */
  async enumerateVideoDevices() {
    try {
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      this.devices = allDevices.filter((d) => d.kind === 'videoinput');
    } catch (e) {
      this.devices = [];
    }
  }

  /**
   * Alterna a la siguiente camara disponible.
   * @returns {Promise<{success: boolean, error?: string}>}
   */
  async switchCamera() {
    if (this.devices.length > 1) {
      this.currentDeviceIndex = (this.currentDeviceIndex + 1) % this.devices.length;
    } else {
      this.currentFacingMode = this.currentFacingMode === 'user' ? 'environment' : 'user';
    }
    return await this.start();
  }

  /**
   * Captura el fotograma actual del elemento de video como una imagen Base64 JPEG
   * en orientacion natural directa (sin volteo en espejo para alineacion anatomica con el modelo).
   * @returns {string} - Cadena data:image/jpeg;base64,...
   */
  captureSnapshot() {
    if (!this.video || !this.isStreaming) {
      throw new Error('La camara no esta activa.');
    }

    const canvas = document.createElement('canvas');
    canvas.width = this.video.videoWidth || 1280;
    canvas.height = this.video.videoHeight || 720;
    const ctx = canvas.getContext('2d');

    // Capturar orientacion natural sin invertir coordenadas
    ctx.drawImage(this.video, 0, 0, canvas.width, canvas.height);

    // Exportar a JPEG optimizado (0.75) para balance entre calidad y latencia de red
    return canvas.toDataURL('image/jpeg', 0.75);
  }
}

window.CameraService = CameraService;
