/**
 * DIGITALSCALE AI - SERVICIO DE API (api.service.js)
 * Comunicacion HTTP con el backend FastAPI (/api/health y /api/predict-base64).
 */

const ApiService = {
  // Base URL adaptable: detecta si corre bajo http://localhost o fallback
  BASE_URL: (window.location.protocol.startsWith('http')) ? window.location.origin : 'http://localhost:8000',

  /**
   * Verifica la disponibilidad del backend y el estado del dispositivo de computo.
   * @returns {Promise<{available: boolean, device?: string, gpu_available?: boolean, gpu_name?: string}>}
   */
  async checkHealth() {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 6000);

      const response = await fetch(`${this.BASE_URL}/api/health`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        return { available: false };
      }

      const data = await response.json();
      const isOnline = data.status === 'online' || data.status === 'ok';

      return {
        available: isOnline,
        device: data.device || 'CPU',
        gpu_name: data.gpu_name || '',
        gpu_available: Boolean(data.cuda_available || data.gpu_available)
      };
    } catch (error) {
      console.warn('[WARN] No se pudo conectar con /api/health:', error.message);
      return { available: false };
    }
  },

  /**
   * Envia un fotograma en Base64 para inferencia de IMC por vision computacional.
   * @param {string} base64Image - Imagen codificada en Base64 (data:image/jpeg;base64,...).
   * @returns {Promise<{
   *   success: boolean,
   *   bmi?: number,
   *   category?: string,
   *   category_code?: string,
   *   category_color?: string,
   *   posture_cluster?: number,
   *   is_valid_pose?: boolean,
   *   cropped_torso_base64?: string,
   *   estimated_weight_kg?: number,
   *   height_m?: number,
   *   message?: string
   * }>}
   */
  async predictBMI(base64Input) {
    try {
      const payload = Array.isArray(base64Input)
        ? { images_base64: base64Input }
        : { image_base64: base64Input };

      const response = await fetch(`${this.BASE_URL}/api/predict-base64`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Error en el servidor (${response.status})`);
      }

      const data = await response.json();

      if (!data.success) {
        return {
          success: false,
          message: data.message || 'La persona no se encuentra encuadrada o erguida.'
        };
      }

      return {
        success: true,
        bmi: data.bmi,
        category: data.category,
        category_code: data.category_code,
        category_color: data.category_color,
        posture_cluster: data.posture_cluster,
        is_valid_pose: data.is_valid_pose,
        cropped_torso_base64: data.cropped_torso_base64,
        estimated_weight_kg: data.estimated_weight_kg,
        height_m: data.height_m,
        message: data.message
      };
    } catch (error) {
      console.error('[ERROR] Error al procesar inferencia de IMC:', error);
      return {
        success: false,
        message: error.message || 'Error de conexion con el motor de IA.'
      };
    }
  }
};

window.ApiService = ApiService;
