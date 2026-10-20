/**
 * DIGITALSCALE AI - GESTOR DE ALMACENAMIENTO DE SESION (storage.js)
 * Transfiere los resultados de la inferencia y las imagenes entre el escaner y la pantalla de resultados.
 */

const StorageUtil = {
  PREDICTION_KEY: 'digitalscale_last_prediction',

  /**
   * Guarda el resultado completo de la prediccion en sessionStorage.
   * @param {Object} predictionData - Datos devueltos por el backend + imagen original en Base64.
   */
  savePrediction(predictionData) {
    try {
      sessionStorage.setItem(this.PREDICTION_KEY, JSON.stringify(predictionData));
    } catch (e) {
      console.error('[ERROR] Error al guardar en sessionStorage:', e);
    }
  },

  /**
   * Obtiene la ultima prediccion almacenada.
   * @returns {Object|null}
   */
  getPrediction() {
    try {
      const data = sessionStorage.getItem(this.PREDICTION_KEY);
      return data ? JSON.parse(data) : null;
    } catch (e) {
      console.error('[ERROR] Error al leer de sessionStorage:', e);
      return null;
    }
  },

  /**
   * Elimina los datos de la prediccion actual.
   */
  clearPrediction() {
    try {
      sessionStorage.removeItem(this.PREDICTION_KEY);
    } catch (e) {
      // Ignorar
    }
  }
};

window.StorageUtil = StorageUtil;
