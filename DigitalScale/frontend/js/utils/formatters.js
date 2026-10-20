/**
 * DIGITALSCALE AI - FORMATEADORES Y TEXTOS CLINICOS OMS (formatters.js)
 * Traduce valores numericos a clasificaciones diagnosticas oficiales y formatos legibles.
 */

const Formatters = {
  /**
   * Formatea el valor de IMC a dos decimales.
   * @param {number|string} bmi - Valor numerico del IMC.
   * @returns {string} - Cadena con formato "XX.XX" o "--.--".
   */
  formatBMI(bmi) {
    const val = Number(bmi);
    return isNaN(val) ? '--.--' : val.toFixed(2);
  },

  /**
   * Formatea una fecha ISO o marca temporal a texto legible en espanol.
   * @param {string|Date} isoString - Fecha en formato ISO o milisegundos.
   * @returns {string} - Ejemplo: "20 de octubre de 2026, 01:24".
   */
  formatDateTime(isoString) {
    if (!isoString) return 'Fecha no disponible';
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return 'Fecha no disponible';

    const dateOptions = { day: '2-digit', month: 'long', year: 'numeric' };
    const timeOptions = { hour: '2-digit', minute: '2-digit' };

    const formattedDate = date.toLocaleDateString('es-ES', dateOptions);
    const formattedTime = date.toLocaleTimeString('es-ES', timeOptions);

    return `${formattedDate}, ${formattedTime}`;
  },

  /**
   * Obtiene la clasificacion clinica segun estandares OMS.
   * @param {number|string} bmi - Valor numerico del IMC.
   * @returns {{ className: string, label: string, code: string, color: string }}
   */
  classifyWHO(bmi) {
    const val = Number(bmi);

    if (val < 18.5) {
      return {
        className: 'underweight',
        code: 'underweight',
        label: 'BAJO PESO (< 18.5)',
        color: '#006398'
      };
    }

    if (val >= 18.5 && val < 25.0) {
      return {
        className: 'normal',
        code: 'normal',
        label: 'PESO NORMAL (SALUDABLE)',
        color: '#006948'
      };
    }

    if (val >= 25.0 && val < 30.0) {
      return {
        className: 'overweight',
        code: 'overweight',
        label: 'SOBREPESO (25.0 - 29.9)',
        color: '#8d4b00'
      };
    }

    return {
      className: 'obese',
      code: 'obese',
      label: 'OBESIDAD (>= 30.0)',
      color: '#ba1a1a'
    };
  },

  /**
   * Proporciona diagnostico textual, nivel de riesgo y recomendacion segun el IMC.
   * @param {number|string} bmi - Valor numerico del IMC.
   * @returns {{ clinicalText: string, riskLevel: string, recommendation: string }}
   */
  getDiagnosticAdvice(bmi) {
    const val = Number(bmi);

    if (val < 18.5) {
      return {
        clinicalText: 'Tu Indice de Masa Corporal se situa por debajo del rango estandar recomendado. Se aconseja una revision nutricional para asegurar un aporte calorico y proteico adecuado.',
        riskLevel: 'Moderado (Deficit Nutricional)',
        recommendation: 'Consultar a un profesional de nutricion para disenar un plan hipercalorico balanceado y descartar deficiencias de micronutrientes.'
      };
    }

    if (val >= 18.5 && val < 25.0) {
      return {
        clinicalText: 'Tu peso corporal se encuentra en el rango saludable y equilibrado recomendado por la OMS para tu complexion fisica.',
        riskLevel: 'Bajo',
        recommendation: 'Mantener un patron alimenticio balanceado rico en vegetales y fuentes proteicas magras, complementado con 150 minutos semanales de actividad fisica aerobica moderada.'
      };
    }

    if (val >= 25.0 && val < 30.0) {
      return {
        clinicalText: 'Tu Indice de Masa Corporal supera ligeramente el rango normopeso. Adoptar pautas de ejercicio aerobico regular y moderar azucares refinados favorece el retorno al rango optimo.',
        riskLevel: 'Incremento Leve',
        recommendation: 'Incorporar entrenamiento de fuerza dos veces por semana, reducir el consumo de alimentos ultraprocesados y mantener una hidratacion constante.'
      };
    }

    return {
      clinicalText: 'Tu Indice de Masa Corporal se encuentra en el rango de obesidad clinica segun los criterios de la OMS. Se recomienda consultar con un medico especialista para un plan personalizado.',
      riskLevel: 'Elevado',
      recommendation: 'Programar una valoracion medica integral (perfil lipidico y glucemico) y establecer un programa supervisado de acondicionamiento fisico y reeducacion alimentaria.'
    };
  },

  /**
   * Formatea la estatura en metros.
   * @param {number|string} meters - Altura en metros.
   * @returns {string} - Ejemplo: "1.74 m".
   */
  formatHeight(meters) {
    const val = Number(meters);
    return isNaN(val) ? '-- m' : `${val.toFixed(2)} m`;
  },

  /**
   * Formatea el peso corporal en kilogramos.
   * @param {number|string} kg - Peso en kilogramos.
   * @returns {string} - Ejemplo: "71.0 kg".
   */
  formatWeight(kg) {
    const val = Number(kg);
    return isNaN(val) ? '-- kg' : `${val.toFixed(1)} kg`;
  },

  /**
   * Metodo de compatibilidad con versiones anteriores.
   * @param {number|string} bmi
   */
  classifyBMI(bmi) {
    const who = this.classifyWHO(bmi);
    const advice = this.getDiagnosticAdvice(bmi);
    return {
      code: who.code,
      label: who.label,
      diagnosis: advice.clinicalText,
      risk: `Riesgo cardiovascular estimado: ${advice.riskLevel}`,
      color: who.color
    };
  },

  /**
   * Formatea un valor numerico a una cantidad fija de decimales.
   * @param {number|string} val
   * @param {number} decimals
   * @returns {string}
   */
  formatMetric(val, decimals = 2) {
    const num = Number(val);
    return isNaN(num) ? '--.-' : num.toFixed(decimals);
  },

  /**
   * Genera una cadena legible con la fecha actual.
   * @returns {string}
   */
  getReadableDate() {
    return this.formatDateTime(new Date().toISOString());
  }
};

window.Formatters = Formatters;
window.FormatterUtil = Formatters;
