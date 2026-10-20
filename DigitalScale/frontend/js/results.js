/**
 * DIGITALSCALE AI - CONTROLADOR DE RESULTADOS CLÍNICOS (results.js)
 * Carga los datos de predicción desde sessionStorage, anima el velocímetro Stitch
 * y genera el informe de composición corporal.
 */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Obtener la última predicción guardada
  let data = StorageUtil.getPrediction();

  // Si no hay datos guardados (acceso directo a la URL), usar datos de ejemplo clínicos
  if (!data || !data.bmi) {
    console.warn('[WARN] No se encontro prediccion activa en sesion. Mostrando perfil clinico de demostracion.');
    data = {
      bmi: 23.45,
      category: 'Normal',
      category_code: 'normal',
      category_color: '#006948',
      posture_cluster: 1,
      is_valid_pose: true,
      timestamp: new Date().toISOString()
    };
  }

  // 2. Elementos del DOM
  const dom = {
    evaluationDate: document.getElementById('evaluation-date'),
    bmiNumber: document.getElementById('bmi-number'),
    whoBadge: document.getElementById('who-badge'),
    whoBadgeText: document.getElementById('who-badge-text'),
    diagnosisText: document.getElementById('diagnosis-text'),
    auditOrigImg: document.getElementById('audit-orig-img'),
    auditCropImg: document.getElementById('audit-crop-img'),
    auditPostureVal: document.getElementById('audit-posture-val'),
    auditDeviceVal: document.getElementById('audit-device-val'),
    riskStatusText: document.getElementById('risk-status-text'),
    riskAdviceText: document.getElementById('risk-advice-text'),
    btnExportPdf: document.getElementById('btn-export-pdf'),
    chipUnderweight: document.getElementById('chip-legend-underweight'),
    chipNormal: document.getElementById('chip-legend-normal'),
    chipOverweight: document.getElementById('chip-legend-overweight'),
    chipObese: document.getElementById('chip-legend-obese')
  };

  // 3. Formatear y Pintar Datos Principales
  const bmiVal = Number(data.bmi);
  if (dom.bmiNumber) {
    dom.bmiNumber.textContent = Formatters.formatBMI(bmiVal);
  }

  if (dom.evaluationDate) {
    dom.evaluationDate.textContent = Formatters.formatDateTime(data.timestamp || new Date().toISOString());
  }

  // Clasificación y Textos según la OMS
  const whoInfo = Formatters.classifyWHO(bmiVal);
  const advice = Formatters.getDiagnosticAdvice(bmiVal);

  if (dom.whoBadge && dom.whoBadgeText) {
    dom.whoBadge.className = `who-chip-status ${whoInfo.className}`;
    dom.whoBadgeText.textContent = whoInfo.label;
  }

  if (dom.diagnosisText) {
    dom.diagnosisText.textContent = advice.clinicalText;
  }

  if (dom.riskStatusText) {
    dom.riskStatusText.textContent = `Riesgo cardiovascular estimado: ${advice.riskLevel}`;
  }

  if (dom.riskAdviceText) {
    dom.riskAdviceText.textContent = advice.recommendation;
  }

  // Resaltar el chip de leyenda activo
  [dom.chipUnderweight, dom.chipNormal, dom.chipOverweight, dom.chipObese].forEach((chip) => {
    if (chip) chip.classList.remove('active-chip');
  });

  if (whoInfo.className === 'underweight' && dom.chipUnderweight) dom.chipUnderweight.classList.add('active-chip');
  else if (whoInfo.className === 'normal' && dom.chipNormal) dom.chipNormal.classList.add('active-chip');
  else if (whoInfo.className === 'overweight' && dom.chipOverweight) dom.chipOverweight.classList.add('active-chip');
  else if (whoInfo.className === 'obese' && dom.chipObese) dom.chipObese.classList.add('active-chip');

  // Auditoría Visual
  if (dom.auditOrigImg) {
    if (data.original_image_base64) {
      dom.auditOrigImg.src = data.original_image_base64;
    } else {
      dom.auditOrigImg.style.display = 'none';
      if (dom.auditOrigImg.parentElement) dom.auditOrigImg.parentElement.style.display = 'none';
    }
  }

  if (dom.auditCropImg) {
    if (data.cropped_torso_base64) {
      dom.auditCropImg.src = data.cropped_torso_base64;
    } else if (data.original_image_base64) {
      dom.auditCropImg.src = data.original_image_base64;
    } else {
      dom.auditCropImg.style.display = 'none';
      if (dom.auditCropImg.parentElement) dom.auditCropImg.parentElement.style.display = 'none';
    }
  }

  if (dom.auditPostureVal) {
    const postureStr = data.is_valid_pose ? 'Erguida (Validada)' : 'Alineación Aceptable';
    const clusterStr = data.posture_cluster !== undefined ? ` - Cluster ${data.posture_cluster}` : '';
    dom.auditPostureVal.textContent = `${postureStr}${clusterStr}`;
  }

  // Dispositivo de Inferencia (deteccion dinamica sin nombres fijos)
  if (dom.auditDeviceVal) {
    if (data.device_name) {
      dom.auditDeviceVal.textContent = data.device_name;
    } else if (typeof ApiService !== 'undefined' && typeof ApiService.checkHealth === 'function') {
      ApiService.checkHealth()
        .then((health) => {
          if (health && health.available) {
            if (health.gpu_available && health.gpu_name) {
              dom.auditDeviceVal.textContent = `${health.gpu_name} (CUDA)`;
            } else if (health.device) {
              dom.auditDeviceVal.textContent = health.device.toUpperCase();
            } else {
              dom.auditDeviceVal.textContent = 'CPU';
            }
          } else {
            dom.auditDeviceVal.textContent = 'Aceleración por Hardware / CPU';
          }
        })
        .catch(() => {
          dom.auditDeviceVal.textContent = 'Aceleración por Hardware / CPU';
        });
    } else {
      dom.auditDeviceVal.textContent = 'Aceleración por Hardware / CPU';
    }
  }

  // 4. Inicializar y Animar Velocímetro SVG de Stitch
  try {
    const gauge = new GaugeComponent();
    // Iniciar animación tras breve pausa para suavidad visual
    setTimeout(() => {
      gauge.setValue(bmiVal);
    }, 150);
  } catch (err) {
    console.error('[ERROR] Error al animar velocimetro Stitch:', err);
  }

  // 5. Botón de Exportar / Imprimir
  if (dom.btnExportPdf) {
    dom.btnExportPdf.addEventListener('click', () => {
      window.print();
    });
  }
});
