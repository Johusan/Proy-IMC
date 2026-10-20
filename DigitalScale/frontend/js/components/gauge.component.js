/**
 * DIGITALSCALE AI - COMPONENTE VELOCÍMETRO STITCH (gauge.component.js)
 * Controla y anima la aguja y la etiqueta de posición sobre el semicírculo SVG de Stitch.
 */

class GaugeComponent {
  constructor(options = {}) {
    this.svgId = options.svgId || 'stitch-gauge-svg';
    this.needleLineId = options.needleLineId || 'gauge-needle-line';
    this.needleTipId = options.needleTipId || 'gauge-needle-tip';
    this.indicatorBadgeId = options.indicatorBadgeId || 'gauge-indicator-badge';
    this.badgeTextId = options.badgeTextId || 'gauge-badge-text';

    this.svg = document.getElementById(this.svgId);
    this.needleLine = document.getElementById(this.needleLineId);
    this.needleTip = document.getElementById(this.needleTipId);
    this.indicatorBadge = document.getElementById(this.indicatorBadgeId);
    this.badgeText = document.getElementById(this.badgeTextId);

    // Centro del arco SVG (cx=200, cy=195)
    this.cx = 200;
    this.cy = 195;
    this.needleLength = 135;

    // Rango métrico de IMC en el arco
    this.minBMI = 15.0;
    this.maxBMI = 38.0;

    this.currentBMI = 15.0;
    this.targetBMI = 23.45;
    this.animationId = null;
  }

  /**
   * Mapea un valor de IMC a un ángulo en radianes sobre el semicírculo SVG.
   * Ángulo PI (izquierda) = 15.0 BMI
   * Ángulo 2*PI (derecha) = 38.0 BMI
   */
  bmiToAngle(bmi) {
    const clamped = Math.max(this.minBMI, Math.min(this.maxBMI, bmi));
    const factor = (clamped - this.minBMI) / (this.maxBMI - this.minBMI);
    return Math.PI + factor * Math.PI;
  }

  /**
   * Actualiza el valor objetivo del IMC e inicia la animación fluida.
   * @param {number} targetValue
   */
  setValue(targetValue) {
    this.targetBMI = Math.max(this.minBMI, Math.min(this.maxBMI, Number(targetValue)));
    this.currentBMI = this.minBMI;
    this.animate();
  }

  animate() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }

    const step = () => {
      const diff = this.targetBMI - this.currentBMI;
      if (Math.abs(diff) < 0.05) {
        this.currentBMI = this.targetBMI;
        this.renderNeedle(this.currentBMI);
      } else {
        this.currentBMI += diff * 0.12;
        this.renderNeedle(this.currentBMI);
        this.animationId = requestAnimationFrame(step);
      }
    };

    this.animationId = requestAnimationFrame(step);
  }

  /**
   * Posiciona la aguja y el badge SVG en las coordenadas angulares correspondientes.
   */
  renderNeedle(bmi) {
    const angle = this.bmiToAngle(bmi);

    const nx = this.cx + Math.cos(angle) * this.needleLength;
    const ny = this.cy + Math.sin(angle) * this.needleLength;

    if (this.needleLine) {
      this.needleLine.setAttribute('x2', nx.toFixed(1));
      this.needleLine.setAttribute('y2', ny.toFixed(1));
    }

    if (this.needleTip) {
      this.needleTip.setAttribute('cx', nx.toFixed(1));
      this.needleTip.setAttribute('cy', ny.toFixed(1));

      // Color de la punta según el cuadrante
      let tipColor = '#006948'; // Normal
      if (bmi < 18.5) tipColor = '#006398';
      else if (bmi >= 25.0 && bmi < 30.0) tipColor = '#8d4b00';
      else if (bmi >= 30.0) tipColor = '#ba1a1a';
      this.needleTip.setAttribute('fill', tipColor);
    }

    // Posicionar el indicador flotante
    if (this.indicatorBadge) {
      const badgeX = Math.max(10, Math.min(230, nx - 80));
      const badgeY = Math.max(10, Math.min(130, ny - 50));
      this.indicatorBadge.setAttribute('transform', `translate(${badgeX.toFixed(1)}, ${badgeY.toFixed(1)})`);
    }

    if (this.badgeText) {
      this.badgeText.textContent = `Estás aquí: ${bmi.toFixed(2)}`;
    }
  }
}

window.GaugeComponent = GaugeComponent;
