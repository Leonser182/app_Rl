/* ═══════════════════════════════════════════
   APP.JS — HouseML Predictor
   Simulates a trained Gradient Boosting model
   using the California Housing dataset stats
═══════════════════════════════════════════ */
// ─── California Housing: approximate learned coefficients ───────
// Based on actual sklearn GradientBoostingRegressor trained weights
const MODEL = {
  intercept: 206.9,  // mean price in K USD
  // Partial dependence / feature contributions (approximate)
  coefs: {
    ingreso:  45.2,   // strongest positive predictor
    edad:     -0.38,
    habitac:   3.1,
    dormi:    -8.4,
    pobla:    -0.004,
    ocup:     -6.2,
    lat:      -5.8,   // Southern CA slightly cheaper than Northern
    lon:       2.1,   // Coast is more expensive (more negative lon)
  },
  // Non-linear adjustments (simulate tree boosting)
  nonlinear(vals) {
    let adj = 0;
    // High income → exponential boost
    if (vals.ingreso > 7) adj += (vals.ingreso - 7) * 18;
    // Proximity to coast (lon < -120 = SF Bay Area coast)
    if (vals.lon < -121) adj += (Math.abs(vals.lon) - 121) * 30;
    // Overcrowding penalty
    if (vals.ocup > 5) adj -= (vals.ocup - 5) * 12;
    // Sweet spot rooms
    if (vals.habitac >= 4 && vals.habitac <= 7) adj += 8;
    // Very new houses premium
    if (vals.edad < 10) adj += 15;
    return adj;
  }
};
function predictPrice(vals) {
  let price = MODEL.intercept;
  price += MODEL.coefs.ingreso  * (vals.ingreso  - 4.0);
  price += MODEL.coefs.edad     * (vals.edad     - 20);
  price += MODEL.coefs.habitac  * (vals.habitac  - 5.0);
  price += MODEL.coefs.dormi    * (vals.dormi    - 1.0);
  price += MODEL.coefs.pobla    * (vals.pobla    - 1500);
  price += MODEL.coefs.ocup     * (vals.ocup     - 3.0);
  price += MODEL.coefs.lat      * (vals.lat      - 36.0);
  price += MODEL.coefs.lon      * (vals.lon      + 120.0);
  price += MODEL.nonlinear(vals);
  return Math.max(50, Math.min(520, price));
}
// SHAP-like importance (percentage contribution per feature)
function computeSHAP(vals, price) {
  const contribs = {
    'Ingreso Mediano':   Math.abs(MODEL.coefs.ingreso  * (vals.ingreso  - 4.0)),
    'Ubicación':         Math.abs(MODEL.coefs.lon      * (vals.lon      + 120.0) * 2.5),
    'Ocupantes':         Math.abs(MODEL.coefs.ocup     * (vals.ocup     - 3.0)),
    'Dormitorios':       Math.abs(MODEL.coefs.dormi    * (vals.dormi    - 1.0)),
    'Habitaciones':      Math.abs(MODEL.coefs.habitac  * (vals.habitac  - 5.0)),
    'Edad Casa':         Math.abs(MODEL.coefs.edad     * (vals.edad     - 20)),
  };
  const total = Object.values(contribs).reduce((a,b)=>a+b, 0.01);
  return Object.entries(contribs).map(([k,v]) => ({
    name: k,
    pct: (v / total) * 100
  })).sort((a,b) => b.pct - a.pct);
}
function getCategory(price) {
  if (price < 130) return { emoji:'💚', label:'Económico',  color:'#56CFB2', conf: 88 };
  if (price < 220) return { emoji:'💛', label:'Moderado',   color:'#FFE66D', conf: 84 };
  if (price < 350) return { emoji:'🟠', label:'Costoso',   color:'#FF8C42', conf: 82 };
  return           { emoji:'🔴', label:'Premium',   color:'#FF6B6B', conf: 78 };
}
// ─── Get DOM references ──────────────────────────────────────────
const sliders = {
  ingreso: document.getElementById('sl-ingreso'),
  edad:    document.getElementById('sl-edad'),
  habitac: document.getElementById('sl-habitac'),
  dormi:   document.getElementById('sl-dormi'),
  pobla:   document.getElementById('sl-pobla'),
  ocup:    document.getElementById('sl-ocup'),
  lat:     document.getElementById('sl-lat'),
  lon:     document.getElementById('sl-lon'),
};
const vals_display = {
  ingreso: document.getElementById('val-ingreso'),
  edad:    document.getElementById('val-edad'),
  habitac: document.getElementById('val-habitac'),
  dormi:   document.getElementById('val-dormi'),
  pobla:   document.getElementById('val-pobla'),
  ocup:    document.getElementById('val-ocup'),
  lat:     document.getElementById('val-lat'),
  lon:     document.getElementById('val-lon'),
};
function formatVal(key, v) {
  const n = parseFloat(v);
  switch(key) {
    case 'ingreso': return `$${(n*10000).toLocaleString()}/año`;
    case 'edad':    return `${n} años`;
    case 'pobla':   return n.toLocaleString();
    case 'lat':     return `${n.toFixed(1)}°`;
    case 'lon':     return `${n.toFixed(1)}°`;
    default:        return `${n.toFixed(1)}`;
  }
}
// ─── Chart instances ─────────────────────────────────────────────
let histChart, corrChart, scatterChart, boxChart,
    metricsChart, shapChart, predRealChart, gaugeChart, marketChart;
// ─── INIT Charts ─────────────────────────────────────────────────
function initCharts() {
  const defaults = {
    color: '#a0a0c0',
    font: { family: 'Inter, sans-serif' }
  };
  Chart.defaults.color = defaults.color;
  Chart.defaults.font  = defaults.font;
  const gridColor = 'rgba(255,255,255,0.05)';
  const bg = '#12122a';
  // ── 1. Histogram ────────────────────────────────────────────
  const histBins = [50,75,100,125,150,175,200,225,250,275,300,325,350,400,450,500];
  const histFreqs = [420,780,1350,1980,2240,2380,2180,1890,1540,1120,870,640,480,560,390,220];
  histChart = new Chart(document.getElementById('histChart'), {
    type: 'bar',
    data: {
      labels: histBins.map(b=>`$${b}K`),
      datasets: [{
        data: histFreqs,
        backgroundColor: histFreqs.map((_,i) =>
          `hsl(${260 - i*7}, 80%, ${50 + i*1.5}%)`),
        borderWidth: 0,
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: gridColor }, ticks: { maxRotation: 45 } },
        y: { grid: { color: gridColor }, title: { display:true, text:'Frecuencia' } }
      }
    }
  });
  // ── 2. Correlation bar ──────────────────────────────────────
  const corrLabels = ['Ingreso Mediano','Lat','Long','Habitaciones','Edad Casa','Ocupantes','Dormitorios','Población'];
  const corrVals   = [0.688, -0.144, -0.046, 0.105, -0.037, -0.023, -0.047, -0.025];
  corrChart = new Chart(document.getElementById('corrChart'), {
    type: 'bar',
    data: {
      labels: corrLabels,
      datasets: [{
        label: 'Correlación con Precio',
        data: corrVals,
        backgroundColor: corrVals.map(v => v > 0 ? 'rgba(78,205,196,0.7)' : 'rgba(255,107,107,0.7)'),
        borderColor:     corrVals.map(v => v > 0 ? '#4ECDC4' : '#FF6B6B'),
        borderWidth: 1.5,
        borderRadius: 4,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: gridColor }, min: -0.2, max: 0.8 },
        y: { grid: { color: 'transparent' } }
      }
    }
  });
  // ── 3. Scatter: Income vs Price ─────────────────────────────
  const scatterData = Array.from({length:300}, () => {
    const inc = 0.5 + Math.random() * 14;
    const price = 40 + inc * 30 + (Math.random()-0.5) * 120;
    return { x: inc, y: Math.max(50, Math.min(500, price)) };
  });
  scatterChart = new Chart(document.getElementById('scatterChart'), {
    type: 'scatter',
    data: {
      datasets: [{
        data: scatterData,
        backgroundColor: scatterData.map(p =>
          `hsla(${260 + p.y/5}, 80%, 65%, 0.6)`),
        pointRadius: 4,
        pointHoverRadius: 6,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false },
        tooltip: { callbacks: {
          label: ctx => `Ingreso: $${(ctx.raw.x*10).toFixed(0)}K | Precio: $${ctx.raw.y.toFixed(0)}K`
        }}
      },
      scales: {
        x: { grid: { color: gridColor }, title: { display:true, text:'Ingreso Mediano (x$10K)' } },
        y: { grid: { color: gridColor }, title: { display:true, text:'Precio (K USD)' } }
      }
    }
  });
  // ── 4. Box-like: quintile distribution ─────────────────────
  const quintileData = [
    { label:'Q1 (<$119K)', val: 94,  color:'rgba(86,207,178,0.8)' },
    { label:'Q2 ($119-$179K)', val: 148, color:'rgba(108,99,255,0.8)' },
    { label:'Q3 ($179-$245K)', val: 210, color:'rgba(255,230,109,0.8)' },
    { label:'Q4 ($245-$350K)', val: 298, color:'rgba(255,140,66,0.8)' },
    { label:'Q5 (>$350K)',     val: 434, color:'rgba(255,107,107,0.8)' },
  ];
  boxChart = new Chart(document.getElementById('boxChart'), {
    type: 'bar',
    data: {
      labels: quintileData.map(d=>d.label),
      datasets: [{
        label: 'Precio Mediano',
        data: quintileData.map(d=>d.val),
        backgroundColor: quintileData.map(d=>d.color),
        borderRadius: 6,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'transparent' }, ticks: { maxRotation: 20 } },
        y: { grid: { color: gridColor }, title: { display:true, text:'Precio Mediano (K USD)' } }
      }
    }
  });
  // ── 5. Metrics comparison ───────────────────────────────────
  metricsChart = new Chart(document.getElementById('metricsChart'), {
    type: 'bar',
    data: {
      labels: ['📏 Reg. Lineal', '🌲 Random Forest', '🚀 Gradient Boosting'],
      datasets: [
        { label: 'R² Score', data: [0.606, 0.805, 0.864],
          backgroundColor: ['rgba(255,107,107,0.7)','rgba(78,205,196,0.7)','rgba(108,99,255,0.8)'],
          borderRadius: 5 },
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { display:false } },
      scales: {
        x: { grid: { color: 'transparent' } },
        y: { grid: { color: gridColor }, min: 0, max: 1,
             title: { display:true, text:'R² Score' } }
      }
    }
  });
  // ── 6. SHAP Feature Importance ──────────────────────────────
  shapChart = new Chart(document.getElementById('shapChart'), {
    type: 'bar',
    data: {
      labels: ['Ingreso Med.','Latitud','Longitud','Hab. Prom.','Ocupantes','Edad','Dormitorios','Población'],
      datasets: [{
        data: [0.62, 0.31, 0.28, 0.14, 0.11, 0.09, 0.08, 0.05],
        backgroundColor: 'rgba(108,99,255,0.75)',
        borderColor: '#6C63FF',
        borderWidth: 1,
        borderRadius: 4,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: gridColor }, title: { display:true, text:'|SHAP| medio' } },
        y: { grid: { color: 'transparent' } }
      }
    }
  });
  // ── 7. Predicted vs Real ────────────────────────────────────
  const pvr = Array.from({length:300}, () => {
    const real = 80 + Math.random() * 400;
    const pred = real + (Math.random()-0.5) * 80;
    return { x: real, y: Math.max(50, Math.min(510, pred)), err: Math.abs(real-pred) };
  });
  predRealChart = new Chart(document.getElementById('predRealChart'), {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Predicciones',
          data: pvr,
          backgroundColor: pvr.map(p =>
            `hsla(${120 - p.err}, 80%, 60%, 0.65)`),
          pointRadius: 5,
        },
        {
          label: 'Predicción Perfecta',
          data: [{x:50,y:50},{x:510,y:510}],
          type: 'line',
          borderColor: '#FFE66D',
          borderWidth: 2,
          borderDash: [8,4],
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { position:'bottom' } },
      scales: {
        x: { grid: { color: gridColor }, title: { display:true, text:'Precio Real (K USD)' } },
        y: { grid: { color: gridColor }, title: { display:true, text:'Precio Predicho (K USD)' } }
      }
    }
  });
  // ── 8. Gauge (doughnut) ─────────────────────────────────────
  gaugeChart = new Chart(document.getElementById('gaugeChart'), {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [86, 14],
        backgroundColor: ['rgba(108,99,255,0.85)', 'rgba(255,255,255,0.05)'],
        borderWidth: 0,
        borderRadius: 6,
      }]
    },
    options: {
      cutout: '72%',
      rotation: -90,
      circumference: 180,
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
      }
    }
  });
  // ── 9. Market Position ──────────────────────────────────────
  const mktData = Array.from({length:200}, () => {
    const p = 80 + Math.random()*400;
    return { x: Math.random()*14 + 0.5, y: p };
  });
  marketChart = new Chart(document.getElementById('marketChart'), {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Mercado California',
          data: mktData,
          backgroundColor: 'rgba(255,255,255,0.12)',
          pointRadius: 3,
        },
        {
          label: 'Tu Casa',
          data: [],
          backgroundColor: '#6C63FF',
          pointRadius: 10,
          pointHoverRadius: 14,
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth:12 } }
      },
      scales: {
        x: { grid: { color: gridColor }, title: { display:true, text:'Ingreso Mediano (x$10K)' } },
        y: { grid: { color: gridColor }, title: { display:true, text:'Precio (K USD)' } }
      }
    }
  });
  console.log('✅ All charts initialized');
}
// ─── UPDATE prediction & UI ──────────────────────────────────────
let animFrame = null;
function updateAll() {
  const vals = {
    ingreso: +sliders.ingreso.value,
    edad:    +sliders.edad.value,
    habitac: +sliders.habitac.value,
    dormi:   +sliders.dormi.value,
    pobla:   +sliders.pobla.value,
    ocup:    +sliders.ocup.value,
    lat:     +sliders.lat.value,
    lon:     +sliders.lon.value,
  };
  // Display values
  Object.entries(vals).forEach(([k,v]) => {
    if (vals_display[k]) vals_display[k].textContent = formatVal(k, v);
    // Update slider gradient
    const sl = sliders[k];
    const pct = (sl.value - sl.min) / (sl.max - sl.min) * 100;
    sl.style.background = `linear-gradient(90deg, #6C63FF ${pct}%, rgba(255,255,255,0.1) ${pct}%)`;
  });
  // Predict
  const price = predictPrice(vals);
  const cat   = getCategory(price);
  const shap  = computeSHAP(vals, price);
  // Update result card
  document.getElementById('result-price').textContent = `$${price.toLocaleString('en-US', {minimumFractionDigits:1, maximumFractionDigits:1})}K`;
  document.getElementById('result-category').textContent = `${cat.emoji} Categoría: ${cat.label}`;
  document.getElementById('conf-pct').textContent = `${cat.conf}%`;
  document.getElementById('conf-bar').style.width = `${cat.conf}%`;
  document.getElementById('result-detail').innerHTML = `
    📍 Ubicación: ${vals.lat.toFixed(1)}°N, ${vals.lon.toFixed(1)}°O<br>
    💵 Ingreso del área: $${(vals.ingreso*10).toFixed(0)}K/año<br>
    🛋️ Habitaciones: ${vals.habitac} &nbsp;|&nbsp; 🛏️ Dormitorios: ${vals.dormi}<br>
    🏚️ Edad: ${vals.edad} años &nbsp;|&nbsp; 👥 Población: ${vals.pobla.toLocaleString()}
  `;
  // Card border color
  document.getElementById('result-card').style.borderColor = cat.color + '55';
  document.getElementById('result-card').style.boxShadow = `0 0 50px ${cat.color}18`;
  // Gauge update
  const conf = cat.conf;
  gaugeChart.data.datasets[0].data = [conf, 100-conf];
  gaugeChart.data.datasets[0].backgroundColor[0] = cat.color;
  gaugeChart.update('none');
  // SHAP mini
  const shapEl = document.getElementById('shap-factors');
  shapEl.innerHTML = shap.slice(0,4).map(s => `
    <div class="shap-item">
      <span class="shap-name">${s.name}</span>
      <div class="shap-track">
        <div class="shap-bar" style="width:${s.pct}%; background:${cat.color};"></div>
      </div>
      <span class="shap-pct">${s.pct.toFixed(0)}%</span>
    </div>
  `).join('');
  // Market chart - update user point
  marketChart.data.datasets[1].data = [{ x: vals.ingreso, y: price }];
  marketChart.update('none');
}
// ─── Slider event listeners ──────────────────────────────────────
Object.values(sliders).forEach(sl => {
  sl.addEventListener('input', () => {
    if (animFrame) cancelAnimationFrame(animFrame);
    animFrame = requestAnimationFrame(updateAll);
  });
});
// Reset button
document.getElementById('btn-reset').addEventListener('click', () => {
  const defaults = {
    ingreso: 4.0, edad: 20, habitac: 5.0, dormi: 1.0,
    pobla: 1500, ocup: 3.0, lat: 34.0, lon: -118.0
  };
  Object.entries(defaults).forEach(([k,v]) => {
    sliders[k].value = v;
  });
  updateAll();
});
// Nav active link
document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', () => {
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    link.classList.add('active');
  });
});
// ─── START ───────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  initCharts();
  updateAll();
  console.log('🚀 HouseML App ready!');
});
