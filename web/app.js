// Dubai Property Price Estimator: browser app.
// The gradient-boosting model was trained in Python (see analysis/) and exported as plain
// JSON: every tree is a list of nodes. predictLogPpsf() walks each tree and adds up the
// leaf values, exactly like scikit-learn does.

const SQM_TO_SQFT = 10.7639;
const LIMITS = { Flat: { min: 200, max: 6000, beds: [0, 1, 2, 3, 4, 5, 6] }, Villa: { min: 700, max: 20000, beds: [1, 2, 3, 4, 5, 6, 7] } };
const DEFAULT_SIZE = { Flat: 750, Villa: 2500 };

const $ = (id) => document.getElementById(id);
let model, areas, dashboard, metrics;

/* ---------- Formatting ---------- */

const aed = (n) => "AED " + Math.round(n).toLocaleString("en-US");
const aedShort = (n) => (n >= 1e9 ? `AED ${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `AED ${(n / 1e6).toFixed(n >= 1e7 ? 1 : 2)}M` : aed(n));
const roundPrice = (n) => (n >= 1e6 ? Math.round(n / 10000) * 10000 : Math.round(n / 1000) * 1000);

/* ---------- Model ---------- */

function predictLogPpsf(features) {
  let total = model.baseline;
  for (const tree of model.trees) {
    let i = 0;
    while (!tree.leaf[i]) i = features[tree.f[i]] <= tree.t[i] ? tree.l[i] : tree.r[i];
    total += tree.v[i];
  }
  return total;
}

function buildFeatures(area, type, offplan, bedrooms, sqft) {
  // Same order and meaning as FEATURES in the Python notebook.
  const typeInfo = area.types[type];
  return [
    area.value,
    typeInfo ? typeInfo.value : model.type_value[type],
    area.sales_log,
    type === "Villa" ? 1 : 0,
    offplan,
    bedrooms,
    Math.log(sqft),
  ];
}

/* ---------- Form ---------- */

function selected(name) {
  return document.querySelector(`input[name="${name}"]:checked`).value;
}

function sizeInSqft() {
  const v = Number($("size").value);
  return selected("unit") === "sqm" ? v * SQM_TO_SQFT : v;
}

function fillBedrooms(type, reset = false) {
  const current = Number($("bedrooms").value || 1);
  $("bedrooms").innerHTML = LIMITS[type].beds
    .map((b) => `<option value="${b}">${b === 0 ? "Studio" : `${b} bedroom${b > 1 ? "s" : ""}`}</option>`)
    .join("");
  const beds = LIMITS[type].beds;
  const fallback = type === "Villa" ? 4 : 1;
  $("bedrooms").value = String(!reset && beds.includes(current) ? current : fallback);
}

function update() {
  const area = areas.find((a) => a.name === $("area").value);
  const type = selected("type");
  const offplan = Number(selected("status"));
  const bedrooms = Number($("bedrooms").value);
  const sqft = sizeInSqft();
  const { min, max } = LIMITS[type];
  const unit = selected("unit");
  const toUnit = (s) => (unit === "sqm" ? Math.round(s / SQM_TO_SQFT) + " m²" : s.toLocaleString("en-US") + " sq ft");

  if (!area || !Number.isFinite(sqft) || sqft < min || sqft > max) {
    $("size-help").textContent = `Enter a size between ${toUnit(min)} and ${toUnit(max)} for ${type === "Villa" ? "villas" : "apartments"}.`;
    $("size-help").classList.add("error");
    $("price").textContent = "–";
    ["range", "compare"].forEach((id) => ($(id).textContent = ""));
    ["ppsf", "area-ppsf", "area-sales"].forEach((id) => ($(id).textContent = "–"));
    return;
  }
  $("size-help").textContent = `Accepted ${type === "Villa" ? "villa" : "apartment"} sizes: ${toUnit(min)} to ${toUnit(max)}.`;
  $("size-help").classList.remove("error");

  const ppsf = Math.exp(predictLogPpsf(buildFeatures(area, type, offplan, bedrooms, sqft)));
  const price = ppsf * sqft;
  $("price").textContent = aed(roundPrice(price));
  $("range").textContent = `Likely range ${aedShort(roundPrice(price * model.band.low))} – ${aedShort(roundPrice(price * model.band.high))} (80% of test sales fell within this band)`;
  $("ppsf").textContent = aed(ppsf);

  const typeInfo = area.types[type];
  $("area-sales").textContent = area.sales.toLocaleString("en-US");
  const warn = $("warn");
  if (typeInfo) {
    $("area-ppsf").textContent = aed(typeInfo.median_ppsf);
    const diff = ppsf / typeInfo.median_ppsf - 1;
    const word = type === "Villa" ? "villas" : "apartments";
    $("compare").textContent =
      Math.abs(diff) < 0.03
        ? `About the same per sq ft as the typical ${word} sold in ${area.name}.`
        : `${Math.round(Math.abs(diff) * 100)}% ${diff > 0 ? "above" : "below"} the typical price per sq ft for ${word} in ${area.name}.`;
    warn.hidden = typeInfo.sales >= 30;
    warn.textContent = `Only ${typeInfo.sales} ${word} sold here in 2026, so treat this estimate with extra care.`;
  } else {
    $("area-ppsf").textContent = "–";
    $("compare").textContent = "";
    warn.hidden = false;
    warn.textContent = `No ${type === "Villa" ? "villa" : "apartment"} sales in ${area.name} in 2026, so this estimate is less reliable.`;
  }
}

function setupForm() {
  const sorted = [...areas].sort((a, b) => a.name.localeCompare(b.name));
  $("area").innerHTML = sorted.map((a) => `<option value="${a.name}">${a.name}</option>`).join("");
  $("area").value = areas.some((a) => a.name === "Jumeirah Village Circle") ? "Jumeirah Village Circle" : sorted[0].name;
  fillBedrooms("Flat");

  document.querySelectorAll('input[name="type"]').forEach((r) =>
    r.addEventListener("change", () => {
      const type = selected("type");
      fillBedrooms(type, true);
      const s = DEFAULT_SIZE[type];
      $("size").value = selected("unit") === "sqm" ? Math.round(s / SQM_TO_SQFT) : s;
      update();
    }),
  );
  document.querySelectorAll('input[name="unit"]').forEach((r) =>
    r.addEventListener("change", () => {
      const v = Number($("size").value);
      $("size").value = selected("unit") === "sqm" ? Math.round(v / SQM_TO_SQFT) : Math.round(v * SQM_TO_SQFT);
      $("size").step = selected("unit") === "sqm" ? 1 : 10;
      update();
    }),
  );
  $("form").addEventListener("input", update);
  $("form").addEventListener("submit", (e) => e.preventDefault());
  update();
}

/* ---------- Dashboard ---------- */

function fillStats() {
  const gb = metrics.results.find((r) => r.model === "Gradient boosting");
  $("stat-sales").textContent = dashboard.total_sales.toLocaleString("en-US");
  $("lead-sales").textContent = dashboard.total_sales.toLocaleString("en-US");
  $("stat-value").textContent = aedShort(dashboard.total_value);
  $("stat-ppsf").textContent = aed(dashboard.median_ppsf);
  $("stat-error").textContent = gb["median % error"];
  $("period").textContent = `Residential sales registered with the Dubai Land Department, ${dashboard.period}. October is partial.`;
  $("model-intro").textContent = `The model learned from ${metrics.train_period} sales (${metrics.train_sales.toLocaleString("en-US")}) and was tested on ${metrics.test_period} sales it had never seen (${metrics.test_sales.toLocaleString("en-US")}).`;
  $("results").innerHTML = metrics.results
    .map((r) => `<tr class="${r.model === "Gradient boosting" ? "best" : ""}"><td>${r.model}</td><td>${r["median % error"]}</td><td>${r["within 10%"]}</td><td>${r["within 20%"]}</td></tr>`)
    .join("");
}

function drawCharts() {
  Chart.defaults.font.family = "Manrope, Segoe UI, sans-serif";
  Chart.defaults.color = "#5a6283";
  Chart.defaults.plugins.legend.labels.boxWidth = 12;
  const grid = { color: "rgba(13,18,48,0.06)" };
  const months = dashboard.monthly.map((m) => m.month);
  const k = (v) => (v >= 1000 ? (v / 1000).toFixed(v >= 10000 ? 0 : 1) + "k" : v);

  new Chart($("c-monthly"), {
    type: "bar",
    data: { labels: months, datasets: [
      { label: "Apartments", data: dashboard.monthly.map((m) => m.flats), backgroundColor: "#2b59ff", borderRadius: 6 },
      { label: "Villas", data: dashboard.monthly.map((m) => m.villas), backgroundColor: "#d63c8f", borderRadius: 6 },
    ] },
    options: { maintainAspectRatio: false, scales: { x: { stacked: true, grid: { display: false } }, y: { stacked: true, grid, ticks: { callback: k } } } },
  });

  new Chart($("c-ppsf"), {
    type: "line",
    data: { labels: months, datasets: [
      { label: "Apartments", data: dashboard.monthly.map((m) => m.flat_ppsf), borderColor: "#2b59ff", backgroundColor: "#2b59ff", tension: 0.35 },
      { label: "Villas", data: dashboard.monthly.map((m) => m.villa_ppsf), borderColor: "#d63c8f", backgroundColor: "#d63c8f", tension: 0.35 },
    ] },
    options: { maintainAspectRatio: false, scales: { x: { grid: { display: false } }, y: { grid } } },
  });

  const hbar = (id, items, label, value, color, fmt) =>
    new Chart($(id), {
      type: "bar",
      data: { labels: items.map(label), datasets: [{ data: items.map(value), backgroundColor: color, borderRadius: 6 }] },
      options: {
        indexAxis: "y", maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => fmt(c.raw) } } },
        scales: { x: { grid, ticks: { callback: (v) => fmt(v), maxTicksLimit: 6, maxRotation: 0 } }, y: { grid: { display: false }, ticks: { autoSkip: false, font: { size: 11.5 } } } },
      },
    });

  hbar("c-expensive", dashboard.expensive, (d) => d.area, (d) => d.ppsf, "#d63c8f", (v) => v.toLocaleString("en-US"));
  hbar("c-affordable", dashboard.affordable, (d) => d.area, (d) => d.ppsf, "#0fb5d4", (v) => v.toLocaleString("en-US"));
  hbar("c-busy", dashboard.top_areas, (d) => d.area, (d) => d.sales, "#7a4dff", (v) => k(v));
  hbar("c-bed", dashboard.bedrooms, (d) => d.label, (d) => d.median_price, "#2b59ff", (v) => (v === 0 ? "0" : `AED ${(v / 1e6).toFixed(1)}M`));

  const imp = Object.entries(metrics.importance).sort((a, b) => b[1] - a[1]);
  hbar("c-importance", imp, (d) => d[0], (d) => d[1], "#7a4dff", (v) => Number(v).toFixed(2));
}

/* ---------- Start ---------- */

async function init() {
  [model, areas, dashboard, metrics] = await Promise.all(
    ["data/model.json", "data/areas.json", "data/dashboard.json", "data/metrics.json"].map((u) => fetch(u).then((r) => r.json())),
  );
  setupForm();
  fillStats();
  if (window.Chart) drawCharts();
}

init().catch((err) => {
  console.error(err);
  $("price").textContent = "Couldn't load the model";
});
