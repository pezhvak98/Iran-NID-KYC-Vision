// ── عناصر DOM ─────────────────────────────────────────────
const stage = document.getElementById("stage");
const cameraRotator = document.getElementById("cameraRotator");
const cameraStream = document.getElementById("cameraStream");
const cameraUrlInput = document.getElementById("cameraUrlInput");
const connectBtn = document.getElementById("connectBtn");
const rotateBtn = document.getElementById("rotateBtn");
const rotateLabel = document.getElementById("rotateLabel");
const captureBtn = document.getElementById("captureBtn");
const statusBadge = document.getElementById("statusBadge");
const statusText = document.getElementById("statusText");
const previewModal = document.getElementById("previewModal");
const previewImage = document.getElementById("previewImage");
const retakeBtn = document.getElementById("retakeBtn");
const confirmBtn = document.getElementById("confirmBtn");
const toast = document.getElementById("toast");
const fileInput = document.getElementById("fileInput");
const originalWrap = document.getElementById("originalWrap");
const originalImage = document.getElementById("originalImage");
const emptyState = document.getElementById("emptyState");
const reviewBanner = document.getElementById("reviewBanner");
const resultArea = document.getElementById("resultArea");
const photoCard = document.getElementById("photoCard");
const photoImg = document.getElementById("photoImg");
const photoStatus = document.getElementById("photoStatus");
const metaInfo = document.getElementById("metaInfo");
const reviewForm = document.getElementById("reviewForm");
const approveBtn = document.getElementById("approveBtn");
const warningsWrap = document.getElementById("warningsWrap");
const warningsCount = document.getElementById("warningsCount");
const warningsList = document.getElementById("warningsList");
const roiLayer = document.getElementById("roiLayer");
const errorPanel = document.getElementById("errorPanel");
const errorList = document.getElementById("errorList");
const errorDismiss = document.getElementById("errorDismiss");
const successCard = document.getElementById("successCard");
const successFields = document.getElementById("successFields");
const successTimestamp = document.getElementById("successTimestamp");
const newCaseBtn = document.getElementById("newCaseBtn");
const jsonOut = document.getElementById("jsonOut");
const copyBtn = document.getElementById("copyBtn");

const API_BASE = location.origin && location.origin.startsWith("http") ? location.origin : "http://127.0.0.1:8000";
const DEFAULT_STREAM_URL = `${API_BASE}/api/v1/camera-stream`;

let capturedBlob = null;
let capturedUrl = null;
let sourceUrl = null;
let lastResult = null;
let processing = false;

// جهت چرخش اصلاحی تصویر دوربین: ۱ = ساعتگرد، ۱- = پادساعتگرد.
// دوربین به‌صورت افقی نگه داشته می‌شود اما قاب راهنما عمودی است،
// پس تصویر خام باید ۹۰ درجه بچرخد تا با قاب هم‌جهت شود.
let rotationDir = 1;

// ── مختصات دقیق ROI (بر اساس card_layout.py) ───────────────
// فرمت: [x1, y1, x2, y2] -> درصدی از عرض و ارتفاع کارت
const ROI_COORDS = {
  "photo":       { "box": [0.045, 0.185, 0.352, 0.746],  },
  "national_id": { "box": [0.566, 0.243, 0.831, 0.338],  },
  "first_name":  { "box": [0.449, 0.357, 0.831, 0.445],  },
  "last_name":   { "box": [0.446, 0.459, 0.830, 0.557],  },
  "birth_date":  { "box": [0.572, 0.571, 0.831, 0.649], },
  "father_name": { "box": [0.452, 0.665, 0.831, 0.748],  },
  "expiry_date": { "box": [0.563, 0.759, 0.820, 0.844],  }
};

const FIELD_DEFS = [
  ["national_id", "شماره ملی"],
  ["first_name", "نام"],
  ["last_name", "نام خانوادگی"],
  ["father_name", "نام پدر"],
  ["birth_date", "تاریخ تولد"],
  ["expiry_date", "پایان اعتبار"],
];

// ── توابع کمکی ─────────────────────────────────────────────
function setStatus(text, type = "info") {
  statusText.textContent = text;
  statusBadge.className = `status-badge ${type}`;
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 3200);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ── پنل خطاهای سراسری (خطاهای HTTP/سیستمی دریافتی از بک‌اند) ─
function showError(message) {
  const li = document.createElement("li");
  li.textContent = message;
  errorList.appendChild(li);
  errorPanel.hidden = false;
}

function clearErrors() {
  errorList.innerHTML = "";
  errorPanel.hidden = true;
}

errorDismiss.addEventListener("click", clearErrors);

// ── رسم شماتیک کارت و باکس‌های ROI ─────────────────────────
function renderROIGuides() {
  if (!roiLayer) return;
  roiLayer.innerHTML = "";

  for (const [key, data] of Object.entries(ROI_COORDS)) {
    const [x1, y1, x2, y2] = data.box;

    const box = document.createElement("div");
    box.className = "roi-box";
    box.style.left = `${x1 * 100}%`;
    box.style.top = `${y1 * 100}%`;
    box.style.width = `${(x2 - x1) * 100}%`;
    box.style.height = `${(y2 - y1) * 100}%`;

    const label = document.createElement("span");
    label.className = "roi-label";
    label.textContent = data.label;
    //box.appendChild(label);

    roiLayer.appendChild(box);
  }
}

// ── چرخش تصویر دوربین ───────────────────────────────────────
// چون دوربین افقی نگه داشته می‌شود ولی قاب راهنما عمودی است،
// اندازهٔ چرخاننده را برابر با ابعادِ جابه‌جاشدهٔ استیج تنظیم می‌کنیم
// تا پس از چرخش ۹۰ درجه، دقیقاً قاب عمودی را پر کند.
function applyRotatorSize() {
  const rect = stage.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  cameraRotator.style.width = `${rect.height}px`;
  cameraRotator.style.height = `${rect.width}px`;
  cameraRotator.style.transform = `translate(-50%, -50%) rotate(${rotationDir * 90}deg)`;
}

if (typeof ResizeObserver !== "undefined") {
  new ResizeObserver(() => applyRotatorSize()).observe(stage);
} else {
  window.addEventListener("resize", applyRotatorSize);
}

rotateBtn.addEventListener("click", () => {
  rotationDir *= -1;
  const isCCW = rotationDir === -1;
  rotateBtn.setAttribute("aria-pressed", String(isCCW));
  rotateLabel.textContent = isCCW ? "چرخش پادساعتگرد" : "چرخش ساعتگرد";
  applyRotatorSize();
});

// ── اتصال دوربین ──────────────────────────────────────────
function connectCamera() {
  const url = cameraUrlInput.value.trim() || DEFAULT_STREAM_URL;
  captureBtn.disabled = false;
  setStatus("در حال اتصال به دوربین...", "info");
  cameraStream.crossOrigin = "anonymous"; 
  const separator = url.includes("?") ? "&" : "?";
  cameraStream.src = `${url}${separator}t=${Date.now()}`;
}

cameraStream.addEventListener("load", () => {
  setStatus("دوربین متصل شد.", "success");
  captureBtn.disabled = false;
  applyRotatorSize();
});

cameraStream.addEventListener("error", () => {
  setStatus("اتصال به دوربین ناموفق بود؛ از آپلود استفاده کنید.", "error");
  captureBtn.disabled = true;
});

connectBtn.addEventListener("click", connectCamera);

// ── ثبت تصویر از استریم ────────────────────────────────────
captureBtn.addEventListener("click", () => {
  const width = cameraStream.naturalWidth;
  const height = cameraStream.naturalHeight;
  if (!width || !height) {
    showToast("تصویر دوربین در دسترس نیست.");
    return;
  }
  const canvas = document.createElement("canvas");
  // ابعاد بوم را جابه‌جا می‌کنیم تا خروجیِ ثبت‌شده هم‌جهت با پیش‌نمایش عمودی باشد
  canvas.width = height;
  canvas.height = width;
  const context = canvas.getContext("2d");
  try {
    context.translate(canvas.width / 2, canvas.height / 2);
    context.rotate((rotationDir * 90 * Math.PI) / 180);
    context.drawImage(cameraStream, -width / 2, -height / 2, width, height);
    canvas.toBlob((blob) => {
      if (!blob) return;
      capturedBlob = blob;
      if (capturedUrl) URL.revokeObjectURL(capturedUrl);
      capturedUrl = URL.createObjectURL(blob);
      previewImage.src = capturedUrl;
      previewModal.classList.add("open");
    }, "image/jpeg", 0.92);
  } catch (error) {
    console.error(error);
    showToast("امکان گرفتن عکس از دوربین نیست (CORS)؛ از آپلود استفاده کنید.");
  }
});

retakeBtn.addEventListener("click", () => previewModal.classList.remove("open"));

confirmBtn.addEventListener("click", () => {
  previewModal.classList.remove("open");
  if (capturedBlob) {
    setSource(capturedUrl);
    processBlob(capturedBlob, "capture.jpg");
  }
});

// ── آپلود تصویر ────────────────────────────────────────────
fileInput.addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  setSource(URL.createObjectURL(file));
  processBlob(file, file.name);
  fileInput.value = "";
});

function setSource(url) {
  sourceUrl = url;
  originalImage.src = url;
  originalWrap.hidden = false;
}

// ── پردازش و ارسال به بک‌اند ───────────────────────────────
async function processBlob(blob, filename = "capture.jpg") {
  if (processing) return;
  processing = true;
  captureBtn.disabled = true;
  setStatus("در حال پردازش تصویر...", "info");
  clearErrors();
  successCard.hidden = true;

  const fd = new FormData();
  fd.append("file", blob, filename);

  try {
    const res = await fetch(`${API_BASE}/api/v1/process-card`, { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => null);
      throw new Error((err && err.detail) || `HTTP ${res.status}`);
    }
    const result = await res.json();
    lastResult = result;
    renderResult(result);

    if (result.needs_review) {
      setStatus("نیاز به بازبینی اپراتور", "error");
      showToast("استخراج انجام شد؛ نیاز به بازبینی دارد.");
    } else {
      setStatus("استخراج کامل شد", "success");
      showToast("استخراج با موفقیت کامل شد.");
    }
  } catch (error) {
    console.error(error);
    setStatus("پردازش ناموفق بود.", "error");
    showError("درخواست پردازش تصویر ناموفق بود: " + error.message);
    showToast("خطا در پردازش تصویر.");
  } finally {
    processing = false;
    if (cameraStream.naturalWidth) captureBtn.disabled = false;
  }
}

// ── رندر نتایج به صورت فرم قابل ویرایش ────────────────────
function renderResult(r) {
  emptyState.hidden = true;
  resultArea.hidden = false;
  successCard.hidden = true;
  const details = successCard.querySelector(".json-details");
  if (details) details.open = false;

  reviewBanner.hidden = !r.needs_review;
  if (r.needs_review) {
    reviewBanner.textContent = "این استخراج نیاز به تأیید اپراتور دارد؛ لطفاً مقادیر زیر را بررسی و در صورت نیاز اصلاح کنید.";
  }

  if (r.photo && r.photo.available && r.photo.base64_png) {
    photoCard.hidden = false;
    photoImg.src = "data:image/png;base64," + r.photo.base64_png;
    photoStatus.textContent = r.photo.face_detected ? "چهره تشخیص داده شد" : "چهره خودکار تشخیص نشد";
  } else {
    photoCard.hidden = true;
  }

  metaInfo.innerHTML =
    `<span class="chip ${r.card && r.card.detected ? 'ok' : 'warn'}">کارت: ${r.card && r.card.detected ? 'تشخیص داده شد' : 'تراز با شابلون'}</span>` +
    `<span class="chip ${r.validation && r.validation.national_id ? 'ok' : 'warn'}">Checksum: ${r.validation && r.validation.national_id ? 'معتبر' : 'نامعتبر'}</span>` +
    `<span class="chip">Blur: ${r.quality ? Math.round(r.quality.blur_score) : '-'}</span>`;

  reviewForm.innerHTML = "";
  FIELD_DEFS.forEach(([key, label]) => {
    const isDate = key === "birth_date" || key === "expiry_date";
    const val = r.data ? r.data[key] : null;
    const conf = r.field_confidence ? Math.round((r.field_confidence[key] || 0) * 100) : 0;
    const current = isDate ? ((val && val.jalali) || "") : (val || "");

    const wrap = document.createElement("div");
    wrap.className = "field";
    wrap.innerHTML = `<label>${label} <small>${conf}٪</small></label><input id="edit_${key}" value="${escapeHtml(current)}" placeholder="—">`;
    reviewForm.appendChild(wrap);

    if (isDate && val && Array.isArray(val.candidates) && val.candidates.length) {
      const row = document.createElement("div");
      row.className = "cand-row";
      val.candidates.forEach((c) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "chip cand";
        b.textContent = c;
        b.addEventListener("click", () => { wrap.querySelector("input").value = c; });
        row.appendChild(b);
      });
      wrap.appendChild(row);
    }
  });

  const warnings = r.warnings || [];
  warningsWrap.hidden = warnings.length === 0;
  warningsCount.textContent = warnings.length;
  warningsList.innerHTML = "";
  warnings.forEach((w) => {
    const li = document.createElement("li");
    li.textContent = w;
    warningsList.appendChild(li);
  });
}

// ── تأیید نهایی، خروجی ساختاریافته و کارت ثبت موفق ─────────
approveBtn.addEventListener("click", () => {
  if (!lastResult) return;
  const get = (k) => { const el = document.getElementById("edit_" + k); return el ? el.value.trim() : ""; };

  const approved = {
    national_id: get("national_id"),
    first_name: get("first_name"),
    last_name: get("last_name"),
    father_name: get("father_name"),
    birth_date: { jalali: get("birth_date") || null },
    expiry_date: { jalali: get("expiry_date") || null },
    photo_available: !!(lastResult.photo && lastResult.photo.available),
    validation: lastResult.validation || {},
    review_required: !!lastResult.needs_review,
    approved_at: new Date().toISOString(),
  };

  jsonOut.textContent = JSON.stringify(approved, null, 2);
  renderSuccessCard(approved);
  successCard.hidden = false;
  successCard.scrollIntoView({ behavior: "smooth", block: "start" });
  showToast("اطلاعات تأیید و با موفقیت ثبت شد.");
});

// کارت خلاصهٔ ثبت‌شده — بدون عکس، صرفاً فیلدهای متنی تأییدشده
function renderSuccessCard(approved) {
  successTimestamp.textContent = "زمان تأیید: " + new Date().toLocaleString("fa-IR");

  const rows = [
    ["کد ملی", approved.national_id || "—", true],
    ["نام", approved.first_name || "—", false],
    ["نام خانوادگی", approved.last_name || "—", false],
    ["نام پدر", approved.father_name || "—", false],
    ["تاریخ تولد", (approved.birth_date && approved.birth_date.jalali) || "—", true],
    ["پایان اعتبار", (approved.expiry_date && approved.expiry_date.jalali) || "—", true],
    ["اعتبارسنجی کد ملی", (approved.validation && approved.validation.national_id) ? "معتبر" : "نامعتبر", false],
    ["وضعیت بازبینی", approved.review_required ? "نیازمند بازبینی اپراتور" : "بدون نیاز به بازبینی", false],
  ];

  successFields.innerHTML = rows.map(([label, value, mono]) => `
    <div class="id-row">
      <dt>${escapeHtml(label)}</dt>
      <dd${mono ? "" : ' class="rtl-val"'}>${escapeHtml(value)}</dd>
    </div>
  `).join("");
}

copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(jsonOut.textContent);
    showToast("کپی شد.");
  } catch (e) {
    showToast("کپی ممکن نشد.");
  }
});

// ── شروع پروندهٔ جدید ───────────────────────────────────────
newCaseBtn.addEventListener("click", () => {
  lastResult = null;
  capturedBlob = null;
  if (capturedUrl) { URL.revokeObjectURL(capturedUrl); capturedUrl = null; }
  sourceUrl = null;

  resultArea.hidden = true;
  successCard.hidden = true;
  reviewBanner.hidden = true;
  warningsWrap.hidden = true;
  originalWrap.hidden = true;
  emptyState.hidden = false;
  clearErrors();

  setStatus(cameraStream.naturalWidth ? "دوربین متصل است." : "در انتظار اتصال به دوربین", cameraStream.naturalWidth ? "success" : "info");
});

// ── شروع برنامه ───────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  cameraUrlInput.value = DEFAULT_STREAM_URL;
  renderROIGuides();
  applyRotatorSize();
  connectCamera();
});
