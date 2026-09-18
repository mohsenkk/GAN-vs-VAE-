const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";           // 13.333 x 7.5
p.author = "MSc Thesis — Status Report";
p.title  = "Drug-Target Affinity Prediction: Research Path, Model Selection & Code Audit";

// ---------- palette ----------
const INK    = "14213D";
const INK2   = "2B3A55";
const TEAL   = "0E7C86";   // GAN / DCGAN-DTA
const VIOLET = "6C4AB6";   // VAE / Co-VAE
const AMBER  = "C77D14";   // findings / open
const CRIM   = "9E2A2B";   // blockers / mismatch
const GREEN  = "2F6F4E";   // completed / match
const TINT   = "F1F4F8";
const TINT2  = "E6ECF3";
const MUTED  = "62708A";
const W = "FFFFFF";

const HEAD = "Cambria";
const BODY = "Calibri";

const M = 0.55;             // left margin
const CW = 13.333 - 2 * M;  // content width

// ---------- helpers ----------
function titleBar(s, txt, kicker) {
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: M, y: 0.32, w: CW, h: 0.24, fontFace: BODY, fontSize: 11,
      bold: true, color: TEAL, charSpacing: 1.6, margin: 0
    });
  }
  s.addText(txt, {
    x: M, y: kicker ? 0.58 : 0.42, w: CW, h: 0.62, fontFace: HEAD, fontSize: 30,
    bold: true, color: INK, margin: 0
  });
}

function card(s, x, y, w, h, fill) {
  s.addShape(p.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.07,
    fill: { color: fill || TINT }, line: { color: TINT2, width: 0.75 }
  });
}

function chip(s, x, y, label, color, wOverride) {
  const w = wOverride || (0.30 + label.length * 0.078);
  s.addShape(p.ShapeType.roundRect, {
    x, y, w, h: 0.26, rectRadius: 0.13,
    fill: { color: color }, line: { color: color, width: 0.5 }
  });
  s.addText(label, {
    x, y, w, h: 0.26, fontFace: BODY, fontSize: 9, bold: true,
    color: W, align: "center", valign: "middle", margin: 0
  });
  return w;
}

function dot(s, x, y, d, color, glyph, sz) {
  s.addShape(p.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color }, line: { color, width: 0 } });
  if (glyph) {
    s.addText(glyph, {
      x, y, w: d, h: d, fontFace: BODY, fontSize: sz || 12, bold: true,
      color: W, align: "center", valign: "middle", margin: 0
    });
  }
}

function srcNote(s, txt) {
  s.addText(txt, {
    x: M, y: 6.97, w: CW, h: 0.34, fontFace: BODY, fontSize: 8.5,
    color: MUTED, italic: true, margin: 0, valign: "top"
  });
}

function arrowDown(s, x, y, h, color) {
  s.addShape(p.ShapeType.line, {
    x, y, w: 0, h, line: { color: color || MUTED, width: 1.5, endArrowType: "triangle" }
  });
}
function arrowRight(s, x, y, w, color) {
  s.addShape(p.ShapeType.line, {
    x, y, w, h: 0, line: { color: color || MUTED, width: 1.5, endArrowType: "triangle" }
  });
}

const SURVEY = "Conceptual structure adapted from Wang Y., Lv J., Xia Y., Xu J., Meng Y., Cui F., Wei L., Zou Q., Zhang Z., “A unified survey on drug–target interaction and binding affinity prediction: Models, representations, and challenges,” Biotechnology Advances, 2026 (PMID 41690335). Diagram redrawn — original figure not reproduced.";

// =====================================================================
// SLIDE 1 — Title
// =====================================================================
{
  const s = p.addSlide();
  s.background = { color: INK };
  s.addText("Drug–Target Affinity Prediction", {
    x: M, y: 2.05, w: CW, h: 0.95, fontFace: HEAD, fontSize: 44, bold: true, color: W, margin: 0
  });
  s.addText("Research Path, Model Selection & Code Audit", {
    x: M, y: 3.02, w: CW, h: 0.5, fontFace: BODY, fontSize: 21, color: "C8D4E6", margin: 0
  });

  s.addShape(p.ShapeType.roundRect, { x: M, y: 3.95, w: 2.55, h: 0.62, rectRadius: 0.1, fill: { color: TEAL }, line: { color: TEAL, width: 0 } });
  s.addText("DCGAN-DTA", { x: M, y: 3.95, w: 2.55, h: 0.62, fontFace: BODY, fontSize: 16, bold: true, color: W, align: "center", valign: "middle", margin: 0 });
  s.addText("vs", { x: 3.25, y: 3.95, w: 0.6, h: 0.62, fontFace: HEAD, fontSize: 15, italic: true, color: "8FA3BF", align: "center", valign: "middle", margin: 0 });
  s.addShape(p.ShapeType.roundRect, { x: 3.9, y: 3.95, w: 2.55, h: 0.62, rectRadius: 0.1, fill: { color: VIOLET }, line: { color: VIOLET, width: 0 } });
  s.addText("Co-VAE", { x: 3.9, y: 3.95, w: 2.55, h: 0.62, fontFace: BODY, fontSize: 16, bold: true, color: W, align: "center", valign: "middle", margin: 0 });

  s.addShape(p.ShapeType.roundRect, { x: M, y: 5.15, w: 7.4, h: 0.5, rectRadius: 0.08, fill: { color: "1E3157" }, line: { color: AMBER, width: 1 } });
  s.addText("Audit-stage report — no reproduction of either paper is claimed", {
    x: M + 0.18, y: 5.15, w: 7.0, h: 0.5, fontFace: BODY, fontSize: 12.5, color: "F0D9AE", valign: "middle", margin: 0
  });

  s.addText("MSc Thesis · Supervisor Status Meeting", {
    x: M, y: 6.55, w: CW, h: 0.3, fontFace: BODY, fontSize: 11, color: "8FA3BF", margin: 0
  });

  s.addNotes(
`Frame the meeting in one sentence: this is a status report on where the project stands, not a results presentation.

Say up front what the deck does: it walks from the problem definition, through how the DTA field got to where it is, to why I picked these two specific papers, and then to what I actually found when I opened their code.

The most important thing to flag early is the amber box. Neither paper has been reproduced. What has been done is two full paper audits and two full code audits. If the supervisor takes one thing from this meeting, it should be that the audits changed what the next experiment needs to look like.

Keep this slide to about 30 seconds.`);
}

// =====================================================================
// SLIDE 2 — The Research Problem
// =====================================================================
{
  const s = p.addSlide();
  titleBar(s, "The Research Problem", "Problem definition");

  // left flow
  const fx = M, fw = 5.1;
  card(s, fx, 1.45, 2.42, 0.95, TINT);
  dot(s, fx + 0.18, 1.63, 0.42, TEAL, "D", 13);
  s.addText("Drug", { x: fx + 0.72, y: 1.56, w: 1.6, h: 0.26, fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
  s.addText("small molecule,\ngiven as a SMILES string", { x: fx + 0.72, y: 1.81, w: 1.62, h: 0.5, fontFace: BODY, fontSize: 9, color: MUTED, margin: 0 });

  s.addText("+", { x: fx + 2.48, y: 1.45, w: 0.26, h: 0.95, fontFace: HEAD, fontSize: 20, bold: true, color: MUTED, align: "center", valign: "middle", margin: 0 });

  card(s, fx + 2.7, 1.45, 2.4, 0.95, TINT);
  dot(s, fx + 2.88, 1.63, 0.42, VIOLET, "T", 13);
  s.addText("Target protein", { x: fx + 3.42, y: 1.56, w: 1.6, h: 0.26, fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
  s.addText("given as an\namino-acid sequence", { x: fx + 3.42, y: 1.81, w: 1.6, h: 0.5, fontFace: BODY, fontSize: 9, color: MUTED, margin: 0 });

  arrowDown(s, fx + fw / 2, 2.48, 0.36, INK2);

  s.addShape(p.ShapeType.roundRect, { x: fx + 1.0, y: 2.92, w: 3.1, h: 0.62, rectRadius: 0.08, fill: { color: INK2 }, line: { color: INK2, width: 0 } });
  s.addText("Binding interaction", { x: fx + 1.0, y: 2.92, w: 3.1, h: 0.62, fontFace: BODY, fontSize: 13, bold: true, color: W, align: "center", valign: "middle", margin: 0 });

  arrowDown(s, fx + fw / 2, 3.62, 0.36, INK2);

  s.addShape(p.ShapeType.roundRect, { x: fx + 0.55, y: 4.06, w: 4.0, h: 0.88, rectRadius: 0.08, fill: { color: INK }, line: { color: INK, width: 0 } });
  s.addText("Binding affinity", { x: fx + 0.55, y: 4.14, w: 4.0, h: 0.3, fontFace: BODY, fontSize: 14, bold: true, color: W, align: "center", margin: 0 });
  s.addText("a continuous value — Kd, Ki or IC50, usually log-transformed", {
    x: fx + 0.65, y: 4.44, w: 3.8, h: 0.4, fontFace: BODY, fontSize: 9.5, color: "AFC0D8", align: "center", margin: 0
  });

  // right: DTI vs DTA
  const rx = 6.35, rw = 6.45;
  card(s, rx, 1.45, rw, 1.62, W);
  s.addShape(p.ShapeType.roundRect, { x: rx, y: 1.45, w: rw, h: 1.62, rectRadius: 0.07, fill: { color: W }, line: { color: TEAL, width: 1.25 } });
  chip(s, rx + 0.22, 1.66, "DTI", TEAL, 0.75);
  s.addText("Drug–Target Interaction", { x: rx + 1.1, y: 1.62, w: 4.9, h: 0.32, fontFace: BODY, fontSize: 14, bold: true, color: INK, margin: 0 });
  s.addText("Does this drug interact with this target, yes or no? A binary classification problem. Unknown pairs are often treated as pseudo-negatives, which is a known weakness of the formulation.", {
    x: rx + 0.22, y: 2.02, w: rw - 0.44, h: 0.9, fontFace: BODY, fontSize: 10.5, color: INK2, margin: 0
  });

  card(s, rx, 3.28, rw, 1.66, W);
  s.addShape(p.ShapeType.roundRect, { x: rx, y: 3.28, w: rw, h: 1.66, rectRadius: 0.07, fill: { color: W }, line: { color: VIOLET, width: 1.25 } });
  chip(s, rx + 0.22, 3.49, "DTA", VIOLET, 0.75);
  s.addText("Drug–Target Affinity", { x: rx + 1.1, y: 3.45, w: 4.9, h: 0.32, fontFace: BODY, fontSize: 14, bold: true, color: INK, margin: 0 });
  s.addText("How strongly do they bind? A regression problem on a measured, continuous value. Carries strictly more information than a binary label — it distinguishes a weak binder from a potent one.", {
    x: rx + 0.22, y: 3.85, w: rw - 0.44, h: 0.95, fontFace: BODY, fontSize: 10.5, color: INK2, margin: 0
  });

  s.addShape(p.ShapeType.roundRect, { x: M, y: 5.25, w: CW, h: 0.72, rectRadius: 0.07, fill: { color: TINT }, line: { color: TINT2, width: 0.75 } });
  s.addText([
    { text: "This project works on DTA, not DTI.  ", options: { bold: true, color: INK } },
    { text: "Both selected papers formulate the task as regression on a continuous affinity value, evaluated with ranking and error metrics rather than classification accuracy.", options: { color: INK2 } }
  ], { x: M + 0.22, y: 5.25, w: CW - 0.44, h: 0.72, fontFace: BODY, fontSize: 11.5, valign: "middle", margin: 0 });

  s.addNotes(
`This slide exists so we are all using the same vocabulary for the rest of the meeting.

Walk the left flow quickly: a drug comes in as a SMILES string, a target as an amino-acid sequence, they bind, and the thing we want to predict is the strength of that binding.

The distinction on the right is the one that matters. DTI is a yes/no question and is usually posed as classification. DTA is a "how much" question and is posed as regression. The survey makes the point that the binary formulation has to invent negatives, because non-interacting pairs are mostly just unmeasured pairs rather than confirmed non-binders. Affinity sidesteps that.

Conclusion the supervisor should take: everything after this slide is about the regression problem. Both of my papers are DTA papers, which is what makes them comparable at all.

Don't get drawn into the biochemistry here — if asked about Kd versus Ki versus IC50, say the papers use different ones and that this actually turns out to matter in the audit.`);
}

// =====================================================================
// SLIDE 3 — Why Computational DTA?
// =====================================================================
{
  const s = p.addSlide();
  titleBar(s, "Why Computational DTA?", "Motivation");

  // funnel
  const cx = 3.55;
  const rows = [
    { w: 5.6, t: "Chemical space of candidate compounds", sub: "far too large to screen exhaustively", c: INK },
    { w: 4.5, t: "Computational screening", sub: "model scores many pairs cheaply", c: TEAL },
    { w: 3.3, t: "Prioritised candidates", sub: "a shortlist worth the bench time", c: VIOLET },
    { w: 2.2, t: "Experimental validation", sub: "still required", c: AMBER }
  ];
  let y = 1.5;
  rows.forEach((r, i) => {
    s.addShape(p.ShapeType.roundRect, {
      x: cx - r.w / 2, y, w: r.w, h: 0.74, rectRadius: 0.06,
      fill: { color: r.c }, line: { color: r.c, width: 0 }
    });
    s.addText(r.t, { x: cx - r.w / 2, y: y + 0.08, w: r.w, h: 0.3, fontFace: BODY, fontSize: 11.5, bold: true, color: W, align: "center", margin: 0 });
    s.addText(r.sub, { x: cx - r.w / 2, y: y + 0.38, w: r.w, h: 0.28, fontFace: BODY, fontSize: 8.5, color: "DCE6F2", align: "center", margin: 0 });
    if (i < rows.length - 1) arrowDown(s, cx, y + 0.78, 0.28, MUTED);
    y += 1.14;
  });

  // right cards
  const rx = 7.1, rw = 5.7;
  const pts = [
    ["Cost", "Wet-lab assays are expensive per drug–target pair, so exhaustive screening is not affordable."],
    ["Time", "Assays are slow. A model can rank thousands of pairs before a single experiment is run."],
    ["Scale", "The number of plausible drug–target pairs grows multiplicatively; only computation scales with it."],
    ["Coverage", "Models can score pairs for which no measurement exists — the new-drug and new-target settings."]
  ];
  let py = 1.5;
  pts.forEach(([h, b], i) => {
    card(s, rx, py, rw, 1.06, TINT);
    dot(s, rx + 0.2, py + 0.28, 0.5, [TEAL, VIOLET, INK2, AMBER][i], String(i + 1), 12);
    s.addText(h, { x: rx + 0.85, y: py + 0.16, w: rw - 1.1, h: 0.26, fontFace: BODY, fontSize: 12.5, bold: true, color: INK, margin: 0 });
    s.addText(b, { x: rx + 0.85, y: py + 0.42, w: rw - 1.1, h: 0.56, fontFace: BODY, fontSize: 9.5, color: INK2, margin: 0 });
    py += 1.2;
  });

  s.addShape(p.ShapeType.roundRect, { x: M, y: 6.15, w: CW, h: 0.62, rectRadius: 0.07, fill: { color: "FBF3E4" }, line: { color: AMBER, width: 1 } });
  s.addText("Computational prediction narrows the experimental search space. It does not replace experimental validation — the bottom of the funnel stays in the lab.", {
    x: M + 0.22, y: 6.15, w: CW - 0.44, h: 0.62, fontFace: BODY, fontSize: 11.5, color: "6B4A12", valign: "middle", margin: 0
  });

  s.addNotes(
`Short slide. The point is to justify why anyone builds these models at all.

The funnel on the left is the argument in one picture: the candidate space is too big to test, computation cuts it down, and what survives goes to the bench. The four cards on the right are just the reasons that funnel is necessary — cost, time, scale, and coverage of pairs nobody has measured.

The amber bar at the bottom is deliberate and I should say it out loud. I am not claiming these models replace experiments. They reorder the queue. Supervisors tend to push back if a student overclaims here, so I'd rather state the limit myself.

The "coverage" card is worth a beat, because it sets up the new-drug and new-target evaluation settings that both papers use and that I audit later.`);
}

// =====================================================================
// SLIDE 4 — Evolution of DTI/DTA Research
// =====================================================================
{
  const s = p.addSlide();
  titleBar(s, "Evolution of DTI / DTA Research", "Literature review · primary source");

  const stages = [
    ["Similarity / feature-driven", "Hand-built drug and target descriptors; similarity kernels drive prediction.", INK2],
    ["Matrix decomposition", "Interaction matrix factorised into latent drug and target factors.", INK2],
    ["Network-based", "Drugs and targets as nodes; prediction becomes link inference on a heterogeneous graph.", INK2],
    ["Sequence-based modelling", "SMILES and amino-acid sequences used directly, no hand-crafted features.", TEAL],
    ["Structure-based modelling", "3D structural and binding-pocket information incorporated where available.", TEAL],
    ["Deep learning", "Representations learned end-to-end; CNN and RNN encoders over raw sequences.", TEAL],
    ["Graph / attention / multimodal", "Molecular graphs, attention over sequence, and fusion of several modalities.", VIOLET],
    ["Large pretrained models", "Self-supervised pretraining on large unlabelled chemical and protein corpora.", VIOLET]
  ];

  const bw = 2.85, gap = 0.29, bh = 1.5;
  const x0 = M;
  stages.forEach((st, i) => {
    const row = Math.floor(i / 4), col = i % 4;
    const x = x0 + col * (bw + gap);
    const y = 1.42 + row * (bh + 0.62);
    card(s, x, y, bw, bh, W);
    s.addShape(p.ShapeType.roundRect, { x, y, w: bw, h: bh, rectRadius: 0.07, fill: { color: W }, line: { color: st[2], width: 1.1 } });
    dot(s, x + 0.14, y + 0.14, 0.36, st[2], String(i + 1), 10);
    s.addText(st[0], { x: x + 0.58, y: y + 0.14, w: bw - 0.72, h: 0.46, fontFace: BODY, fontSize: 11, bold: true, color: INK, margin: 0 });
    s.addText(st[1], { x: x + 0.14, y: y + 0.66, w: bw - 0.28, h: 0.74, fontFace: BODY, fontSize: 8.5, color: MUTED, margin: 0 });
    if (col < 3) arrowRight(s, x + bw + 0.04, y + bh / 2, gap - 0.08, MUTED);
  });
  // wrap arrow row1 -> row2 (routed entirely inside the 0.62" gap between rows)
  const r1bot = 1.42 + bh, r2top = 1.42 + bh + 0.62, mid = (r1bot + r2top) / 2;
  s.addShape(p.ShapeType.line, { x: x0 + 3 * (bw + gap) + bw / 2, y: r1bot, w: 0, h: mid - r1bot, line: { color: MUTED, width: 1.5 } });
  s.addShape(p.ShapeType.line, { x: x0 + bw / 2, y: mid, w: 3 * (bw + gap), h: 0, line: { color: MUTED, width: 1.5 } });
  s.addShape(p.ShapeType.line, { x: x0 + bw / 2, y: mid, w: 0, h: r2top - mid, line: { color: MUTED, width: 1.5, endArrowType: "triangle" } });

  // task band
  s.addShape(p.ShapeType.roundRect, { x: M, y: 5.42, w: CW, h: 0.78, rectRadius: 0.07, fill: { color: TINT }, line: { color: TINT2, width: 0.75 } });
  s.addText([
    { text: "Across the same period the task itself shifts:  ", options: { color: INK2 } },
    { text: "DTI (binary interaction)", options: { bold: true, color: TEAL } },
    { text: "  →  ", options: { color: MUTED } },
    { text: "DTA (continuous affinity)", options: { bold: true, color: VIOLET } },
    { text: "  — richer supervision, and no need to invent negative pairs.", options: { color: INK2 } }
  ], { x: M + 0.22, y: 5.42, w: CW - 0.44, h: 0.78, fontFace: BODY, fontSize: 11.5, valign: "middle", margin: 0 });

  srcNote(s, SURVEY);

  s.addNotes(
`This is the slide I want the most time on. It is the spine of the literature review.

Walk left to right along the top row, then drop to the second row. The story is a steady removal of human feature engineering. Stage 1 depends entirely on descriptors somebody designed by hand. By stage 6 the model learns its own representation from the raw sequence. By stage 8 the representation is learned before the DTA task is even seen, on unlabelled corpora.

Two things to draw attention to. First, the colour shift is deliberate: dark for the classical era, teal for the first deep-learning era, violet for the current one. Second, the band at the bottom — the field also moved from asking "do they interact" to "how strongly", and that is a change in the task, not just in the model.

Be honest about the source: I am following the taxonomy from the 2026 Biotechnology Advances survey and I have redrawn the diagram myself rather than reproducing their figure. I should flag that I have worked from the survey's categorisation and abstract; I have not had access to the full text, so if the supervisor wants the figure itself we need library access.

Do not turn this into a list of papers. If asked for representative methods, give one or two per stage and move on.`);
}

// =====================================================================
// SLIDE 5 — Representation evolution
// =====================================================================
{
  const s = p.addSlide();
  titleBar(s, "How Drug and Target Representation Evolved", "Literature review");

  s.addText("Model progress and representation progress are the same story — each new model family exists because a new representation became learnable.", {
    x: M, y: 1.28, w: CW, h: 0.3, fontFace: BODY, fontSize: 11.5, color: INK2, margin: 0
  });

  const lanes = [
    ["DRUG", TEAL, ["SMILES strings, fingerprints,\nmolecular descriptors", "Molecular graphs\n(atoms as nodes, bonds as edges)", "Learned molecular\nrepresentations"]],
    ["TARGET", VIOLET, ["Protein descriptors and\nraw amino-acid sequence", "Learned sequence\nrepresentations", "Structural / binding-pocket\nrepresentations"]],
    ["JOINT", INK2, ["Feature fusion\n(concatenate, add, Kronecker)", "Explicit interaction\nmodelling / attention", "Affinity prediction head\n→ continuous value"]]
  ];

  let ly = 1.72;
  lanes.forEach(([name, col, steps]) => {
    s.addShape(p.ShapeType.roundRect, { x: M, y: ly, w: 1.28, h: 1.42, rectRadius: 0.06, fill: { color: col }, line: { color: col, width: 0 } });
    s.addText(name, { x: M, y: ly, w: 1.28, h: 1.42, fontFace: BODY, fontSize: 12.5, bold: true, color: W, align: "center", valign: "middle", margin: 0, charSpacing: 1 });

    const sw = 3.35, sg = 0.42;
    steps.forEach((t, j) => {
      const sx = M + 1.28 + 0.42 + j * (sw + sg);
      card(s, sx, ly, sw, 1.42, TINT);
      s.addText(t, { x: sx + 0.16, y: ly, w: sw - 0.32, h: 1.42, fontFace: BODY, fontSize: 10, color: INK, valign: "middle", margin: 0 });
      if (j < 2) arrowRight(s, sx + sw + 0.05, ly + 0.71, sg - 0.1, col);
    });
    ly += 1.62;
  });

  s.addShape(p.ShapeType.roundRect, { x: M, y: 6.4, w: CW, h: 0.5, rectRadius: 0.07, fill: { color: TINT }, line: { color: TINT2, width: 0.75 } });
  s.addText([
    { text: "Where my two papers sit:  ", options: { bold: true, color: INK } },
    { text: "both stay in the left-hand column — sequence-level input (SMILES + amino-acid string), no molecular graph, no 3D structure.", options: { color: INK2 } }
  ], { x: M + 0.22, y: 6.4, w: CW - 0.44, h: 0.5, fontFace: BODY, fontSize: 11, valign: "middle", margin: 0 });

  srcNote(s, SURVEY);

  s.addNotes(
`The purpose of this slide is to make the previous timeline concrete. The stages on slide 4 are not arbitrary — each one corresponds to a change in how the drug and the target are encoded.

Read it as three parallel tracks. Drugs move from strings and fingerprints to graphs to learned embeddings. Targets move from descriptors and raw sequence to learned sequence models to structure. The joint track is how the two sides get combined, and it moves from simple fusion to explicit interaction modelling.

The bottom bar is the one that matters for my project. Both DCGAN-DTA and Co-VAE sit in the leftmost column on both tracks — they consume label-encoded sequences and nothing else. That is a deliberate scoping choice, and it is also what makes them comparable to each other: they take the same kind of input.

If the supervisor asks why I am not looking at graph or structure-based methods, the answer is that keeping representation fixed isolates the variable I actually want to study, which is the generative family.`);
}

// =====================================================================
// SLIDE 6 — Modern DTA modelling landscape
// =====================================================================
{
  const s = p.addSlide();
  titleBar(s, "The Modern DTA Modelling Landscape", "Literature review");

  const fams = [
    ["CNN", "Convolutional encoders over SMILES and sequence"],
    ["RNN / LSTM", "Sequential encoders capturing long-range order"],
    ["GNN", "Message passing over molecular graphs"],
    ["Transformer / Attention", "Self-attention over tokens; interaction attention"],
    ["Hybrid", "Combinations, e.g. CNN encoder + attention fusion"],
    ["Structure-based", "3D structure, docking poses, binding pockets"],
    ["Multimodal", "Sequence + graph + structure + knowledge combined"],
    ["Pretrained", "Self-supervised chemical / protein language models"]
  ];
  const bw = 2.87, gap = 0.24, bh = 1.02;
  fams.forEach((f, i) => {
    const row = Math.floor(i / 4), col = i % 4;
    const x = M + col * (bw + gap), y = 1.36 + row * (bh + 0.22);
    card(s, x, y, bw, bh, TINT);
    s.addText(f[0], { x: x + 0.16, y: y + 0.12, w: bw - 0.32, h: 0.28, fontFace: BODY, fontSize: 12, bold: true, color: INK, margin: 0 });
    s.addText(f[1], { x: x + 0.16, y: y + 0.42, w: bw - 0.32, h: 0.5, fontFace: BODY, fontSize: 8.5, color: MUTED, margin: 0 });
  });

  // generative highlight band
  s.addShape(p.ShapeType.roundRect, { x: M, y: 4.12, w: CW, h: 1.55, rectRadius: 0.08, fill: { color: "F6F2FB" }, line: { color: VIOLET, width: 1.25 } });
  chip(s, M + 0.24, 4.32, "GENERATIVE / REPRESENTATION LEARNING", VIOLET, 3.75);
  s.addText("A cross-cutting branch rather than a separate architecture family. These models learn a representation (or a data distribution) as well as — or before — predicting affinity.", {
    x: M + 0.24, y: 4.68, w: CW - 0.5, h: 0.42, fontFace: BODY, fontSize: 10.5, color: INK2, margin: 0
  });

  s.addShape(p.ShapeType.roundRect, { x: M + 0.24, y: 5.12, w: 3.0, h: 0.42, rectRadius: 0.07, fill: { color: TEAL }, line: { color: TEAL, width: 0 } });
  s.addText("GAN — adversarial", { x: M + 0.24, y: 5.12, w: 3.0, h: 0.42, fontFace: BODY, fontSize: 11, bold: true, color: W, align: "center", valign: "middle", margin: 0 });
  s.addShape(p.ShapeType.roundRect, { x: M + 3.44, y: 5.12, w: 3.0, h: 0.42, rectRadius: 0.07, fill: { color: VIOLET }, line: { color: VIOLET, width: 0 } });
  s.addText("VAE — latent-variable", { x: M + 3.44, y: 5.12, w: 3.0, h: 0.42, fontFace: BODY, fontSize: 11, bold: true, color: W, align: "center", valign: "middle", margin: 0 });
  s.addText("←  the branch this thesis examines", { x: M + 6.7, y: 5.12, w: 5.0, h: 0.42, fontFace: BODY, fontSize: 10.5, italic: true, color: VIOLET, valign: "middle", margin: 0 });

  s.addShape(p.ShapeType.roundRect, { x: M, y: 5.92, w: CW, h: 0.62, rectRadius: 0.07, fill: { color: "FBF3E4" }, line: { color: AMBER, width: 1 } });
  s.addText("Important caveat: GAN- and VAE-based methods are a minority branch of DTA research, not the dominant one. Current state-of-the-art results are mostly reported by graph, attention and pretrained models.", {
    x: M + 0.22, y: 5.92, w: CW - 0.44, h: 0.62, fontFace: BODY, fontSize: 11, color: "6B4A12", valign: "middle", margin: 0
  });

  srcNote(s, SURVEY);

  s.addNotes(
`This slide positions my work inside the field rather than overselling it.

The eight cards are the model families the survey organises modern DTA work into. I do not need to dwell on each one — point at them, note that they are mostly about how the encoder is built.

The violet band is the real content. Generative and representation-learning methods cut across the families rather than sitting beside them: a GAN or a VAE is a way of learning the representation, and the affinity head can then be a CNN or anything else. That is exactly what both of my papers do.

The amber caveat is the honest part and I should say it without being prompted. GAN and VAE approaches are not where the leaderboard is. If the supervisor asks "why study a branch that isn't state of the art", the answer is that the interesting question is whether the generative component actually contributes what the papers claim it contributes — which is a reproducibility question, and it turns out to be a good one.`);
}

// =====================================================================
// SLIDE 7 — Why GAN and VAE
// =====================================================================
{
  const s = p.addSlide();
  titleBar(s, "Why GAN and VAE?", "Conceptual bridge");

  // GAN box
  s.addShape(p.ShapeType.roundRect, { x: M, y: 1.34, w: 6.05, h: 2.42, rectRadius: 0.08, fill: { color: W }, line: { color: TEAL, width: 1.4 } });
  s.addText("GAN — adversarial learning", { x: M + 0.26, y: 1.5, w: 5.5, h: 0.3, fontFace: BODY, fontSize: 14, bold: true, color: TEAL, margin: 0 });

  s.addShape(p.ShapeType.roundRect, { x: M + 0.3, y: 2.0, w: 1.75, h: 0.62, rectRadius: 0.06, fill: { color: TINT }, line: { color: TEAL, width: 0.9 } });
  s.addText("Generator", { x: M + 0.3, y: 2.0, w: 1.75, h: 0.62, fontFace: BODY, fontSize: 11, bold: true, color: INK, align: "center", valign: "middle", margin: 0 });
  arrowRight(s, M + 2.12, 2.31, 0.62, TEAL);
  s.addShape(p.ShapeType.roundRect, { x: M + 2.82, y: 2.0, w: 1.9, h: 0.62, rectRadius: 0.06, fill: { color: TINT }, line: { color: TEAL, width: 0.9 } });
  s.addText("Discriminator", { x: M + 2.82, y: 2.0, w: 1.9, h: 0.62, fontFace: BODY, fontSize: 11, bold: true, color: INK, align: "center", valign: "middle", margin: 0 });
  s.addText("real / fake", { x: M + 4.8, y: 2.0, w: 1.1, h: 0.62, fontFace: BODY, fontSize: 9.5, color: MUTED, valign: "middle", margin: 0 });

  s.addText("Two networks trained against each other. The discriminator learns features that separate real sequences from generated ones — and those features can then be reused for prediction.", {
    x: M + 0.3, y: 2.76, w: 5.5, h: 0.86, fontFace: BODY, fontSize: 10, color: INK2, margin: 0
  });

  // VAE box
  const vx = 6.95;
  s.addShape(p.ShapeType.roundRect, { x: vx, y: 1.34, w: 5.85, h: 2.42, rectRadius: 0.08, fill: { color: W }, line: { color: VIOLET, width: 1.4 } });
  s.addText("VAE — probabilistic latent variables", { x: vx + 0.26, y: 1.5, w: 5.3, h: 0.3, fontFace: BODY, fontSize: 14, bold: true, color: VIOLET, margin: 0 });

  const vb = [["Encoder", 0.3, 1.55], ["z", 2.1, 0.62], ["Decoder", 3.0, 1.55]];
  vb.forEach(([t, dx, w]) => {
    s.addShape(p.ShapeType.roundRect, { x: vx + dx, y: 2.0, w, h: 0.62, rectRadius: 0.06, fill: { color: TINT }, line: { color: VIOLET, width: 0.9 } });
    s.addText(t, { x: vx + dx, y: 2.0, w, h: 0.62, fontFace: BODY, fontSize: 11, bold: true, color: INK, align: "center", valign: "middle", margin: 0 });
  });
  arrowRight(s, vx + 1.9, 2.31, 0.16, VIOLET);
  arrowRight(s, vx + 2.78, 2.31, 0.18, VIOLET);
  s.addText("reconstruction", { x: vx + 4.62, y: 2.0, w: 1.15, h: 0.62, fontFace: BODY, fontSize: 9.5, color: MUTED, valign: "middle", margin: 0 });

  s.addText("An encoder maps the input to a distribution over a latent code; a decoder reconstructs from a sample. The latent code is a compressed, probabilistic representation usable for prediction.", {
    x: vx + 0.3, y: 2.76, w: 5.3, h: 0.86, fontFace: BODY, fontSize: 10, color: INK2, margin: 0
  });

  // funnel chain
  const chain = [
    ["DTA", INK], ["Deep learning", INK2], ["Generative / representation learning", MUTED],
    ["GAN  ·  VAE", VIOLET], ["DCGAN-DTA  ·  Co-VAE", TEAL]
  ];
  let cyx = M;
  const cw = [1.35, 2.05, 3.55, 1.75, 2.65];
  chain.forEach((c, i) => {
    s.addShape(p.ShapeType.roundRect, { x: cyx, y: 4.28, w: cw[i], h: 0.68, rectRadius: 0.07, fill: { color: c[1] }, line: { color: c[1], width: 0 } });
    s.addText(c[0], { x: cyx, y: 4.28, w: cw[i], h: 0.68, fontFace: BODY, fontSize: 11, bold: true, color: W, align: "center", valign: "middle", margin: 0 });
    if (i < chain.length - 1) arrowRight(s, cyx + cw[i] + 0.02, 4.62, 0.16, MUTED);
    cyx += cw[i] + 0.2;
  });

  s.addShape(p.ShapeType.roundRect, { x: M, y: 5.35, w: CW, h: 1.12, rectRadius: 0.07, fill: { color: TINT }, line: { color: TINT2, width: 0.75 } });
  s.addText([
    { text: "The shared idea.  ", options: { bold: true, color: INK } },
    { text: "Both families learn a representation from data that is not labelled with affinity — the GAN from an adversarial game, the VAE from reconstruction — and then use that representation for the affinity regression. ", options: { color: INK2 } },
    { text: "Whether that transferred representation actually helps is precisely what the audits put in question.", options: { bold: true, color: CRIM } }
  ], { x: M + 0.24, y: 5.35, w: CW - 0.48, h: 1.12, fontFace: BODY, fontSize: 11.5, valign: "middle", margin: 0 });

  s.addNotes(
`This is the hinge of the presentation — it connects the literature review to my two papers.

Keep the two boxes conceptual. GAN: two networks compete, and the discriminator ends up with useful features. VAE: encode to a distribution, sample, decode, and the latent code is the useful representation. Do not go into the loss functions here; the maths comes later and only where the audit needs it.

The chain across the middle is the argument in one line: DTA, narrowed to deep learning, narrowed to generative and representation learning, narrowed to GAN and VAE, and then to the two specific papers I selected.

The last sentence in the grey box is the one to land, and it is deliberately in red. Both papers claim the generative component contributes something. The audits raise concrete doubts about whether it does in the released code. That is the thread running through the second half of the deck.

If asked why not compare GAN to GAN or VAE to VAE: because the research question is about the two families, not about two implementations of one family.`);
}

// =====================================================================
// SLIDE 8 — Why these two papers
// =====================================================================
{
  const s = p.addSlide();
  titleBar(s, "Why These Two Papers?", "Selection rationale");

  const crit = [
    ["Same task, same input level", "Both formulate DTA as regression on continuous affinity, from sequence-level input only — SMILES plus amino-acid string.", TEAL],
    ["Different generative families", "One adversarial (DCGAN), one latent-variable (VAE). That contrast is the variable of interest.", VIOLET],
    ["Author-released source code", "Both papers cite a public repository, so the implementation can be inspected rather than guessed at.", INK2],
    ["Peer-reviewed venues", "BMC Genomics (2024) and IEEE TPAMI (2022) — the claims are on the record and citable.", INK2],
    ["Self-contained implementations", "Single-language codebases with bundled data; no external services, so data-level checks are possible.", INK2],
    ["Feasible scope for an MSc", "Two codebases of a few thousand lines each — auditable in depth within the time available.", MUTED]
  ];
  const bw = 6.2, gap = 0.35, bh = 1.24;
  crit.forEach((c, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = M + col * (bw + gap), y = 1.34 + row * (bh + 0.22);
    card(s, x, y, bw, bh, TINT);
    dot(s, x + 0.2, y + 0.2, 0.46, c[2], String(i + 1), 11);
    s.addText(c[0], { x: x + 0.82, y: y + 0.14, w: bw - 1.05, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: INK, margin: 0 });
    s.addText(c[1], { x: x + 0.82, y: y + 0.44, w: bw - 1.05, h: 0.68, fontFace: BODY, fontSize: 9.5, color: INK2, margin: 0 });
  });

  s.addShape(p.ShapeType.roundRect, { x: M, y: 5.72, w: CW, h: 1.0, rectRadius: 0.07, fill: { color: "FBF3E4" }, line: { color: AMBER, width: 1.1 } });
  chip(s, M + 0.24, 5.9, "VERIFIED", AMBER, 1.0);
  s.addText("Found during the audit, not used as a selection criterion: the two papers share no dataset. DCGAN-DTA uses BindingDB and PDBbind; Co-VAE uses Davis and KIBA. Any head-to-head comparison therefore needs a common protocol built deliberately — it cannot be read off the published tables.", {
    x: M + 1.4, y: 5.82, w: CW - 1.7, h: 0.8, fontFace: BODY, fontSize: 10.5, color: "6B4A12", valign: "middle", margin: 0
  });

  s.addNotes(
`Explain the selection honestly: the first five criteria are why I picked these papers up front, and the sixth is practical scope.

Criteria one and two are the real scientific reasons. Same task and same input level means the comparison is not confounded by representation. Different generative families means there is actually something to compare.

Criteria three to five are what made an audit possible at all — released code, a citable venue, and a codebase small enough to read line by line with bundled data I can check.

The amber box at the bottom is a finding, not a criterion, and I want to be clear about that distinction. I discovered during the audit that the two papers have no dataset in common. That is a problem for the comparison and it directly shapes the next phase: I cannot just line up their published numbers, I have to build a shared evaluation protocol. Expect the supervisor to ask what that protocol should be — my current answer is that it is an open design decision and one of the things I would like to discuss today.`);
}

// =====================================================================
// SLIDE 9 — DCGAN-DTA paper overview
// =====================================================================
{
  const s = p.addSlide();
  titleBar(s, "DCGAN-DTA — What the Paper Proposes", "Paper audit · all claims [PAPER]");

  // architecture strip
  s.addShape(p.ShapeType.roundRect, { x: M, y: 1.3, w: 7.45, h: 2.72, rectRadius: 0.08, fill: { color: W }, line: { color: TEAL, width: 1.3 } });
  s.addText("Architecture (simplified, adapted from Fig. 1, p. 5)", { x: M + 0.22, y: 1.42, w: 7.0, h: 0.28, fontFace: BODY, fontSize: 11, bold: true, color: TEAL, margin: 0 });

  const steps = [
    ["Label encoding\n+ embedding", TINT],
    ["DCGAN\npretraining", TEAL],
    ["Discriminator\nfeature reuse", TEAL],
    ["CNN block\n3 conv + max-pool", TINT],
    ["Add\nmerge", TINT],
    ["FC block\n→ affinity", INK2]
  ];
  const sw = 1.12, sg = 0.11;
  steps.forEach((st, i) => {
    const x = M + 0.26 + i * (sw + sg);
    const dark = st[1] === TEAL || st[1] === INK2;
    s.addShape(p.ShapeType.roundRect, { x, y: 1.82, w: sw, h: 0.9, rectRadius: 0.06, fill: { color: st[1] }, line: { color: dark ? st[1] : TINT2, width: 0.9 } });
    s.addText(st[0], { x, y: 1.82, w: sw, h: 0.9, fontFace: BODY, fontSize: 8, bold: true, color: dark ? W : INK, align: "center", valign: "middle", margin: 0 });
    if (i < steps.length - 1) arrowRight(s, x + sw + 0.005, 2.27, sg - 0.01, MUTED);
  });
  s.addText("Two DCGANs (drug + protein) pretrained on unlabelled sequence data; the learned model is reused for feature extraction. Three variants A / B / C differ in protein encoding and whether a protein DCGAN is used.", {
    x: M + 0.26, y: 2.82, w: 7.0, h: 0.62, fontFace: BODY, fontSize: 9.5, color: INK2, margin: 0
  });
  s.addText("Generator: 3×Conv1DTranspose, filters 128/64/1, kernel 3, ReLU/ReLU/tanh, no BatchNorm.   Discriminator: 5×Conv1D, filters 4/8/16/32/64, kernel 3, ReLU → Flatten → Dense(1), tanh.", {
    x: M + 0.26, y: 3.44, w: 7.0, h: 0.46, fontFace: BODY, fontSize: 8.5, color: MUTED, margin: 0
  });

  // right spec
  const rx = 8.32, rw = 4.45;
  const spec = [
    ["Datasets", "BindingDB 9864×1088, 42,203 pairs\nPDBbind refined v2020 4231×1606, 5,014"],
    ["Representation", "Drug: SMILES → label encoding → embedding\nTarget: label encoding, or BLOSUM 25-dim"],
    ["Training", "300 epochs, batch 256, Adam lr 0.001,\ndropout 0.25, early stopping"],
    ["Evaluation", "5-fold CV; CI, MSE, AUPR (threshold 7), r²m"],
    ["Headline result", "Best CI on both datasets: 0.866 BindingDB,\n0.774 PDBbind"]
  ];
  let sy = 1.3;
  spec.forEach(([h, b]) => {
    card(s, rx, sy, rw, 1.02, TINT);
    s.addText(h, { x: rx + 0.18, y: sy + 0.1, w: rw - 0.36, h: 0.26, fontFace: BODY, fontSize: 11, bold: true, color: INK, margin: 0 });
    s.addText(b, { x: rx + 0.18, y: sy + 0.38, w: rw - 0.36, h: 0.56, fontFace: BODY, fontSize: 9, color: INK2, margin: 0 });
    sy += 1.1;
  });

  s.addShape(p.ShapeType.roundRect, { x: M, y: 4.3, w: 7.45, h: 1.42, rectRadius: 0.07, fill: { color: TINT }, line: { color: TINT2, width: 0.75 } });
  s.addText([
    { text: "What the paper leaves unspecified.  ", options: { bold: true, color: INK } },
    { text: "All four numbered equations are evaluation metrics. There is no training loss, no adversarial objective, and no statement of which discriminator layers transfer or whether they are frozen — the mechanism the method is named for is never written down.", options: { color: INK2 } }
  ], { x: M + 0.24, y: 4.3, w: 7.0, h: 1.42, fontFace: BODY, fontSize: 11, valign: "middle", margin: 0 });

  s.addText("Kalemati M., Zamani Emani M., Koohi S., “DCGAN-DTA: Predicting drug-target binding affinity with deep convolutional generative adversarial networks,” BMC Genomics 25:411 (2024). DOI 10.1186/s12864-024-10326-x. Architecture diagram redrawn from Fig. 1; original figure not reproduced.", {
    x: M, y: 5.9, w: CW, h: 0.5, fontFace: BODY, fontSize: 8.5, color: MUTED, italic: true, margin: 0
  });

  s.addNotes(
`Everything on this slide is what the paper says. I have not mixed in any code findings — that is the next slide, and keeping them apart is the whole method of the audit.

Walk the architecture strip left to right: encode, pretrain a DCGAN, reuse the discriminator's features, run a CNN block, merge the drug and protein branches with an add layer, and predict. The three variants differ only in how the protein side is encoded.

The specs on the right are the reproducible parts — datasets with exact counts, a clear training configuration, four defined metrics.

The grey box is the finding I want to land. The paper has four numbered equations and every one of them is an evaluation metric. There is no loss function anywhere in the paper, no adversarial objective, and no statement of what actually gets transferred from the GAN into the predictor. For a method named after its GAN, that is a substantial gap — and it means the code is the only place that information exists.

Note the redrawn figure and the citation. I did not reproduce their figure.`);
}

// =====================================================================
// SLIDE 10 — DCGAN-DTA code audit
// =====================================================================
{
  const s = p.addSlide();
  titleBar(s, "DCGAN-DTA — What the Code Actually Contains", "Code audit · read-only inspection");

  // pipeline
  const stages = [
    ["Data", "MATCH", GREEN],
    ["Preprocessing", "MATCH", GREEN],
    ["Representation", "PARTIAL", AMBER],
    ["Model", "MISMATCH", CRIM],
    ["Training", "PARTIAL", AMBER],
    ["Evaluation", "MISMATCH", CRIM]
  ];
  const pw = 1.98, pg = 0.14;
  stages.forEach((st, i) => {
    const x = M + i * (pw + pg);
    card(s, x, 1.3, pw, 0.86, W);
    s.addShape(p.ShapeType.roundRect, { x, y: 1.3, w: pw, h: 0.86, rectRadius: 0.06, fill: { color: W }, line: { color: st[2], width: 1.2 } });
    s.addText(st[0], { x, y: 1.4, w: pw, h: 0.28, fontFace: BODY, fontSize: 11, bold: true, color: INK, align: "center", margin: 0 });
    s.addText(st[1], { x, y: 1.7, w: pw, h: 0.26, fontFace: BODY, fontSize: 9, bold: true, color: st[2], align: "center", margin: 0 });
    if (i < stages.length - 1) arrowRight(s, x + pw + 0.01, 1.73, pg - 0.02, MUTED);
  });

  const finds = [
    ["VERIFIED", CRIM, "Transferred layer does not fit the branch it is used in.", "gan.layers[-3] is a Conv1D built on 1 input channel, then applied to a 32-channel embedding — in the drug branch of all three variants."],
    ["VERIFIED", CRIM, "Three of five experiment families have no code.", "No cold-start/logP split, no adversarial-control shuffling, no baselines, no concatenation ablation — Figs. 5–8 have no code path."],
    ["VERIFIED", CRIM, "Reported CI is not the paper’s equation.", "The exact CI in emetrics.py is never imported. The reported value is a Keras per-batch metric averaged across batches."],
    ["VERIFIED", AMBER, "BatchNorm is present in all three generators.", "The paper states explicitly, with justification, that BatchNorm was removed. It is absent from the discriminators only."],
    ["VERIFIED", AMBER, "Model selection uses the test fold.", "The CV routine runs twice; the reported pass early-stops and picks the epoch on the test fold."],
    ["VERIFIED", GREEN, "Dataset and folds match the paper exactly.", "Both matrices and all fold index sets reproduce the published counts — 42,203 and 5,014 pairs."]
  ];
  let fy = 2.38;
  finds.forEach((f) => {
    card(s, M, fy, CW, 0.66, TINT);
    chip(s, M + 0.16, fy + 0.2, f[0], f[1], 0.82);
    s.addText(f[2], { x: M + 1.12, y: fy + 0.06, w: 5.0, h: 0.28, fontFace: BODY, fontSize: 10.5, bold: true, color: INK, margin: 0 });
    s.addText(f[3], { x: M + 1.12, y: fy + 0.33, w: 11.0, h: 0.3, fontFace: BODY, fontSize: 8.8, color: INK2, margin: 0 });
    fy += 0.74;
  });

  s.addShape(p.ShapeType.roundRect, { x: M, y: 6.84, w: CW, h: 0.5, rectRadius: 0.07, fill: { color: INK }, line: { color: INK, width: 0 } });
  s.addText([
    { text: "Traceability verdict: RED.   ", options: { bold: true, color: "FFB4A2" } },
    { text: "Also resolved from code — FC widths are 1024/1024/512, matching the figure and contradicting the paper text and supplementary table.", options: { color: "C8D4E6" } }
  ], { x: M + 0.24, y: 6.84, w: CW - 0.48, h: 0.5, fontFace: BODY, fontSize: 10.5, valign: "middle", margin: 0 });

  s.addNotes(
`This is where the deck turns. Everything here came from reading the repository, and every line is tagged VERIFIED because I traced it in the source or confirmed it against the bundled data files.

Start with the pipeline strip so the supervisor can see it is not uniformly bad. Data and preprocessing match the paper exactly — I loaded both affinity matrices and the fold files and confirmed the published counts to the pair. That is a genuine positive and I should say so.

Then the problems. The first one is the most serious: the layer the method transfers from the GAN into the predictor was built for a one-channel input and is then applied to a thirty-two-channel embedding. That is the method's namesake mechanism and, as written, the shapes do not line up. I have not executed it, so I am careful to say the shapes are verified and the consequence is inferred — that distinction matters and Phase 4 has to settle it.

Second: three of the five experiment families in the paper simply have no code. Four of the eight figures cannot be produced from this repository.

Third: the headline metric is not the metric the paper defines.

The green row matters for balance. This is not a bad repository across the board; it is a repository whose data handling is solid and whose model and evaluation diverge from the paper.

If asked "is this a bug or a difference?", my answer is that I am reporting behaviour, not intent, and Phase 4 is designed to settle the one item that is still inferred.`);
}

// =====================================================================
// SLIDE 11 — Co-VAE paper overview
// =====================================================================
{
  const s = p.addSlide();
  titleBar(s, "Co-VAE — What the Paper Proposes", "Paper audit · all claims [PAPER]");

  s.addShape(p.ShapeType.roundRect, { x: M, y: 1.3, w: 7.45, h: 2.9, rectRadius: 0.08, fill: { color: W }, line: { color: VIOLET, width: 1.3 } });
  s.addText("Architecture (simplified, adapted from Fig. 4, p. 8866)", { x: M + 0.22, y: 1.38, w: 7.0, h: 0.26, fontFace: BODY, fontSize: 11, bold: true, color: VIOLET, margin: 0 });

  // two VAE lanes + reg block.  Lane content is inset by 0.4" to leave a clear
  // left gutter (at gx) for the drug-latent bus, so no connector crosses a box.
  const lx = M + 0.3;
  const gx = lx + 0.15;        // gutter x for the drug bus
  const laneX = lx + 0.4;      // lane content start
  const DRUG_Y = 1.95, TGT_Y = 2.75, LH = 0.56;

  [["Drug SMILES", DRUG_Y, TEAL], ["Target sequence", TGT_Y, VIOLET]].forEach(([lbl, ly, col]) => {
    s.addShape(p.ShapeType.roundRect, { x: laneX, y: ly, w: 1.5, h: LH, rectRadius: 0.06, fill: { color: TINT }, line: { color: TINT2, width: 0.9 } });
    s.addText(lbl, { x: laneX, y: ly, w: 1.5, h: LH, fontFace: BODY, fontSize: 8.5, bold: true, color: INK, align: "center", valign: "middle", margin: 0 });
    arrowRight(s, laneX + 1.52, ly + 0.28, 0.2, col);
    s.addShape(p.ShapeType.roundRect, { x: laneX + 1.76, y: ly, w: 1.25, h: LH, rectRadius: 0.06, fill: { color: col }, line: { color: col, width: 0 } });
    s.addText("Encoder\nGatedCNN", { x: laneX + 1.76, y: ly, w: 1.25, h: LH, fontFace: BODY, fontSize: 7.5, bold: true, color: W, align: "center", valign: "middle", margin: 0 });
    arrowRight(s, laneX + 3.03, ly + 0.28, 0.18, col);
    s.addShape(p.ShapeType.roundRect, { x: laneX + 3.25, y: ly, w: 0.82, h: LH, rectRadius: 0.06, fill: { color: TINT }, line: { color: col, width: 0.9 } });
    s.addText("μ, σ → z", { x: laneX + 3.25, y: ly, w: 0.82, h: LH, fontFace: BODY, fontSize: 8, bold: true, color: INK, align: "center", valign: "middle", margin: 0 });
    arrowRight(s, laneX + 4.09, ly + 0.28, 0.18, col);
    s.addShape(p.ShapeType.roundRect, { x: laneX + 4.31, y: ly, w: 1.25, h: LH, rectRadius: 0.06, fill: { color: col }, line: { color: col, width: 0 } });
    s.addText("Decoder\ndeconv", { x: laneX + 4.31, y: ly, w: 1.25, h: LH, fontFace: BODY, fontSize: 7.5, bold: true, color: W, align: "center", valign: "middle", margin: 0 });
    s.addText("reconstruct", { x: laneX + 5.62, y: ly, w: 1.0, h: LH, fontFace: BODY, fontSize: 7.5, color: MUTED, valign: "middle", margin: 0 });
  });

  // Reg block
  const regX = laneX + 2.85, regY = 3.55, regW = 2.5, regH = 0.5;
  s.addShape(p.ShapeType.roundRect, { x: regX, y: regY, w: regW, h: regH, rectRadius: 0.06, fill: { color: INK }, line: { color: INK, width: 0 } });
  s.addText("Reg block  →  affinity", { x: regX, y: regY, w: regW, h: regH, fontFace: BODY, fontSize: 9, bold: true, color: W, align: "center", valign: "middle", margin: 0 });

  // target latent -> reg block: short straight drop
  s.addShape(p.ShapeType.line, { x: laneX + 3.66, y: TGT_Y + LH, w: 0, h: regY - (TGT_Y + LH), line: { color: MUTED, width: 1.2, endArrowType: "triangle" } });

  // drug latent -> reg block: routed up, along the left gutter, then in from the left
  const busTop = DRUG_Y - 0.2, busMidY = regY + regH / 2;
  s.addShape(p.ShapeType.line, { x: laneX + 3.66, y: busTop, w: 0, h: DRUG_Y - busTop, line: { color: MUTED, width: 1.2 } });
  s.addShape(p.ShapeType.line, { x: gx, y: busTop, w: laneX + 3.66 - gx, h: 0, line: { color: MUTED, width: 1.2 } });
  s.addShape(p.ShapeType.line, { x: gx, y: busTop, w: 0, h: busMidY - busTop, line: { color: MUTED, width: 1.2 } });
  s.addShape(p.ShapeType.line, { x: gx, y: busMidY, w: regX - gx, h: 0, line: { color: MUTED, width: 1.2, endArrowType: "triangle" } });

  const rx = 8.32, rw = 4.45;
  const spec = [
    ["Datasets", "Davis 68×442 (pKd)\nKIBA 2111×229, density 24.4%"],
    ["Representation", "Both sides: label encoding → embedding (128)\n64 SMILES chars, 25 protein chars"],
    ["Objective (Thm 2.1, Eq. 5)", "L = L_DrugVAE + L_TargetVAE + L_CoREG,\nmaximised; co-regularisation weight λ ∈ {−3, −5}"],
    ["Protocol", "Six folds (5 train / 1 test); new-drug and\nnew-target settings; ten random repetitions"],
    ["Evaluation", "CI, MSE, MAE, r²m; AUC on KIBA only\n(threshold sweep 10.5–12.5)"]
  ];
  let sy = 1.3;
  spec.forEach(([h, b]) => {
    card(s, rx, sy, rw, 1.02, TINT);
    s.addText(h, { x: rx + 0.18, y: sy + 0.1, w: rw - 0.36, h: 0.26, fontFace: BODY, fontSize: 11, bold: true, color: INK, margin: 0 });
    s.addText(b, { x: rx + 0.18, y: sy + 0.38, w: rw - 0.36, h: 0.56, fontFace: BODY, fontSize: 9, color: INK2, margin: 0 });
    sy += 1.1;
  });

  s.addShape(p.ShapeType.roundRect, { x: M, y: 4.3, w: 7.45, h: 1.42, rectRadius: 0.07, fill: { color: TINT }, line: { color: TINT2, width: 0.75 } });
  s.addText([
    { text: "Strong where DCGAN-DTA was weak.  ", options: { bold: true, color: INK } },
    { text: "The variational objective is derived formally, with priors, posterior family, reparameterisation and KL all stated. But the latent dimension is never given, λ’s sign is ambiguous, and a validation partition appears only in a figure caption — never in the protocol text.", options: { color: INK2 } }
  ], { x: M + 0.24, y: 4.3, w: 7.0, h: 1.42, fontFace: BODY, fontSize: 11, valign: "middle", margin: 0 });

  s.addText("Li T., Zhao X.-M., Li L., “Co-VAE: Drug-Target Binding Affinity Prediction by Co-Regularized Variational Autoencoders,” IEEE TPAMI 44(12):8861–8873 (2022). DOI 10.1109/TPAMI.2021.3120428. Architecture diagram redrawn from Fig. 4; original figure not reproduced.", {
    x: M, y: 5.9, w: CW, h: 0.5, fontFace: BODY, fontSize: 8.5, color: MUTED, italic: true, margin: 0
  });

  s.addNotes(
`Same structure as the DCGAN-DTA paper slide, so the supervisor can compare like with like. Again, everything here is paper-side only.

The architecture is two parallel VAEs — one over the drug SMILES, one over the target sequence — each with a gated-CNN encoder, a latent code, and a deconvolutional decoder. The regression block takes both latent codes and predicts the affinity. That regression term is what couples the two VAEs, which is what "co-regularised" means.

Point out the contrast with the previous paper. Where DCGAN-DTA had no objective at all, Co-VAE derives one formally as a theorem, with an explicit lower bound on the joint likelihood. This is a TPAMI paper and it shows.

But the grey box is important: it is not complete either. The latent dimension is never stated, so you literally cannot instantiate the model from the paper. Lambda's sign does not work out as printed. And a validation set appears in the caption of Figure 6 but nowhere in the protocol description. Those three gaps are exactly what the code audit had to resolve.`);
}

// =====================================================================
// SLIDE 12 — Co-VAE code audit
// =====================================================================
{
  const s = p.addSlide();
  titleBar(s, "Co-VAE — What the Code Actually Contains", "Code audit · read-only inspection");

  const stages = [
    ["Data", "PARTIAL", AMBER],
    ["Preprocessing", "MISMATCH", CRIM],
    ["Representation", "MATCH", GREEN],
    ["Model", "PARTIAL", AMBER],
    ["Training", "PARTIAL", AMBER],
    ["Evaluation", "MISMATCH", CRIM]
  ];
  const pw = 1.98, pg = 0.14;
  stages.forEach((st, i) => {
    const x = M + i * (pw + pg);
    s.addShape(p.ShapeType.roundRect, { x, y: 1.3, w: pw, h: 0.86, rectRadius: 0.06, fill: { color: W }, line: { color: st[2], width: 1.2 } });
    s.addText(st[0], { x, y: 1.4, w: pw, h: 0.28, fontFace: BODY, fontSize: 11, bold: true, color: INK, align: "center", margin: 0 });
    s.addText(st[1], { x, y: 1.7, w: pw, h: 0.26, fontFace: BODY, fontSize: 9, bold: true, color: st[2], align: "center", margin: 0 });
    if (i < stages.length - 1) arrowRight(s, x + pw + 0.01, 1.73, pg - 0.02, MUTED);
  });

  const finds = [
    ["VERIFIED", CRIM, "The KIBA matrix built at runtime is not the published one.", "Paper: 2111×229. Code produces 1954×217 with 109,296 pairs — an undocumented length filter runs on top of the already-filtered bundled files."],
    ["VERIFIED", CRIM, "The paper’s KIBA affinity filter never runs.", "Runtime affinities still span 0.0–17.2, contradicting “removed pairs with affinity < 10” and the paper’s own Fig. 5."],
    ["VERIFIED", CRIM, "Hyperparameters, early stopping and reporting all use the test fold.", "val_sets is constructed and then discarded; test_sets is passed to both the grid search and the final evaluation."],
    ["VERIFIED", AMBER, "One repetition, not ten; MAE never implemented.", "for i in range(1). MAE appears for every method in the paper’s Table 2 but has no implementation anywhere."],
    ["VERIFIED", AMBER, "ε sampled at the wrong scale; average pooling, not max.", "normal_(0, 0.1) against a KL derived for N(0, I); AdaptiveAvgPool1d where the paper and figure say max-pooling."],
    ["VERIFIED", GREEN, "Code resolves what the paper left open.", "λ is used as 10^λ, so the sign paradox does not occur; latent dimension is 96; FC widths are 96 / 192→1024→512→1."]
  ];
  let fy = 2.38;
  finds.forEach((f) => {
    card(s, M, fy, CW, 0.66, TINT);
    chip(s, M + 0.16, fy + 0.2, f[0], f[1], 0.82);
    s.addText(f[2], { x: M + 1.12, y: fy + 0.06, w: 6.4, h: 0.28, fontFace: BODY, fontSize: 10.5, bold: true, color: INK, margin: 0 });
    s.addText(f[3], { x: M + 1.12, y: fy + 0.33, w: 11.0, h: 0.3, fontFace: BODY, fontSize: 8.8, color: INK2, margin: 0 });
    fy += 0.74;
  });

  s.addShape(p.ShapeType.roundRect, { x: M, y: 6.84, w: CW, h: 0.5, rectRadius: 0.07, fill: { color: INK }, line: { color: INK, width: 0 } });
  s.addText([
    { text: "Traceability verdict: RED.   ", options: { bold: true, color: "FFB4A2" } },
    { text: "Three confirmed execution blockers: random.sample on a set (Python ≥ 3.11), --lamda omitted, and CUDA with no CPU fallback. Only Co-VAE is implemented — no baselines, no drug generation.", options: { color: "C8D4E6" } }
  ], { x: M + 0.24, y: 6.84, w: CW - 0.48, h: 0.5, fontFace: BODY, fontSize: 10.5, valign: "middle", margin: 0 });

  s.addNotes(
`Same framework as slide 10 so the two audits are directly comparable.

The single most important finding is the first one. The KIBA dataset the code actually builds is not the dataset the paper reports — it is 1954 by 217 instead of 2111 by 229. I traced why: the bundled data files already contain the paper's filtered 2111 by 229, so that step happened offline, and then the code applies a second length filter on top that nobody documents. That is a concrete, explainable divergence and it affects every KIBA number in the paper.

Second finding: the affinity filter the paper describes does not run at all. I confirmed this by loading the matrix — values below ten are still there.

Third: the same model-selection problem as DCGAN-DTA. The code builds a validation set and then throws it away, passing the test fold to the grid search instead. That the same issue appears independently in two unrelated repositories is itself worth noting.

The green row is genuinely positive and I should not skip it. The code answers three questions the paper left open — lambda is used as ten to the power of lambda, so the sign paradox I flagged in the paper audit simply does not arise; the latent dimension is 96; and I now have every FC width. Reading the code made the paper more reproducible, not less.

Finish on the blockers. As shipped, this code cannot run on a modern Python, cannot run without a specific flag that no README documents, and cannot run without a GPU.`);
}

// =====================================================================
// SLIDE 13 — Comparison
// =====================================================================
{
  const s = p.addSlide();
  titleBar(s, "DCGAN-DTA vs Co-VAE — Audited Comparison", "Populated only from audited evidence");

  const rows = [
    ["Task", "DTA regression on continuous affinity", "DTA regression on continuous affinity"],
    ["Model family", "GAN (deep convolutional, adversarial)", "VAE (co-regularised, latent-variable)"],
    ["Main architecture", "2 DCGANs → discriminator feature reuse → CNN → Add → FC", "2 VAEs (GatedCNN enc / deconv dec) + regression block"],
    ["Drug representation", "SMILES → label encoding → embedding (32)", "SMILES → label encoding → embedding (128)"],
    ["Target representation", "Label encoding (A) or BLOSUM — 20-dim in code, 25 claimed", "Label encoding → embedding (128)"],
    ["Datasets", "BindingDB, PDBbind — counts match paper exactly", "Davis matches; KIBA does not (1954×217 vs 2111×229)"],
    ["Training strategy", "300 epochs, batch 256, Adam; GAN retrained per fold/grid point", "100 epochs, batch 256, Adam; single joint objective"],
    ["Evaluation", "CI (per-batch approx.), MSE, AUPR@7, r²m; folds from files", "CI, MSE, r²m, AUC@7/12.1; folds generated at runtime"],
    ["Code availability", "Public, runs are plausible; no execution blocker confirmed", "Public, but three confirmed execution blockers"],
    ["Reproducibility status", "RED — data solid, model and evaluation diverge", "RED — data diverges, objective live but mis-scaled"],
    ["Key audit finding", "Transferred layer channel mismatch; 3 of 5 experiment families absent", "Runtime KIBA ≠ published KIBA; test fold used for model selection"],
    ["Current status", "Paper + code audit complete; not executed", "Paper + code audit complete; not executed"]
  ];

  const c0 = 2.55, c1 = 4.84, c2 = 4.84;
  const hy = 1.28, rh = 0.435;

  s.addShape(p.ShapeType.rect, { x: M, y: hy, w: c0, h: 0.4, fill: { color: INK }, line: { color: INK, width: 0 } });
  s.addShape(p.ShapeType.rect, { x: M + c0, y: hy, w: c1, h: 0.4, fill: { color: TEAL }, line: { color: TEAL, width: 0 } });
  s.addShape(p.ShapeType.rect, { x: M + c0 + c1, y: hy, w: c2, h: 0.4, fill: { color: VIOLET }, line: { color: VIOLET, width: 0 } });
  s.addText("Dimension", { x: M + 0.12, y: hy, w: c0, h: 0.4, fontFace: BODY, fontSize: 10.5, bold: true, color: W, valign: "middle", margin: 0 });
  s.addText("DCGAN-DTA", { x: M + c0 + 0.12, y: hy, w: c1, h: 0.4, fontFace: BODY, fontSize: 10.5, bold: true, color: W, valign: "middle", margin: 0 });
  s.addText("Co-VAE", { x: M + c0 + c1 + 0.12, y: hy, w: c2, h: 0.4, fontFace: BODY, fontSize: 10.5, bold: true, color: W, valign: "middle", margin: 0 });

  rows.forEach((r, i) => {
    const y = hy + 0.4 + i * rh;
    const bg = i % 2 === 0 ? W : TINT;
    s.addShape(p.ShapeType.rect, { x: M, y, w: c0 + c1 + c2, h: rh, fill: { color: bg }, line: { color: TINT2, width: 0.5 } });
    s.addText(r[0], { x: M + 0.12, y, w: c0 - 0.2, h: rh, fontFace: BODY, fontSize: 9, bold: true, color: INK, valign: "middle", margin: 0 });
    s.addText(r[1], { x: M + c0 + 0.12, y, w: c1 - 0.2, h: rh, fontFace: BODY, fontSize: 8.5, color: INK2, valign: "middle", margin: 0 });
    s.addText(r[2], { x: M + c0 + c1 + 0.12, y, w: c2 - 0.2, h: rh, fontFace: BODY, fontSize: 8.5, color: INK2, valign: "middle", margin: 0 });
  });

  s.addText("Every cell is drawn from the Stage/Phase 2 paper audits and the Stage/Phase 3 code audits. “Not executed” is literal — no training run has been performed for either method.", {
    x: M, y: 7.0, w: CW, h: 0.32, fontFace: BODY, fontSize: 8.5, color: MUTED, italic: true, margin: 0
  });

  s.addNotes(
`Do not read this table aloud. Let the supervisor scan it and point at four rows.

Row one and two: the comparison is fair at the task level. Same problem, same kind of input, genuinely different model families. That is what makes the study worth doing.

The datasets row is where it gets awkward. DCGAN-DTA's data reproduces the paper exactly — I verified both matrices and every fold file. Co-VAE's Davis reproduces, but KIBA does not. So the two codebases are not equally trustworthy at the data level, and that asymmetry has to be handled in the comparison.

The code-availability row: DCGAN-DTA has no confirmed execution blocker, Co-VAE has three. So the practical effort to get each running is very different, and that shapes the Phase 4 plan.

The last row is the honest one. Both are audited, neither is executed. I want to be unambiguous that nothing in this deck is a performance claim.

If the supervisor asks which paper is better — that is not a question the audits can answer, and I should say so. They answer which paper is more faithfully implemented, which is a different thing.`);
}

// =====================================================================
// SLIDE 14 — What the audit revealed
// =====================================================================
{
  const s = p.addSlide();
  titleBar(s, "What the Code Audit Revealed", "Methodological lessons");

  const lessons = [
    ["Paper description ≠ implementation", "In both repositories the released code differs from the published method in ways a reader could not detect from the paper. DCGAN-DTA: BatchNorm present where the paper says removed. Co-VAE: average pooling where the paper says max.", CRIM],
    ["Dataset construction is not a detail", "Co-VAE’s runtime KIBA matrix is a different dataset from the published one — 1954×217 against 2111×229 — because of an undocumented filter. Every KIBA number depends on which matrix was used.", CRIM],
    ["Split handling decides what a number means", "Both repositories build a validation set and then evaluate and early-stop on the test fold instead. Two unrelated codebases, the same pattern. Reported metrics are therefore selection-optimistic.", CRIM],
    ["Papers under-specify the training objective", "DCGAN-DTA states no loss function at all — its four equations are all metrics. Co-VAE derives its objective formally but omits the latent dimension. Neither model can be built from its paper alone.", AMBER],
    ["Evaluation choices break comparability", "DCGAN-DTA’s reported CI is a per-batch approximation, not its own Equation 1. Comparing that number against a baseline’s global CI is not a like-for-like comparison.", AMBER],
    ["Reading code can also repair a paper", "The Co-VAE code resolved three gaps the paper left open — λ is applied as 10^λ, the latent dimension is 96, and all FC widths are recoverable. Audit is not only a fault-finding exercise.", GREEN]
  ];

  const bw = 6.2, gap = 0.35, bh = 1.58;
  lessons.forEach((l, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = M + col * (bw + gap), y = 1.32 + row * (bh + 0.22);
    s.addShape(p.ShapeType.roundRect, { x, y, w: bw, h: bh, rectRadius: 0.07, fill: { color: W }, line: { color: l[2], width: 1.15 } });
    dot(s, x + 0.2, y + 0.2, 0.42, l[2], String(i + 1), 11);
    s.addText(l[0], { x: x + 0.78, y: y + 0.16, w: bw - 1.0, h: 0.5, fontFace: BODY, fontSize: 11.5, bold: true, color: INK, margin: 0 });
    s.addText(l[1], { x: x + 0.24, y: y + 0.68, w: bw - 0.48, h: 0.82, fontFace: BODY, fontSize: 9, color: INK2, margin: 0 });
  });

  s.addNotes(
`This slide is the intellectual payoff of the audit phase, and it is probably the one the supervisor will engage with most. These are transferable lessons, not a list of bugs.

Lesson three is the one I would foreground. Two completely unrelated codebases, different authors, different countries, different frameworks, different years — and both build a validation set and then quietly evaluate on the test fold. That is not a coincidence about these two papers; it says something about how the field's reference implementations get written. It also means published numbers in this area are probably selection-optimistic more often than the papers admit.

Lesson two is the most consequential for my own next step. If the dataset the code builds is not the dataset the paper reports, then reproducing the paper's number and reproducing the code's number are two different goals, and I have to decide which one I am pursuing.

Lesson six is deliberately last and deliberately green. Auditing is not only about finding faults. Reading the Co-VAE code told me three things the paper does not contain, and it dissolved an apparent contradiction I had flagged in the paper audit. That is a genuinely useful outcome and it argues for doing the code audit before, not after, attempting reproduction.

If there is time, this is a good slide to pause on and invite discussion.`);
}

// =====================================================================
// SLIDE 15 — Current status
// =====================================================================
{
  const s = p.addSlide();
  titleBar(s, "Current Research Status", "Honest status board");

  const cols = [
    ["COMPLETED", GREEN, [
      "DTA problem definition and scoping",
      "Literature / survey analysis",
      "Selection of the two papers",
      "DCGAN-DTA paper forensic audit",
      "Co-VAE paper forensic audit",
      "Repository reconnaissance (both)",
      "DCGAN-DTA code traceability audit",
      "Co-VAE code traceability audit",
      "Dataset + fold verification (4 datasets)",
      "Architecture and objective tracing"
    ]],
    ["OPEN / TO VERIFY", AMBER, [
      "Do DCGAN-DTA models construct at all?",
      "Co-VAE in-place .exp_() effect on the KL",
      "Provenance of DCGAN-DTA pretraining corpora",
      "Whether published baselines came from either repo",
      "Can the published 2111×229 KIBA be recovered?",
      "Size of the CI discrepancy in DCGAN-DTA",
      "Viable TensorFlow / Keras version window",
      "Effect of the test-fold selection on reported margins"
    ]],
    ["PLANNED", TEAL, [
      "Clear the three Co-VAE execution blockers",
      "Minimal construction / smoke tests",
      "Quantify metric and protocol deviations",
      "Design a common evaluation protocol",
      "Controlled comparative experiments",
      "Analysis and write-up"
    ]]
  ];

  const bw = 4.05, gap = 0.29;
  cols.forEach((c, i) => {
    const x = M + i * (bw + gap);
    s.addShape(p.ShapeType.roundRect, { x, y: 1.3, w: bw, h: 5.42, rectRadius: 0.08, fill: { color: TINT }, line: { color: TINT2, width: 0.75 } });
    s.addShape(p.ShapeType.roundRect, { x, y: 1.3, w: bw, h: 0.52, rectRadius: 0.08, fill: { color: c[1] }, line: { color: c[1], width: 0 } });
    s.addText(c[0], { x, y: 1.3, w: bw, h: 0.52, fontFace: BODY, fontSize: 12, bold: true, color: W, align: "center", valign: "middle", margin: 0, charSpacing: 0.8 });
    let iy = 2.0;
    c[2].forEach((t) => {
      dot(s, x + 0.22, iy + 0.055, 0.16, c[1]);
      s.addText(t, { x: x + 0.5, y: iy - 0.02, w: bw - 0.72, h: 0.44, fontFace: BODY, fontSize: 9.5, color: INK2, margin: 0 });
      iy += 0.47;
    });
  });

  s.addShape(p.ShapeType.roundRect, { x: M, y: 6.88, w: CW, h: 0.46, rectRadius: 0.07, fill: { color: "FBF3E4" }, line: { color: AMBER, width: 1 } });
  s.addText("Nothing in the PLANNED column has been started. Nothing in the OPEN column has been resolved. No training run has been executed for either method.", {
    x: M + 0.22, y: 6.88, w: CW - 0.44, h: 0.46, fontFace: BODY, fontSize: 10.5, color: "6B4A12", valign: "middle", margin: 0
  });

  s.addNotes(
`This is the accountability slide. Be precise and do not blur the columns.

The completed column is substantial and I should own it — two full paper audits, two full code audits, plus data and fold verification on all four datasets. That is real work and it produced findings that change the plan.

The open column is what I genuinely do not know. The first item is the one I care about most: I have verified that the shapes in DCGAN-DTA's transferred layer do not match, but I have not executed it, so I do not yet know whether the model constructs at all. If it does not, then a large part of what the paper reports needs a different explanation, and that is a significant claim I am not yet in a position to make.

The planned column has not been started. The amber bar says that explicitly because the easiest way to lose a supervisor's trust is to let planned work drift into sounding completed.

If asked how long the open column takes: the diagnostics are small and cheap, but they depend on getting an environment with a GPU and the right framework versions, which is the practical constraint.`);
}

// =====================================================================
// SLIDE 16 — Roadmap & next step
// =====================================================================
{
  const s = p.addSlide();
  s.background = { color: INK };

  s.addText("RESEARCH ROADMAP", {
    x: M, y: 0.42, w: CW, h: 0.26, fontFace: BODY, fontSize: 11, bold: true, color: TEAL, charSpacing: 1.6, margin: 0
  });
  s.addText("Where the project stands, and what comes next", {
    x: M, y: 0.68, w: CW, h: 0.5, fontFace: HEAD, fontSize: 28, bold: true, color: W, margin: 0
  });

  const steps = [
    ["Literature review", true], ["Problem definition", true], ["Paper selection", true],
    ["Paper analysis", true], ["Code audit", true],
    ["Experimental verification", false], ["Common evaluation protocol", false],
    ["Comparative experiments", false], ["Analysis", false], ["Thesis results", false]
  ];

  const sw = 2.35, sg = 0.22, sh = 0.72;
  steps.forEach((st, i) => {
    const row = Math.floor(i / 5), col = i % 5;
    const x = M + col * (sw + sg);
    const y = 1.62 + row * (sh + 1.0);
    const done = st[1];
    s.addShape(p.ShapeType.roundRect, {
      x, y, w: sw, h: sh, rectRadius: 0.07,
      fill: { color: done ? TEAL : "22304F" },
      line: { color: done ? TEAL : "3E5378", width: 1.1 }
    });
    s.addText(st[0], {
      x: x + 0.08, y, w: sw - 0.16, h: sh, fontFace: BODY, fontSize: 10,
      bold: true, color: done ? W : "9FB3D1", align: "center", valign: "middle", margin: 0
    });
    if (done) {
      s.addText("✓", { x: x + sw - 0.32, y: y + 0.04, w: 0.26, h: 0.26, fontFace: BODY, fontSize: 11, bold: true, color: W, align: "center", margin: 0 });
    }
    if (col < 4) arrowRight(s, x + sw + 0.01, y + sh / 2, sg - 0.02, "5C7099");
  });
  // wrap arrow (routed inside the gap between the two rows)
  const rr1bot = 1.62 + sh, rr2top = 1.62 + sh + 1.0, rmid = rr1bot + 0.33;
  s.addShape(p.ShapeType.line, { x: M + 4 * (sw + sg) + sw / 2, y: rr1bot, w: 0, h: rmid - rr1bot, line: { color: "5C7099", width: 1.4 } });
  s.addShape(p.ShapeType.line, { x: M + sw / 2, y: rmid, w: 4 * (sw + sg), h: 0, line: { color: "5C7099", width: 1.4 } });
  s.addShape(p.ShapeType.line, { x: M + sw / 2, y: rmid, w: 0, h: rr2top - rmid, line: { color: "5C7099", width: 1.4, endArrowType: "triangle" } });

  // current position marker — sits in the inter-row gap, clear of the drop arrow
  s.addShape(p.ShapeType.roundRect, { x: 2.15, y: 2.82, w: 2.95, h: 0.42, rectRadius: 0.2, fill: { color: AMBER }, line: { color: AMBER, width: 0 } });
  s.addText("CURRENT POSITION", { x: 2.15, y: 2.82, w: 2.95, h: 0.42, fontFace: BODY, fontSize: 10, bold: true, color: "3A2708", align: "center", valign: "middle", margin: 0 });

  // research question
  s.addShape(p.ShapeType.roundRect, { x: M, y: 4.92, w: CW, h: 1.52, rectRadius: 0.09, fill: { color: "1E3157" }, line: { color: TEAL, width: 1.3 } });
  s.addText("THE NEXT RESEARCH QUESTION", {
    x: M + 0.3, y: 5.08, w: CW - 0.6, h: 0.26, fontFace: BODY, fontSize: 10, bold: true, color: TEAL, charSpacing: 1.4, margin: 0
  });
  s.addText("“How do GAN-based and VAE-based approaches compare for drug–target affinity prediction under a controlled and reproducible evaluation protocol?”", {
    x: M + 0.3, y: 5.34, w: CW - 0.6, h: 0.62, fontFace: HEAD, fontSize: 17, italic: true, color: W, margin: 0
  });
  s.addText("Open question, not an established result. The audits establish that the published numbers cannot answer it — the two papers share no dataset, and in both repositories the reported metrics are selection-optimistic.", {
    x: M + 0.3, y: 6.02, w: CW - 0.6, h: 0.34, fontFace: BODY, fontSize: 10, color: "9FB3D1", margin: 0
  });

  s.addText("Immediate next step: clear the three Co-VAE execution blockers and run construction-only smoke tests on both codebases — before any comparative experiment is attempted.", {
    x: M, y: 6.72, w: CW, h: 0.4, fontFace: BODY, fontSize: 11, bold: true, color: AMBER, margin: 0
  });

  s.addNotes(
`Close on this slide and leave it up for the discussion.

The roadmap is deliberately literal. Five steps are done, five are not, and the amber marker sits exactly between the code audit and experimental verification. That is where I am.

State the research question as a question. It is what I want to answer, not something I have answered. And explain why the audits mean I cannot shortcut to it: the two papers share no dataset, so their published tables cannot be compared directly; and in both repositories the reported numbers were produced with the test fold informing model selection, so they are not a clean baseline either. Between those two facts, a controlled protocol is not a nicety, it is the only way the question becomes answerable.

The last line is the concrete ask. I am not proposing to jump into comparative experiments. I want to clear three specific blockers and run construction-only smoke tests first, because if DCGAN-DTA does not construct, the whole comparison has to be reframed.

Two things I would like from the supervisor today: agreement on whether I am reproducing the papers' numbers or the code's behaviour, since the Co-VAE dataset divergence makes those different targets; and guidance on what a fair common evaluation protocol looks like given that the datasets do not overlap.`);
}

// ---------- write ----------
const OUT = process.argv[2] || "DTA-status.pptx";
p.writeFile({ fileName: OUT }).then(() => console.log("written:", OUT));
