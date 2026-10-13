/* Academic Advisor front end: reads the form, calls /api/advise, draws the plan.
   All text is inserted with textContent / createTextNode, never as HTML. */
(function () {
  "use strict";

  const OPT = window.ADVISOR_OPTIONS;
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const form = $("#form");
  const courseList = $("#course-list");
  const results = $("#results");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const COURSE_COLORS = [
    "#d6283c", "#2563eb", "#c98a12", "#168a5b", "#7c3aed", "#0891b2",
    "#db2777", "#65a30d", "#ea580c", "#475569", "#0d9488", "#9333ea",
  ];
  const FEATURE_NAMES = {
    level: "level", cgpa: "CGPA", last_gpa: "last semester GPA", avg_score: "course scores",
    credit_load: "credit load", avg_difficulty: "course difficulty", avg_confidence: "confidence",
    study_hours_day: "daily study hours", study_days_week: "study days", attendance_pct: "attendance",
    assignment_rate: "assignment habits", carryovers: "repeat courses", goal_gap: "distance to target",
  };

  /* ---------- tiny DOM helper ---------- */
  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "style") el.style.cssText = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }
  const num = (n, d = 2) => Number(n).toFixed(d);
  const hrs = (n) => `${Number(n)}`.replace(/\.0$/, "") + " h";

  /* ---------- course rows ---------- */
  function selectEl(field, options, value) {
    const el = h("select", { "data-f": field },
      options.map(([v, label]) => h("option", { value: v }, label)));
    el.value = String(value);
    return el;
  }

  function courseRow(data = {}) {
    const field = (cls, label, control) =>
      h("label", { class: `field ${cls}` }, h("span", { class: "lab" }, label), control, h("small", { class: "err" }));

    const code = h("input", { "data-f": "code", placeholder: "CSC 301", maxlength: 20, value: data.code || "" });
    const score = h("input", {
      "data-f": "score", type: "number", inputmode: "decimal", min: 0, max: 100, step: 0.5,
      placeholder: "Optional", value: data.score ?? "",
    });
    const repeat = h("input", { type: "checkbox", "data-f": "repeat" });
    repeat.checked = !!data.repeat;

    const row = h("div", { class: "course-row" },
      field("f-code", "Course code", code),
      field("f-credits", "Credit units", selectEl("credits", [1, 2, 3, 4, 5, 6].map((n) => [n, `${n} unit${n > 1 ? "s" : ""}`]), data.credits || 3)),
      field("f-score", "Score (%)", score),
      field("f-diff", "Difficulty", selectEl("difficulty", [["easy", "Easy"], ["moderate", "Moderate"], ["difficult", "Difficult"]], data.difficulty || "moderate")),
      field("f-conf", "Confidence", selectEl("confidence", [[1, "1 Very low"], [2, "2 Low"], [3, "3 Fair"], [4, "4 High"], [5, "5 Very high"]], data.confidence || 3)),
      h("label", { class: "check f-rep" }, repeat, " Repeat course"),
      h("button", {
        type: "button", class: "icon-btn", "aria-label": "Remove course",
        onclick: () => { if (courseList.children.length > 1) { row.remove(); refreshPriority(); } },
      }, "×"));
    return row;
  }

  function setCourses(list) {
    courseList.replaceChildren(...list.map(courseRow));
    refreshPriority();
  }

  function refreshPriority() {
    const select = $("#priority-course");
    const keep = select.value;
    const codes = $$('[data-f="code"]', courseList).map((i) => i.value.trim().toUpperCase()).filter(Boolean);
    select.replaceChildren(h("option", { value: "" }, "None"), ...[...new Set(codes)].map((c) => h("option", { value: c }, c)));
    select.value = codes.includes(keep) ? keep : "";
  }
  courseList.addEventListener("input", (e) => { if (e.target.matches('[data-f="code"]')) refreshPriority(); });

  /* ---------- form <-> object ---------- */
  const SIMPLE = [
    "level", "department", "semester", "cgpa", "last_gpa", "current_gpa", "target_cgpa", "goal_type",
    "credits_earned", "credits_required", "hours_per_day", "days_per_week", "preferred_period",
    "attendance_pct", "assignment", "exam_date", "challenge", "priority_course",
  ];

  function collect() {
    const out = {};
    for (const name of SIMPLE) out[name] = form.elements[name].value;
    out.methods = $$('input[name="methods"]:checked', form).map((i) => i.value);
    out.materials = $$('input[name="materials"]:checked', form).map((i) => i.value);
    out.misses_deadlines = form.elements.misses_deadlines.checked;
    out.courses = $$(".course-row", courseList).map((row) => {
      const get = (f) => $(`[data-f="${f}"]`, row);
      return {
        code: get("code").value, credits: get("credits").value, score: get("score").value,
        difficulty: get("difficulty").value, confidence: get("confidence").value, repeat: get("repeat").checked,
      };
    });
    return out;
  }

  function fill(data) {
    for (const name of SIMPLE) if (data[name] != null) form.elements[name].value = data[name];
    for (const box of $$('input[name="methods"]', form)) box.checked = (data.methods || []).includes(box.value);
    for (const box of $$('input[name="materials"]', form)) box.checked = (data.materials || []).includes(box.value);
    form.elements.misses_deadlines.checked = !!data.misses_deadlines;
    setCourses(data.courses && data.courses.length ? data.courses : [{}, {}, {}]);
    if (data.priority_course) $("#priority-course").value = data.priority_course;
    clearErrors();
  }

  function clearForm() {
    form.reset();
    setCourses([{}, {}, {}]);
    clearErrors();
    results.hidden = true;
    results.replaceChildren();
  }

  /* ---------- errors ---------- */
  function clearErrors() {
    for (const el of $$(".err", form)) el.textContent = "";
    for (const el of $$(".has-error", form)) el.classList.remove("has-error");
    $("#net-error").hidden = true;
  }

  function showErrors(errors) {
    let first = null;
    const mark = (container, message) => {
      const box = container.closest(".field") || container;
      box.classList.add("has-error");
      const err = $(".err", box);
      if (err) err.textContent = message;
      first = first || box;
    };
    for (const [key, message] of Object.entries(errors)) {
      const m = key.match(/^courses\.(\d+)\.(\w+)$/);
      if (m) {
        const row = $$(".course-row", courseList)[Number(m[1])];
        const input = row && $(`[data-f="${m[2]}"]`, row);
        if (input) mark(input, message);
      } else if (key === "courses") {
        $("#courses-err").textContent = message;
        first = first || $("#courses");
      } else if (form.elements[key] && form.elements[key].nodeType === 1) {
        mark(form.elements[key], message);
      } else if (key === "_") {
        showNetError(message);
      }
    }
    if (first) first.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
  }

  function showNetError(message) {
    const box = $("#net-error");
    box.textContent = message;
    box.hidden = false;
  }

  /* ---------- submit ---------- */
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearErrors();
    const button = $("#submit");
    const status = $("#status");
    button.disabled = true;
    status.textContent = "Building your plan...";
    try {
      const response = await fetch("/api/advise", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collect()),
      });
      const data = await response.json();
      if (data.ok) {
        render(data);
        status.textContent = "Plan ready.";
      } else if (data.errors) {
        showErrors(data.errors);
        status.textContent = "Fix the highlighted fields.";
      } else {
        showNetError(data.error || "The advisor could not build a plan.");
        status.textContent = "";
      }
    } catch (err) {
      showNetError("Could not reach the advisor. Check that `python app.py` is still running, then try again.");
      status.textContent = "";
    } finally {
      button.disabled = false;
    }
  });

  $("#add-course").addEventListener("click", () => {
    if (courseList.children.length >= 12) return;
    courseList.append(courseRow());
    $$('[data-f="code"]', courseList).pop().focus();
  });
  $("#reset-form").addEventListener("click", clearForm);
  $("#load-example").addEventListener("click", async () => {
    try {
      fill(await (await fetch("/api/example")).json());
    } catch (e) {
      showNetError("Could not load the example. Check that the server is running.");
    }
  });

  /* ---------- results ---------- */
  const chipFor = (label) => (label === "Urgent" ? "bad" : label === "Needs attention" ? "warn" : "good");
  const colorFor = (codes, code) => COURSE_COLORS[codes.indexOf(code) % COURSE_COLORS.length];

  function meter(pct, kind = "") {
    const fill = h("i", { class: kind });
    requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${pct}%`; }));
    return h("div", { class: "meter", role: "img", "aria-label": `${pct} percent` }, fill);
  }

  function render(r) {
    const s = r.summary, g = r.goal, t = r.timetable;
    const standingKey = s.predicted_standing.toLowerCase().replace(" ", "-");
    const diff = g.projected_cgpa - g.cgpa;
    const statusText = {
      on_track: ["On track", "good"], stretch: ["Stretch goal", "warn"],
      out_of_reach: ["Not possible in one semester", "bad"], at_risk: ["At risk", "bad"],
    }[g.status];

    const hero = h("div", { class: `card hero ${standingKey}` },
      h("span", { class: "kpi-label" }, "Predicted standing"),
      h("span", { class: "kpi-value" }, s.predicted_standing),
      h("span", { class: "kpi-sub" }, `Expected semester GPA ${num(s.predicted_gpa)} (${s.predicted_class})`),
      h("span", { class: "pill" }, `${Math.round(s.confidence * 100)}% model confidence`),
      h("span", { class: "kpi-sub" }, s.standing_note));

    const projected = h("div", { class: "card" },
      h("span", { class: "kpi-label" }, "Projected CGPA"),
      h("span", { class: "kpi-value" }, num(g.projected_cgpa)),
      h("span", { class: "kpi-sub" }, g.projected_class),
      h("span", { class: "kpi-sub" }, `${diff >= 0 ? "Up" : "Down"} ${num(Math.abs(diff))} from ${num(g.cgpa)}`));

    const credits = h("div", { class: "card" },
      h("span", { class: "kpi-label" }, "Credit units"),
      h("span", { class: "kpi-value" }, `${g.credits_earned}`, h("span", { class: "kpi-sub" }, ` / ${g.credits_required}`)),
      meter(g.credits_pct, "blue"),
      h("span", { class: "kpi-sub" }, `${g.remaining_credits} units to go${g.semesters_left ? `, about ${g.semesters_left} semester${g.semesters_left > 1 ? "s" : ""} at ${s.credit_load} units` : ""}`));

    const goal = h("div", { class: "card" },
      h("span", { class: "kpi-label" }, "Target CGPA"),
      h("span", { class: "kpi-value" }, num(g.target)),
      meter(g.progress_pct, "amber"),
      h("span", { class: `chip ${statusText[1]}`, style: "align-self:flex-start" }, statusText[0]),
      g.needed_gpa != null && h("span", { class: "kpi-sub" }, `Needs a ${num(g.needed_gpa)} GPA this semester`));

    /* GPA outlook */
    const bars = h("div", { class: "bars", style: `--t:${Math.min(g.target / 5, 1)}` },
      h("div", { class: "target" }, h("span", {}, `Target ${num(g.target)}`)),
      r.outlook.map((o) => {
        const fill = h("div", { class: `bar-fill ${o.kind}` }, num(o.value));
        requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${Math.max(o.value / 5 * 100, 9)}%`; }));
        return h("div", { class: "bar-row" }, h("span", { class: "name" }, o.label), h("div", { class: "bar-track" }, fill));
      }));
    const outlook = h("div", { class: "card" },
      h("h2", {}, "GPA outlook"),
      h("p", { class: "sub" }, "Bars run from 0 to 5. The dashed line is your target CGPA."),
      bars);

    /* course priority */
    const prio = h("div", { class: "card" },
      h("h2", {}, "Course priority"),
      h("p", { class: "sub" }, "Based on score, difficulty, confidence and credit units."),
      h("ul", { class: "prio" }, r.courses.map((c) =>
        h("li", {},
          h("span", { class: "rank" }, c.rank),
          h("span", { class: "code" }, c.code, h("span", { class: "muted", style: "font-weight:500" }, ` ${c.credits} units`)),
          h("span", { class: `chip ${chipFor(c.label)}` }, c.label),
          meter(c.priority, c.label === "On track" ? "blue" : ""),
          h("span", { class: "why" }, c.reasons.slice(0, 3).join(", "))))));

    /* timetable */
    const codes = r.courses.map((c) => c.code);
    const exam = t.exam;
    const facts = [
      h("span", { class: "fact" }, `${hrs(t.weekly_hours)} planned per week`),
      h("span", { class: "fact" }, `${hrs(t.recommended_hours)} suggested for ${s.credit_load} units`),
      h("span", { class: "fact" }, `Best time: ${t.period}`),
    ];
    if (exam) {
      facts.push(h("span", { class: "fact" }, exam.passed ? "Exam date has passed" : `Exams in ${exam.days_left} day${exam.days_left === 1 ? "" : "s"}`));
    }
    const week = h("div", { class: "week" }, t.days.map((d) =>
      h("div", { class: `day${d.rest || !d.sessions.length ? " rest" : ""}` },
        h("h3", {}, d.day),
        d.sessions.length
          ? d.sessions.map((x) => h("div", { class: "session", style: `--c:${colorFor(codes, x.course)}` }, h("b", {}, x.course), hrs(x.hours)))
          : h("span", { class: "rest-note" }, "Rest day"))));
    const legend = h("div", { class: "legend" }, t.per_course.map((c) =>
      h("span", { style: `--c:${colorFor(codes, c.course)}` }, h("i"), `${c.course}: ${hrs(c.hours)} a week`)));
    const plan = h("div", { class: "card" },
      h("h2", {}, "Weekly study plan"),
      h("p", { class: "sub" }, "Hours are shared out by course priority. Harder, weaker and heavier courses get more time."),
      h("div", { class: "facts" }, facts),
      t.days.length ? [week, legend] : h("p", { class: "muted" }, "Set study hours above zero to get a timetable."));

    /* advice + model */
    const advice = h("div", { class: "card" },
      h("h2", {}, "Recommendations"),
      h("p", { class: "sub" }, "Most important first."),
      h("ul", { class: "advice" }, r.advice.map((a) =>
        h("li", {},
          h("span", { class: `chip ${a.priority === "high" ? "bad" : a.priority === "medium" ? "warn" : "good"}` },
            a.priority === "high" ? "Act now" : a.priority === "medium" ? "Soon" : "Keep going"),
          h("div", {}, h("b", {}, a.category), h("p", {}, a.message))))));

    const order = ["At Risk", "Fair", "Good", "Excellent"];
    const model = h("div", { class: "card" },
      h("h2", {}, "How sure is the model?"),
      h("p", { class: "sub" }, "Chance of each standing next semester."),
      order.filter((k) => k in s.probabilities).map((k) => {
        const pct = Math.round(s.probabilities[k] * 100);
        return h("div", { class: "prob" }, h("span", {}, k), meter(pct, k === "At Risk" ? "" : "blue"), h("span", { class: "pct" }, `${pct}%`));
      }),
      h("p", { class: "note" }, `The model leans most on ${r.model_focus.map((f) => FEATURE_NAMES[f.feature] || f.feature).join(", ")}. These weights describe the model overall, not your own case.`),
      s.score_estimated && h("p", { class: "note" }, "No scores were entered, so the average score was estimated from last semester's GPA."));

    results.replaceChildren(
      h("div", { class: "results-head" },
        h("div", {}, h("h2", {}, "Your semester plan"),
          h("p", { class: "muted" }, `${r.profile.level} level ${r.profile.department}, ${r.profile.semester} semester`)),
        h("button", { type: "button", class: "btn", onclick: () => window.print() }, "Save as PDF")),
      h("div", { class: "kpis" }, hero, projected, credits, goal),
      h("div", { class: "two" }, outlook, prio),
      plan,
      h("div", { class: "two" }, advice, model));
    results.hidden = false;
    results.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  }

  setCourses([{}, {}, {}]);
})();
