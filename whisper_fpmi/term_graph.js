function boot(data) {
  const courses = data.courses || [];
  const select = document.getElementById("course");
  const canvas = document.getElementById("graph");
  const side = document.getElementById("side");
  const filter = document.getElementById("filter");
  const stats = document.getElementById("stats");
  const ctx = canvas.getContext("2d");
  let nodes = [];
  let edges = [];
  let current = null;
  let selected = null;
  let drag = null;

  courses.forEach((course) => {
    const option = document.createElement("option");
    option.value = course.name;
    option.textContent = course.name + " (" + course.lectures.length + ")";
    select.appendChild(option);
  });

  function resize() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(rect.width, 320);
    canvas.height = Math.max(rect.height, 420);
  }

  function load(name) {
    current = courses.find((course) => course.name === name) || courses[0];
    if (!current) {
      stats.textContent = "Нет курсов";
      return;
    }
    const width = canvas.width || 800;
    const height = canvas.height || 560;
    nodes = current.terms.map((term, index) => {
      const angle = (index / Math.max(current.terms.length, 1)) * Math.PI * 2;
      const radius = 80 + (index % 5) * 28;
      return {
        lemma: term.lemma,
        count: term.count,
        lectures: term.lectures,
        x: width / 2 + Math.cos(angle) * radius,
        y: height / 2 + Math.sin(angle) * radius * 0.72,
        vx: 0,
        vy: 0,
      };
    });
    const byLemma = new Map(nodes.map((node) => [node.lemma, node]));
    edges = current.edges
      .map((edge) => ({
        a: byLemma.get(edge.source),
        b: byLemma.get(edge.target),
        weight: edge.weight,
      }))
      .filter((edge) => edge.a && edge.b);
    selected = null;
    stats.textContent =
      current.lectures.length +
      " лекций · " +
      nodes.length +
      " терминов · " +
      edges.length +
      " связей";
    settle();
    renderSide(null);
    draw();
  }

  function settle() {
    const width = canvas.width || 800;
    const height = canvas.height || 560;
    for (let step = 0; step < 250; step += 1) {
      tick(width, height);
    }
  }

  function tick(width, height) {
    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        const a = nodes[i];
        const b = nodes[j];
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let dist = Math.hypot(dx, dy) || 0.1;
        const force = 900 / (dist * dist);
        dx = (dx / dist) * force;
        dy = (dy / dist) * force;
        a.vx += dx;
        a.vy += dy;
        b.vx -= dx;
        b.vy -= dy;
      }
    }
    edges.forEach((edge) => {
      const dx = edge.b.x - edge.a.x;
      const dy = edge.b.y - edge.a.y;
      const dist = Math.hypot(dx, dy) || 0.1;
      const force = (dist - 130) * 0.012;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      edge.a.vx += fx;
      edge.a.vy += fy;
      edge.b.vx -= fx;
      edge.b.vy -= fy;
    });
    nodes.forEach((node) => {
      node.vx += (width / 2 - node.x) * 0.004;
      node.vy += (height / 2 - node.y) * 0.004;
      node.vx *= 0.72;
      node.vy *= 0.72;
      if (node !== drag) {
        node.x += node.vx;
        node.y += node.vy;
      }
      node.x = Math.max(30, Math.min(width - 30, node.x));
      node.y = Math.max(24, Math.min(height - 24, node.y));
    });
  }

  function radius(node) {
    const maxCount = nodes.reduce((max, item) => Math.max(max, item.count), 1);
    return 8 + 18 * Math.sqrt(node.count / maxCount);
  }

  function shown(node) {
    const query = filter.value.trim().toLowerCase();
    if (!query) {
      return true;
    }
    return node.lemma.toLowerCase().includes(query);
  }

  function draw() {
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#10151c";
    ctx.fillRect(0, 0, width, height);
    edges.forEach((edge) => {
      if (!shown(edge.a) || !shown(edge.b)) {
        return;
      }
      ctx.strokeStyle = "rgba(142, 202, 255, 0.35)";
      ctx.lineWidth = Math.min(1 + edge.weight, 6);
      ctx.beginPath();
      ctx.moveTo(edge.a.x, edge.a.y);
      ctx.lineTo(edge.b.x, edge.b.y);
      ctx.stroke();
    });
    nodes.forEach((node) => {
      if (!shown(node)) {
        return;
      }
      const r = radius(node);
      ctx.beginPath();
      ctx.fillStyle = node === selected ? "#f2c14e" : "#7eb6ff";
      ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#f7f4ea";
      ctx.font = "13px Segoe UI, sans-serif";
      ctx.fillText(node.lemma, node.x + r + 4, node.y + 4);
    });
  }

  function renderSide(node) {
    const lectures = current ? current.lectures : [];
    const allowed = node ? new Set(node.lectures) : null;
    const rows = lectures.filter((lecture) => !allowed || allowed.has(lecture.slug));
    const title = node
      ? "«" + node.lemma + "» · " + rows.length + " лекций"
      : "Лекции курса";
    const items = rows
      .map((lecture) => {
        const label = lecture.title || lecture.slug;
        if (lecture.url) {
          return '<li><a href="' + escapeHtml(lecture.url) + '" target="_blank" rel="noreferrer">' + escapeHtml(label) + "</a></li>";
        }
        return "<li>" + escapeHtml(label) + "</li>";
      })
      .join("");
    side.innerHTML = "<h2>" + escapeHtml(title) + "</h2><ol>" + items + "</ol>";
  }

  function hit(x, y) {
    for (let index = nodes.length - 1; index >= 0; index -= 1) {
      const node = nodes[index];
      if (!shown(node)) {
        continue;
      }
      if (Math.hypot(node.x - x, node.y - y) <= radius(node) + 4) {
        return node;
      }
    }
    return null;
  }

  function pointer(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * canvas.width,
      y: ((event.clientY - rect.top) / rect.height) * canvas.height,
    };
  }

  canvas.addEventListener("pointerdown", (event) => {
    const point = pointer(event);
    drag = hit(point.x, point.y);
    selected = drag;
    if (drag) {
      renderSide(drag);
      draw();
    }
  });
  window.addEventListener("pointermove", (event) => {
    if (!drag) {
      return;
    }
    const point = pointer(event);
    drag.x = point.x;
    drag.y = point.y;
    draw();
  });
  window.addEventListener("pointerup", () => {
    drag = null;
  });
  select.addEventListener("change", () => load(select.value));
  filter.addEventListener("input", draw);
  window.addEventListener("resize", () => {
    resize();
    if (current) {
      load(current.name);
    }
  });

  resize();
  if (courses.length) {
    load(courses[0].name);
  } else {
    stats.textContent = "Нет курсов";
  }
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
