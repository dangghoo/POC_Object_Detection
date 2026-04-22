"""
Interactive 12×4 rack grid drawer — browser-based (no display required).

Draws 5 vertical + 13 horizontal straight lines; intersections form the grid.

Run:
    python src/draw_rack_grid_interactive.py

Then open http://localhost:8765 in your browser.

Phase 1 — click TL then BR to set the initial bounding box.
Phase 2 — drag any of the 36 line endpoints to fine-tune.
Press S to save, R to reset.

Output: sample_data_single/WIN_20260421_13_37_56_Pro_grid.jpg
"""

import base64
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2

SRC = Path("sample_data_single/WIN_20260421_13_37_56_Pro.jpg")
DST = SRC.with_name(SRC.stem + "_grid.jpg")
PORT = 8765
ROWS, COLS = 12, 4
N_HLINES = ROWS + 1   # 13
N_VLINES = COLS + 1   # 5


# ── Python-side drawing (called on save) ────────────────────────────────────

def _line_isect(ax, ay, bx, by, cx, cy, dx, dy):
    """Intersection of infinite line AB with infinite line CD."""
    d = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
    if abs(d) < 1e-9:
        return ((ax + bx) / 2, (ay + by) / 2)
    t = ((ax - cx) * (cy - dy) - (ay - cy) * (cx - dx)) / d
    return (ax + t * (bx - ax), ay + t * (by - ay))


def draw_grid_from_lines(img, vlines, hlines):
    """
    vlines: [{tx,ty,bx,by}, ...]  5 entries (vertical lines, each with top/bottom endpoint)
    hlines: [{lx,ly,rx,ry}, ...]  13 entries (horizontal lines, each with left/right endpoint)
    """
    out = img.copy()

    # grid[h][v] = (x, y) — intersection of hline h and vline v
    grid = [
        [_line_isect(hl['lx'], hl['ly'], hl['rx'], hl['ry'],
                     vl['tx'], vl['ty'], vl['bx'], vl['by'])
         for vl in vlines]
        for hl in hlines
    ]

    for vl in vlines:
        cv2.line(out, (int(round(vl['tx'])), int(round(vl['ty']))),
                      (int(round(vl['bx'])), int(round(vl['by']))), (0, 220, 0), 2)

    for hl in hlines:
        cv2.line(out, (int(round(hl['lx'])), int(round(hl['ly']))),
                      (int(round(hl['rx'])), int(round(hl['ry']))), (0, 220, 0), 2)

    for r in range(ROWS):
        for c in range(COLS):
            cx = int((grid[r][c][0] + grid[r][c+1][0] +
                      grid[r+1][c][0] + grid[r+1][c+1][0]) / 4)
            cy = int((grid[r][c][1] + grid[r][c+1][1] +
                      grid[r+1][c][1] + grid[r+1][c+1][1]) / 4)
            label = f"({r},{c})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(out, label, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(out, label, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return out


# ── HTML / JS ────────────────────────────────────────────────────────────────

IMG_B64 = base64.b64encode(SRC.read_bytes()).decode()

HTML = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Rack Grid — straight lines</title>
<style>
  body {{ margin:0; background:#111; color:#eee; font-family:monospace; user-select:none; }}
  #wrap {{ display:flex; gap:12px; padding:10px; }}
  canvas {{ cursor:crosshair; border:2px solid #444; flex-shrink:0; }}
  #panel {{ width:220px; font-size:0.82rem; line-height:1.7; }}
  h2 {{ margin:0 0 6px; font-size:1rem; }}
  button {{ margin:6px 4px 0 0; padding:6px 14px; font-size:0.85rem;
            background:#333; color:#eee; border:1px solid #666; cursor:pointer; }}
  button:hover {{ background:#555; }}
  #status {{ margin-top:10px; color:#6f6; }}
  .dim {{ color:#888; }}
</style>
</head>
<body>
<div id="wrap">
  <canvas id="c"></canvas>
  <div id="panel">
    <h2>12 × 4 Rack Grid</h2>
    <div id="phase1">
      <b>Phase 1 — bounding box</b><br>
      <span id="s0">⬜ TL (top-left)</span><br>
      <span id="s1">⬜ BR (bottom-right)</span>
    </div>
    <div id="phase2" style="display:none">
      <b>Phase 2 — drag endpoints</b><br>
      <span class="dim">5 vertical + 13 horizontal lines.<br>
      Each has 2 draggable endpoints<br>
      (orange = vertical, blue = horizontal).</span>
    </div>
    <br>
    <button onclick="reset()">Reset (R)</button>
    <button onclick="save()">Save (S)</button>
    <div id="status"></div>
    <br>
    <span class="dim">R = reset &nbsp; S = save</span>
  </div>
</div>

<script>
const ROWS = {ROWS}, COLS = {COLS};
const N_HLINES = ROWS + 1, N_VLINES = COLS + 1;
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const img = new Image();
let scale = 1;

let phase = 1;
let initClicks = [];
let vlines = [], hlines = [];   // original-pixel coords
let dragging = null;             // {{type:'v'|'h', idx, end:'top'|'bot'|'left'|'right'}}
const DRAG_RADIUS = 18;

// ── image load ───────────────────────────────────────────────────────────────
img.onload = () => {{
  scale = Math.min(1, (window.innerHeight - 40) / img.height,
                      (window.innerWidth  - 260) / img.width);
  canvas.width  = Math.round(img.width  * scale);
  canvas.height = Math.round(img.height * scale);
  redraw();
}};
img.src = 'data:image/jpeg;base64,{IMG_B64}';

// ── math ─────────────────────────────────────────────────────────────────────
function lineIntersect(ax,ay,bx,by, cx,cy,dx,dy) {{
  const d = (ax-bx)*(cy-dy) - (ay-by)*(cx-dx);
  if (Math.abs(d) < 1e-9) return {{x:(ax+bx)/2, y:(ay+by)/2}};
  const t = ((ax-cx)*(cy-dy) - (ay-cy)*(cx-dx)) / d;
  return {{x: ax + t*(bx-ax), y: ay + t*(by-ay)}};
}}

function getIntersections() {{
  return hlines.map(hl =>
    vlines.map(vl =>
      lineIntersect(hl.lx,hl.ly,hl.rx,hl.ry, vl.tx,vl.ty,vl.bx,vl.by)
    )
  );
}}

// ── init lines from two clicks ───────────────────────────────────────────────
function initLines(c0, c1) {{
  const tlx = Math.min(c0[0],c1[0]), tly = Math.min(c0[1],c1[1]);
  const brx = Math.max(c0[0],c1[0]), bry = Math.max(c0[1],c1[1]);
  vlines = [];
  for (let v = 0; v < N_VLINES; v++) {{
    const t = v / (N_VLINES - 1);
    const x = tlx + t * (brx - tlx);
    vlines.push({{tx:x, ty:tly, bx:x, by:bry}});
  }}
  hlines = [];
  for (let h = 0; h < N_HLINES; h++) {{
    const t = h / (N_HLINES - 1);
    const y = tly + t * (bry - tly);
    hlines.push({{lx:tlx, ly:y, rx:brx, ry:y}});
  }}
}}

// ── drawing ───────────────────────────────────────────────────────────────────
function redraw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  if (phase === 1) {{
    const labels = ['TL','BR'];
    const colors = ['#ff4444','#44ff44'];
    initClicks.forEach(([ox,oy], i) => {{
      const x = ox*scale, y = oy*scale;
      ctx.fillStyle = colors[i];
      ctx.beginPath(); ctx.arc(x,y,7,0,2*Math.PI); ctx.fill();
      ctx.fillStyle = 'white'; ctx.font = 'bold 13px monospace';
      ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(labels[i], x+9, y-4);
    }});
    return;
  }}

  const grid = getIntersections();

  ctx.strokeStyle = '#00dd00';
  ctx.lineWidth = 2;

  vlines.forEach(vl => {{
    ctx.beginPath();
    ctx.moveTo(vl.tx*scale, vl.ty*scale);
    ctx.lineTo(vl.bx*scale, vl.by*scale);
    ctx.stroke();
  }});

  hlines.forEach(hl => {{
    ctx.beginPath();
    ctx.moveTo(hl.lx*scale, hl.ly*scale);
    ctx.lineTo(hl.rx*scale, hl.ry*scale);
    ctx.stroke();
  }});

  // cell labels
  ctx.font = '11px monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let r = 0; r < ROWS; r++) {{
    for (let c = 0; c < COLS; c++) {{
      const cx = (grid[r][c].x+grid[r][c+1].x+grid[r+1][c].x+grid[r+1][c+1].x)/4 * scale;
      const cy = (grid[r][c].y+grid[r][c+1].y+grid[r+1][c].y+grid[r+1][c+1].y)/4 * scale;
      ctx.fillStyle = 'black'; ctx.lineWidth = 3;
      ctx.strokeText(`(${{r}},${{c}})`, cx, cy);
      ctx.fillStyle = 'white';
      ctx.fillText(`(${{r}},${{c}})`, cx, cy);
    }}
  }}

  // handles
  vlines.forEach(vl => {{
    dot(vl.tx*scale, vl.ty*scale, '#ff8800', 6);
    dot(vl.bx*scale, vl.by*scale, '#ff8800', 6);
  }});
  hlines.forEach(hl => {{
    dot(hl.lx*scale, hl.ly*scale, '#00aaff', 5);
    dot(hl.rx*scale, hl.ry*scale, '#00aaff', 5);
  }});
}}

function dot(x, y, color, r) {{
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(x, y, r, 0, 2*Math.PI); ctx.fill();
}}

// ── handle picking ────────────────────────────────────────────────────────────
function nearestHandle(dx, dy) {{
  let best = null, bestDist = DRAG_RADIUS;
  vlines.forEach((vl, idx) => {{
    [['top',vl.tx,vl.ty],['bot',vl.bx,vl.by]].forEach(([end,x,y]) => {{
      const d = Math.hypot(x*scale-dx, y*scale-dy);
      if (d < bestDist) {{ bestDist=d; best={{type:'v',idx,end}}; }}
    }});
  }});
  hlines.forEach((hl, idx) => {{
    [['left',hl.lx,hl.ly],['right',hl.rx,hl.ry]].forEach(([end,x,y]) => {{
      const d = Math.hypot(x*scale-dx, y*scale-dy);
      if (d < bestDist) {{ bestDist=d; best={{type:'h',idx,end}}; }}
    }});
  }});
  return best;
}}

function applyDrag(handle, ox, oy) {{
  if (handle.type === 'v') {{
    const vl = vlines[handle.idx];
    if (handle.end === 'top') {{ vl.tx=ox; vl.ty=oy; }} else {{ vl.bx=ox; vl.by=oy; }}
  }} else {{
    const hl = hlines[handle.idx];
    if (handle.end === 'left') {{ hl.lx=ox; hl.ly=oy; }} else {{ hl.rx=ox; hl.ry=oy; }}
  }}
}}

// ── mouse events ──────────────────────────────────────────────────────────────
canvas.addEventListener('mousedown', e => {{
  const rect = canvas.getBoundingClientRect();
  const dx = e.clientX-rect.left, dy = e.clientY-rect.top;

  if (phase === 1) {{
    if (initClicks.length < 2) {{
      initClicks.push([Math.round(dx/scale), Math.round(dy/scale)]);
      updateClickLog();
      if (initClicks.length === 2) {{
        initLines(initClicks[0], initClicks[1]);
        phase = 2;
        document.getElementById('phase1').style.display = 'none';
        document.getElementById('phase2').style.display = '';
      }}
      redraw();
    }}
    return;
  }}

  const hit = nearestHandle(dx, dy);
  if (hit) {{
    dragging = hit;
    canvas.style.cursor = 'grabbing';
    e.preventDefault();
  }}
}});

canvas.addEventListener('mousemove', e => {{
  const rect = canvas.getBoundingClientRect();
  const dx = e.clientX-rect.left, dy = e.clientY-rect.top;
  if (dragging) {{
    applyDrag(dragging, dx/scale, dy/scale);
    redraw();
    e.preventDefault();
    return;
  }}
  if (phase === 2) {{
    canvas.style.cursor = nearestHandle(dx, dy) ? 'grab' : 'default';
  }}
}});

canvas.addEventListener('mouseup', () => {{
  dragging = null;
  canvas.style.cursor = phase === 2 ? 'grab' : 'crosshair';
}});

canvas.addEventListener('mouseleave', () => {{ dragging = null; }});

// ── corner log ────────────────────────────────────────────────────────────────
function updateClickLog() {{
  const labels = ['TL (top-left)', 'BR (bottom-right)'];
  labels.forEach((l, i) => {{
    const el = document.getElementById('s'+i);
    if (i < initClicks.length) {{
      el.textContent = `✅ ${{l.split(' ')[0]}} (${{initClicks[i][0]}}, ${{initClicks[i][1]}})`;
      el.style.color = '#6f6';
    }} else if (i === initClicks.length) {{
      el.textContent = `👉 ${{l}} ← click`;
      el.style.color = '#ff0';
    }} else {{
      el.textContent = `⬜ ${{l}}`;
      el.style.color = '';
    }}
  }});
}}

// ── save ──────────────────────────────────────────────────────────────────────
function save() {{
  if (phase !== 2) {{
    document.getElementById('status').textContent = 'Set bounding box first.';
    return;
  }}
  const payload = {{
    vlines: vlines.map(vl => ({{
      tx: Math.round(vl.tx), ty: Math.round(vl.ty),
      bx: Math.round(vl.bx), by: Math.round(vl.by)
    }})),
    hlines: hlines.map(hl => ({{
      lx: Math.round(hl.lx), ly: Math.round(hl.ly),
      rx: Math.round(hl.rx), ry: Math.round(hl.ry)
    }}))
  }};
  fetch('/save', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(payload),
  }}).then(r => r.json()).then(d => {{
    document.getElementById('status').textContent =
      d.ok ? '✅ Saved: ' + d.path : '❌ ' + d.error;
  }});
}}

function reset() {{
  phase = 1; initClicks = []; vlines = []; hlines = [];
  dragging = null;
  canvas.style.cursor = 'crosshair';
  document.getElementById('phase1').style.display = '';
  document.getElementById('phase2').style.display = 'none';
  document.getElementById('status').textContent = '';
  updateClickLog();
  redraw();
}}

document.addEventListener('keydown', e => {{
  if (e.key==='r'||e.key==='R') reset();
  if (e.key==='s'||e.key==='S') save();
}});

updateClickLog();
</script>
</body>
</html>"""


# ── HTTP server ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode())

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        data = json.loads(self.rfile.read(length))
        try:
            img = cv2.imread(str(SRC))
            result = draw_grid_from_lines(img, data["vlines"], data["hlines"])
            cv2.imwrite(str(DST), result, [cv2.IMWRITE_JPEG_QUALITY, 95])

            grid_json = DST.with_suffix(".json")
            payload = {
                "source_size": {"width": img.shape[1], "height": img.shape[0]},
                "rows": ROWS,
                "cols": COLS,
                "vlines": data["vlines"],
                "hlines": data["hlines"],
            }
            grid_json.write_text(json.dumps(payload, indent=2))

            resp = {"ok": True, "path": str(DST), "json": str(grid_json)}
            print(f"\nSaved → {DST}")
            print(f"Saved → {grid_json}")
        except Exception as exc:
            resp = {"ok": False, "error": str(exc)}
        body = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Image not found: {SRC}")
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Open in browser:  http://localhost:{PORT}")
    print(f"Grid: {ROWS} rows × {COLS} cols  —  {N_VLINES} vertical + {N_HLINES} horizontal lines")
    print("Ctrl-C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
