#!/usr/bin/env python3
"""
NHL statistiky týmu – kanadské body, góly, asistence, trestné minuty, střely...

Spuštění:   python nhl_statistiky.py
Pak se v prohlížeči otevře http://localhost:8765

Potřebuje jen Python 3.8+ (žádné další balíčky) a připojení k internetu.
Data bere z oficiálního NHL API (api-web.nhle.com).
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8765
NHL_API = "https://api-web.nhle.com/v1/"
CACHE_SECONDS = 300  # data se drží 5 minut, pak se stáhnou znovu

_cache = {}
_cache_lock = threading.Lock()


def nhl_get(path):
    """Stáhne JSON z NHL API (s krátkou cache)."""
    now = time.time()
    with _cache_lock:
        hit = _cache.get(path)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1]
    req = urllib.request.Request(
        NHL_API + path,
        headers={"User-Agent": "Mozilla/5.0 (nhl-statistiky)", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read()
    with _cache_lock:
        _cache[path] = (now, data)
    return data


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # ticho v konzoli

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path.startswith("/api/"):
            path = self.path[len("/api/"):]
            # povolíme jen bezpečné znaky
            if not all(c.isalnum() or c in "/-_." for c in path):
                self._send(400, b'{"error":"bad path"}', "application/json")
                return
            try:
                data = nhl_get(path)
                self._send(200, data, "application/json; charset=utf-8")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    text = "Pro tuto volbu nejsou žádná data (třeba play-off ještě nezačalo nebo tým nehrál)."
                else:
                    text = f"NHL API vrátilo chybu {e.code}"
                msg = json.dumps({"error": text}).encode()
                self._send(502, msg, "application/json")
            except Exception as e:  # síť, SSL...
                hint = ""
                if "CERTIFICATE" in str(e).upper():
                    hint = " (Na Macu spusť 'Install Certificates.command' ze složky Pythonu.)"
                msg = json.dumps({"error": f"Nepodařilo se stáhnout data: {e}{hint}"}).encode()
                self._send(502, msg, "application/json")
            return
        self._send(404, b"Not found", "text/plain")


PAGE = r"""<!DOCTYPE html>
<html lang="cs">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NHL – kanadské bodování týmu</title>
<style>
  :root {
    --bg: #0f1419; --panel: #172029; --panel2: #1e2a36; --line: #2a3846;
    --text: #e6edf3; --muted: #8b9bab; --accent: #4fa3ff; --gold: #f2c14e;
    --good: #3ecf8e; --bad: #ff6b6b;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         background: var(--bg); color: var(--text); }
  header { padding: 20px 24px 12px; display: flex; flex-wrap: wrap; gap: 16px; align-items: center; }
  header h1 { font-size: 22px; margin: 0; flex: 1 1 auto; }
  header h1 small { color: var(--muted); font-weight: 400; font-size: 14px; margin-left: 8px; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px; padding: 0 24px 16px; }
  select, input, button {
    background: var(--panel); color: var(--text); border: 1px solid var(--line);
    border-radius: 8px; padding: 8px 10px; font-size: 14px;
  }
  button { cursor: pointer; }
  button.on { background: var(--accent); border-color: var(--accent); color: #fff; }
  .seg { display: inline-flex; }
  .seg button { border-radius: 0; }
  .seg button:first-child { border-radius: 8px 0 0 8px; }
  .seg button:last-child { border-radius: 0 8px 8px 0; }
  main { padding: 0 24px 40px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
  .card .lbl { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }
  .card .val { font-size: 28px; font-weight: 700; margin: 4px 0 2px; }
  .card .who { font-size: 14px; }
  h2 { font-size: 16px; margin: 24px 0 10px; color: var(--muted); font-weight: 600; }
  .tablewrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; white-space: nowrap; }
  th, td { padding: 8px 10px; text-align: right; border-bottom: 1px solid var(--line); }
  th { background: var(--panel2); position: sticky; top: 0; cursor: pointer; user-select: none; font-weight: 600; }
  th:hover { color: var(--accent); }
  th.sorted { color: var(--gold); }
  th.left, td.left { text-align: left; }
  tbody tr:nth-child(even) { background: rgba(255,255,255,.02); }
  tbody tr:hover { background: rgba(79,163,255,.08); }
  td.pts { font-weight: 700; color: var(--gold); }
  td.pos { color: var(--muted); }
  .plus { color: var(--good); } .minus { color: var(--bad); }
  .rank { color: var(--muted); width: 32px; }
  .legend { color: var(--muted); font-size: 12px; margin-top: 10px; line-height: 1.6; }
  .msg { padding: 30px; text-align: center; color: var(--muted); }
  .err { color: var(--bad); }
  img.logo { height: 44px; }
</style>
</head>
<body>
<header>
  <img class="logo" id="logo" alt="">
  <h1 id="title">NHL – kanadské bodování<small id="subtitle"></small></h1>
</header>

<div class="controls">
  <select id="team" title="Tým"></select>
  <select id="season" title="Sezóna"></select>
  <div class="seg" id="gtype">
    <button data-v="2" class="on">Základní část</button>
    <button data-v="3">Play-off</button>
  </div>
  <div class="seg" id="posf">
    <button data-v="all" class="on">Všichni</button>
    <button data-v="F">Útočníci</button>
    <button data-v="D">Obránci</button>
  </div>
  <input id="search" placeholder="Hledat hráče…">
  <button id="reload" title="Znovu stáhnout data">↻ Obnovit</button>
</div>

<main>
  <div class="cards" id="cards"></div>
  <h2>Hráči v poli</h2>
  <div class="tablewrap"><table id="skaters"></table></div>
  <h2>Brankáři</h2>
  <div class="tablewrap"><table id="goalies"></table></div>
  <div class="legend">
    Z = zápasy · G = góly · A = asistence · <b>B = kanadské body (G+A)</b> · B/Z = body na zápas ·
    +/- = plus/minus · TM = trestné minuty · S = střely na branku · %S = úspěšnost střelby ·
    PPG = góly v přesilovce · SHG = góly v oslabení · GWG = vítězné góly · TOI = průměrný čas na ledě ·
    %VB = úspěšnost na vhazování.<br>
    Brankáři: V/P/PP = výhry/prohry/prohry v prodloužení · GAA = průměr inkasovaných gólů ·
    %Z = úspěšnost zásahů · ČK = čistá konta. Kliknutím na záhlaví sloupce se tabulka seřadí.
  </div>
</main>

<script>
const FALLBACK_TEAMS = ["ANA","BOS","BUF","CAR","CBJ","CGY","CHI","COL","DAL","DET","EDM","FLA",
  "LAK","MIN","MTL","NJD","NSH","NYI","NYR","OTT","PHI","PIT","SEA","SJS","STL","TBL","TOR",
  "UTA","VAN","VGK","WPG","WSH"];

const state = {
  team: null, season: "now", gameType: "2", pos: "all", q: "",
  skaters: [], goalies: [], teams: {},
  sSort: { key: "points", dir: -1 }, gSort: { key: "wins", dir: -1 },
};

const $ = (id) => document.getElementById(id);
const name = (p) => `${p.firstName?.default ?? ""} ${p.lastName?.default ?? ""}`.trim();
const pct = (v, d = 1) => (v == null ? "–" : (v * 100).toFixed(d));
const num = (v, d = 2) => (v == null ? "–" : Number(v).toFixed(d));
const toi = (sec) => {
  if (sec == null) return "–";
  const s = Math.round(sec);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};
const posName = { C: "C", L: "LW", R: "RW", D: "D", G: "G" };
const fmtSeason = (s) => { s = String(s); return `${s.slice(0, 4)}/${s.slice(6, 8)}`; };

async function api(path) {
  const r = await fetch("/api/" + path);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || "Chyba " + r.status);
  return j;
}

// ---------- sloupce ----------
const SK_COLS = [
  { k: "_rank", t: "#", cls: "rank", nosort: true },
  { k: "_name", t: "Hráč", cls: "left", v: (p) => name(p) },
  { k: "positionCode", t: "Poz", cls: "left pos", f: (v) => posName[v] || v },
  { k: "gamesPlayed", t: "Z" },
  { k: "goals", t: "G" },
  { k: "assists", t: "A" },
  { k: "points", t: "B", cls: "pts" },
  { k: "_ppg", t: "B/Z", v: (p) => (p.gamesPlayed ? p.points / p.gamesPlayed : 0), f: (v) => num(v) },
  { k: "plusMinus", t: "+/-", f: (v) => (v > 0 ? `<span class="plus">+${v}</span>` : v < 0 ? `<span class="minus">${v}</span>` : "0") },
  { k: "penaltyMinutes", t: "TM" },
  { k: "shots", t: "S" },
  { k: "shootingPctg", t: "%S", f: (v) => pct(v) },
  { k: "powerPlayGoals", t: "PPG" },
  { k: "shorthandedGoals", t: "SHG" },
  { k: "gameWinningGoals", t: "GWG" },
  { k: "avgTimeOnIcePerGame", t: "TOI", f: toi },
  { k: "faceoffWinPctg", t: "%VB", f: (v) => (v ? pct(v) : "–") },
];
const G_COLS = [
  { k: "_rank", t: "#", cls: "rank", nosort: true },
  { k: "_name", t: "Brankář", cls: "left", v: (p) => name(p) },
  { k: "gamesPlayed", t: "Z" },
  { k: "gamesStarted", t: "Start" },
  { k: "wins", t: "V" },
  { k: "losses", t: "P" },
  { k: "overtimeLosses", t: "PP" },
  { k: "goalsAgainstAverage", t: "GAA", f: (v) => num(v) },
  { k: "savePercentage", t: "%Z", f: (v) => (v == null ? "–" : Number(v).toFixed(3)) },
  { k: "shotsAgainst", t: "Střely" },
  { k: "saves", t: "Zákroky" },
  { k: "goalsAgainst", t: "OG" },
  { k: "shutouts", t: "ČK" },
  { k: "points", t: "B", cls: "pts" },
  { k: "penaltyMinutes", t: "TM" },
];

const val = (col, p) => (col.v ? col.v(p) : p[col.k]);

function renderTable(el, cols, rows, sort, onSort) {
  const sc = cols.find((c) => c.k === sort.key);
  rows = [...rows].sort((a, b) => {
    const x = val(sc, a), y = val(sc, b);
    if (typeof x === "string") return sort.dir * x.localeCompare(y, "cs");
    return sort.dir * ((x ?? -1e9) - (y ?? -1e9)) || (b.points ?? 0) - (a.points ?? 0);
  });
  const head = "<thead><tr>" + cols.map((c) =>
    `<th class="${c.cls || ""} ${c.k === sort.key ? "sorted" : ""}" data-k="${c.k}">${c.t}${c.k === sort.key ? (sort.dir < 0 ? " ▾" : " ▴") : ""}</th>`
  ).join("") + "</tr></thead>";
  const body = "<tbody>" + (rows.length ? rows.map((p, i) => "<tr>" + cols.map((c) => {
    if (c.k === "_rank") return `<td class="rank">${i + 1}</td>`;
    const v = val(c, p);
    return `<td class="${c.cls || ""}">${c.f ? c.f(v) : (v ?? "–")}</td>`;
  }).join("") + "</tr>").join("")
    : `<tr><td colspan="${cols.length}" class="msg">Žádní hráči</td></tr>`) + "</tbody>";
  el.innerHTML = head + body;
  el.querySelectorAll("th").forEach((th) => {
    const c = cols.find((c) => c.k === th.dataset.k);
    if (c.nosort) return;
    th.onclick = () => {
      if (sort.key === c.k) sort.dir *= -1;
      else { sort.key = c.k; sort.dir = c.cls?.includes("left") ? 1 : -1; }
      onSort();
    };
  });
}

function filteredSkaters() {
  const q = state.q.toLowerCase();
  return state.skaters.filter((p) => {
    if (state.pos === "D" && p.positionCode !== "D") return false;
    if (state.pos === "F" && p.positionCode === "D") return false;
    return !q || name(p).toLowerCase().includes(q);
  });
}

function renderCards() {
  const s = state.skaters;
  if (!s.length) { $("cards").innerHTML = ""; return; }
  const top = (key) => s.reduce((a, b) => ((b[key] ?? 0) > (a[key] ?? 0) ? b : a));
  const tot = (key) => s.reduce((a, b) => a + (b[key] ?? 0), 0);
  const items = [
    ["Nejvíc kanadských bodů", top("points"), "points"],
    ["Nejlepší střelec", top("goals"), "goals"],
    ["Nejvíc asistencí", top("assists"), "assists"],
    ["Nejvíc střel", top("shots"), "shots"],
    ["Nejvíc trestných minut", top("penaltyMinutes"), "penaltyMinutes"],
  ];
  $("cards").innerHTML = items.map(([l, p, k]) =>
    `<div class="card"><div class="lbl">${l}</div><div class="val">${p[k] ?? 0}</div><div class="who">${name(p)}</div></div>`
  ).join("") +
    `<div class="card"><div class="lbl">Tým celkem</div><div class="val">${tot("goals")} G</div>` +
    `<div class="who">${tot("shots")} střel · ${tot("penaltyMinutes")} TM</div></div>`;
}

function render() {
  renderCards();
  renderTable($("skaters"), SK_COLS, filteredSkaters(), state.sSort, render);
  renderTable($("goalies"), G_COLS, state.goalies, state.gSort, render);
}

function setMsg(text, err) {
  $("cards").innerHTML = "";
  $("skaters").innerHTML = `<tbody><tr><td class="msg ${err ? "err" : ""}">${text}</td></tr></tbody>`;
  $("goalies").innerHTML = "";
}

async function loadStats() {
  if (!state.team) return;
  setMsg("Načítám statistiky…");
  const t = state.teams[state.team];
  $("logo").src = t?.logo || "";
  $("logo").style.display = t?.logo ? "" : "none";
  try {
    const path = state.season === "now"
      ? `club-stats/${state.team}/now`
      : `club-stats/${state.team}/${state.season}/${state.gameType}`;
    const d = await api(path);
    state.skaters = d.skaters || [];
    state.goalies = d.goalies || [];
    const label = (t?.name || state.team);
    $("title").firstChild.textContent = label + " – kanadské bodování";
    $("subtitle").textContent = d.season ? `${fmtSeason(d.season)} · ${d.gameType == 3 ? "play-off" : "základní část"}` : "";
    if (!state.skaters.length && !state.goalies.length) { setMsg("Pro tuto volbu nejsou žádná data."); return; }
    render();
  } catch (e) {
    setMsg(e.message, true);
  }
}

async function loadSeasons() {
  const sel = $("season");
  sel.innerHTML = `<option value="now">Aktuální sezóna</option>`;
  state.season = "now";
  try {
    const list = await api(`club-stats-season/${state.team}`);
    list.sort((a, b) => b.season - a.season).forEach((s) => {
      sel.insertAdjacentHTML("beforeend", `<option value="${s.season}">${fmtSeason(s.season)}</option>`);
    });
  } catch (e) { /* jen aktuální sezóna */ }
}

async function loadTeams() {
  try {
    const d = await api("standings/now");
    (d.standings || []).forEach((t) => {
      const ab = t.teamAbbrev?.default;
      if (ab) state.teams[ab] = { name: t.teamName?.default || ab, logo: t.teamLogo };
    });
  } catch (e) { /* použijeme záložní seznam */ }
  if (!Object.keys(state.teams).length) FALLBACK_TEAMS.forEach((ab) => (state.teams[ab] = { name: ab }));
  const sorted = Object.entries(state.teams).sort((a, b) => a[1].name.localeCompare(b[1].name, "cs"));
  $("team").innerHTML = sorted.map(([ab, t]) => `<option value="${ab}">${t.name} (${ab})</option>`).join("");
  let saved = null;
  try { saved = localStorage.getItem("nhl_team"); } catch (e) {}
  state.team = saved && state.teams[saved] ? saved : sorted[0][0];
  $("team").value = state.team;
}

function segment(id, cb) {
  $(id).querySelectorAll("button").forEach((b) => (b.onclick = () => {
    $(id).querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    cb(b.dataset.v);
  }));
}

segment("gtype", (v) => {
  state.gameType = v;
  if (state.season === "now") {
    // u aktuální sezóny vezmeme konkrétní číslo sezóny, aby šlo přepnout na play-off
    const first = $("season").options[1];
    if (v === "3" && first) { state.season = first.value; $("season").value = first.value; }
  }
  loadStats();
});
segment("posf", (v) => { state.pos = v; render(); });
$("search").oninput = (e) => { state.q = e.target.value; render(); };
$("season").onchange = (e) => {
  state.season = e.target.value;
  if (state.season === "now") {
    state.gameType = "2";
    $("gtype").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x.dataset.v === "2"));
  }
  loadStats();
};
$("team").onchange = async (e) => {
  state.team = e.target.value;
  try { localStorage.setItem("nhl_team", state.team); } catch (err) {}
  await loadSeasons();
  loadStats();
};
$("reload").onclick = loadStats;

(async () => {
  setMsg("Načítám týmy…");
  await loadTeams();
  await loadSeasons();
  loadStats();
})();
</script>
</body>
</html>
"""


def main():
    # Na hostingu (Render apod.) je nastavená proměnná PORT -> běžíme jako web pro všechny.
    if os.environ.get("PORT"):
        port = int(os.environ["PORT"])
        server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        print(f"NHL statistiky běží na portu {port}", flush=True)
        server.serve_forever()
        return

    port = PORT
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        print(f"Port {port} je obsazený. Zkus: python nhl_statistiky.py 8800")
        sys.exit(1)
    url = f"http://localhost:{port}"
    print(f"NHL statistiky běží na {url}")
    print("Ukončíš to klávesami Ctrl+C.")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nKonec.")


if __name__ == "__main__":
    main()
