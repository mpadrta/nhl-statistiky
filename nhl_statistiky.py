#!/usr/bin/env python3
"""
NHL statistiky a předpovědi – kanadské body, góly, střely, vyloučení, kurzy, AI chatbot.

Spuštění na počítači:   python nhl_statistiky.py      (otevře http://localhost:8765)
Na hostingu (Render):   start command  python nhl_statistiky.py

Potřebuje jen Python 3.9+ (žádné další balíčky).
Data: oficiální NHL API (api-web.nhle.com, api.nhle.com/stats).
AI chatbot: nastav proměnnou prostředí GEMINI_API_KEY (zdarma z aistudio.google.com).
Klíč NIKDY nedávej přímo do tohoto souboru – repozitář je veřejný.
"""

import datetime
import json
import math
import os
import re
import sys
import threading
import unicodedata
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from zoneinfo import ZoneInfo
    PRAGUE = ZoneInfo("Europe/Prague")
except Exception:  # pragma: no cover
    PRAGUE = datetime.timezone(datetime.timedelta(hours=1))

PORT = 8765
NHL_API = "https://api-web.nhle.com/v1/"
STATS_REST = "https://api.nhle.com/stats/rest/en/"
GEMINI_API = "https://generativelanguage.googleapis.com/v1beta/"
CACHE_SECONDS = 300
REFRESH_SECONDS = 900  # data ligy se obnovují každých 15 minut
HALF_LIFE = 12.0  # váha zápasů: zápas starý 12 zápasů má poloviční váhu

_cache = {}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
#  Stahování dat
# ---------------------------------------------------------------------------
def http_get(url, data=None, headers=None, timeout=25):
    h = {"User-Agent": "Mozilla/5.0 (nhl-statistiky)", "Accept": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def nhl_get(path, base=NHL_API):
    key = base + path
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1]
    data = None
    for attempt in range(3):
        try:
            data = http_get(key)
            break
        except urllib.error.HTTPError as e:
            if e.code in (404, 400) or attempt == 2:
                raise
        except Exception:
            if attempt == 2:
                raise
        time.sleep(0.6 * (attempt + 1))
    with _cache_lock:
        _cache[key] = (now, data)
        if len(_cache) > 300:
            for k in sorted(_cache, key=lambda k: _cache[k][0])[:150]:
                del _cache[k]
    return data


def get_json(path, base=NHL_API):
    return json.loads(nhl_get(path, base))


# ---------------------------------------------------------------------------
#  Matematika
# ---------------------------------------------------------------------------
def _abbr(x):
    return x.get("default") if isinstance(x, dict) else x


def _toi(s):
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + int(sec)
    except Exception:
        return 0


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _pover(line, lam):
    """P(X > line) – Poisson, line je x.5"""
    return 1 - sum(_pmf(k, lam) for k in range(int(math.floor(line)) + 1))


def _nover(line, mean, sd):
    return 1 - _phi((line - mean) / sd)


def _lines(mean, step=1, count=3):
    c = math.floor(mean) + 0.5
    h = count // 2
    return [round(c + step * i, 1) for i in range(-h, h + 1)]


def _pens_from_pim(pim):
    """Nouzový odhad počtu trestů z trestných minut hráče."""
    return {0: 0, 2: 1, 4: 2, 5: 1, 6: 3, 7: 2, 8: 4, 9: 3}.get(pim % 10, pim // 2 % 5)


def _num(nums):
    try:
        return float(nums)
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Ligová data – všechny odehrané zápasy sezóny
# ---------------------------------------------------------------------------
class League:
    def __init__(self):
        self.lock = threading.Lock()
        self.games = {}  # gid -> kompaktní zápas (trvalá cache)
        self.teams = {}  # abbr -> seznam záznamů (nejnovější první)
        self.info = {}  # abbr -> {name, logo}
        self.roster = {}  # abbr -> {"skaters": {id: name}, "goalies": [...]}
        self.lg = {"shots": 28.5, "goals": 3.05, "pens": 3.4, "hits": 22, "blk": 14, "ppg": 0.7, "ppConv": 0.2}
        self.backtest = None
        self.injuries = {}
        self.special = {}
        self.history = {}  # gid -> předpověď (i pro zápasy, které už začaly/skončily)
        self.preds = {}  # gid -> předpověď
        self.upcoming = []  # nadcházející zápasy
        self.season = None
        self.ready = False
        self.refreshing = False
        self.updated = 0
        self.error = None
        self.state = {"phase": "Připravuji…", "done": 0, "total": 0}

    # -- řízení obnovy -------------------------------------------------------
    def ensure(self):
        with self.lock:
            if self.refreshing or time.time() - self.updated < REFRESH_SECONDS:
                return
            self.refreshing = True
        threading.Thread(target=self._refresh_safe, daemon=True).start()

    def _refresh_safe(self):
        try:
            self.refresh()
            self.error = None
        except Exception as e:
            self.error = str(e)
            self.updated = time.time() - REFRESH_SECONDS + 60  # zkusit znovu za minutu
            print("Chyba obnovy dat:", e, flush=True)
        finally:
            self.refreshing = False

    def status(self):
        return {"ready": self.ready, "refreshing": self.refreshing, "error": self.error,
                "updated": self.updated, **self.state}

    # -- obnova ---------------------------------------------------------------
    def refresh(self):
        st = self.state
        st.update(phase="Stahuji tabulku ligy…", done=0, total=0)
        standings = get_json("standings/now").get("standings", [])
        info = {}
        for t in standings:
            ab = _abbr(t.get("teamAbbrev"))
            if ab:
                info[ab] = {"name": _abbr(t.get("teamName")) or ab, "logo": t.get("teamLogo")}
                self.season = t.get("seasonId") or self.season
        if not info:
            raise RuntimeError("NHL nevrátilo tabulku týmů")

        # vyloučení po zápasech (jedno API pro celou ligu)
        st.update(phase="Stahuji vyloučení všech zápasů…")
        pens, game_ids = self._load_penalties()

        if not game_ids:  # záloha: rozpis každého týmu
            st.update(phase="Stahuji rozpisy týmů…")
            for ab in info:
                try:
                    for g in get_json(f"club-schedule-season/{ab}/now").get("games", []):
                        if g.get("gameType") == 2 and g.get("gameState") in ("FINAL", "OFF"):
                            game_ids.add(g["id"])
                except Exception:
                    pass

        todo = [gid for gid in game_ids if gid not in self.games]
        st.update(phase="Stahuji odehrané zápasy…", done=len(game_ids) - len(todo), total=len(game_ids))

        def load(gid):
            try:
                box = json.loads(http_get(NHL_API + f"gamecenter/{gid}/boxscore"))
                if box.get("gameState") in ("FINAL", "OFF"):
                    self.games[gid] = compact_game(box)
            except Exception:
                pass
            st["done"] += 1

        with ThreadPoolExecutor(max_workers=12) as ex:
            list(ex.map(load, todo))

        st.update(phase="Stahuji soupisky týmů…")
        roster = {}

        def load_roster(ab):
            try:
                d = get_json(f"club-stats/{ab}/now")
                roster[ab] = {
                    "skaters": {s["playerId"]: f"{_abbr(s.get('firstName')) or ''} {_abbr(s.get('lastName')) or ''}".strip()
                                for s in d.get("skaters", [])},
                    "goalies": d.get("goalies", []),
                }
            except Exception:
                roster[ab] = {"skaters": {}, "goalies": []}

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(load_roster, list(info)))

        st.update(phase="Stahuji marodku a týmové statistiky…")
        injuries = load_injuries(info)
        special = load_special(self.season, info)

        st.update(phase="Počítám předpovědi…")
        teams = build_team_records(self.games, pens)
        lg = league_averages(self.games, pens)
        upcoming = load_upcoming(info)

        preds = {}
        for g in upcoming:
            if g["home"] in teams and g["away"] in teams:
                try:
                    preds[g["id"]] = predict_game(g, teams, lg, roster, info, injuries, special)
                except Exception as e:
                    print("Předpověď selhala", g["id"], e, flush=True)
        self.pens = pens

        with self.lock:
            self.info, self.roster, self.teams, self.lg = info, roster, teams, lg
            self.upcoming, self.preds = upcoming, preds
            self.injuries, self.special = injuries, special
            now = time.time()
            for gid, p in preds.items():
                self.history[gid] = (now, p)
            for gid in [k for k, v in self.history.items() if now - v[0] > 3 * 86400]:
                del self.history[gid]
            self.ready = True
            self.updated = time.time()
        st.update(phase="Počítám zpětný test přesnosti…")
        try:
            self.backtest = self.run_backtest(pens)
        except Exception as e:
            print("Zpětný test selhal:", e, flush=True)
        st.update(phase="Hotovo", done=st["total"])
        print(f"Data obnovena: {len(self.games)} zápasů, {len(preds)} předpovědí", flush=True)

    def _load_penalties(self):
        pens, ids = {}, set()
        if not self.season:
            return pens, ids
        exp = f"cayenneExp=seasonId={self.season}%20and%20gameTypeId=2"
        start, total = 0, None
        while total is None or start < total:
            try:
                d = get_json(f"team/penalties?isAggregate=false&isGame=true&start={start}&limit=100&{exp}",
                             base=STATS_REST)
            except Exception:
                break
            rows = d.get("data", [])
            total = d.get("total", 0)
            if not rows:
                break
            for r in rows:
                n = (r.get("minors") or 0) + (r.get("majors") or 0) + \
                    (r.get("benchMinorPenalties") or 0) + (r.get("matchPenalties") or 0)
                pens[(r["gameId"], r["teamId"])] = n
                ids.add(r["gameId"])
            start += len(rows)
        return pens, ids

    def retro(self, gid):
        """Předpověď pro zápas, který už začal/skončil – jen z dat před zápasem."""
        if gid in self.history:
            return self.history[gid][1]
        try:
            box = json.loads(http_get(NHL_API + f"gamecenter/{gid}/boxscore"))
        except Exception:
            return None
        d = box.get("gameDate")
        H, A = box["homeTeam"]["abbrev"], box["awayTeam"]["abbrev"]
        prior = {k: g for k, g in self.games.items() if g["date"] < d}
        if len(prior) < 40:
            return None
        teams = build_team_records(prior, getattr(self, "pens", {}))
        lg = league_averages(prior, getattr(self, "pens", {}))
        info = self.info
        gg = {"id": gid, "home": H, "away": A, "date": d, "start": box.get("startTimeUTC"), "state": box.get("gameState"),
              "venue": _abbr(box.get("venue")),
              "homeName": (info.get(H) or {}).get("name", H), "awayName": (info.get(A) or {}).get("name", A),
              "homeLogo": box["homeTeam"].get("darkLogo") or box["homeTeam"].get("logo"),
              "awayLogo": box["awayTeam"].get("darkLogo") or box["awayTeam"].get("logo"), "odds": {}}
        p = predict_game(gg, teams, lg, self.roster, info, self.injuries, self.special)
        p["retro"] = True
        self.history[gid] = (time.time(), p)
        return p

    # -- zpětný test --------------------------------------------------------------
    def run_backtest(self, pens, days=14):
        games = sorted(self.games.values(), key=lambda g: (g["date"] or "", g["id"]))
        if not games:
            return None
        last = datetime.date.fromisoformat(games[-1]["date"])
        start = (last - datetime.timedelta(days=days - 1)).isoformat()
        dates = sorted({g["date"] for g in games if g["date"] >= start})
        rows, bins = [], {}
        for d in dates:
            prior = {gid: g for gid, g in self.games.items() if g["date"] < d}
            if len(prior) < 40:
                continue
            teams = build_team_records(prior, pens)
            lg = league_averages(prior, pens)
            for g in [x for x in games if x["date"] == d]:
                H, A = g["home"]["abbr"], g["away"]["abbr"]
                if len(teams.get(H, [])) < 5 or len(teams.get(A, [])) < 5:
                    continue
                gg = {"id": g["id"], "home": H, "away": A, "date": d, "start": d, "odds": {}}
                try:
                    p = predict_game(gg, teams, lg, self.roster, self.info)
                except Exception:
                    continue
                hs, as_ = g["home"]["score"], g["away"]["score"]
                ph_pen = pens.get((g["id"], g["home"]["id"]))
                pa_pen = pens.get((g["id"], g["away"]["id"]))
                act_pts = {pl["id"]: pl["pts"] for side in ("home", "away") for pl in g[side]["players"]}
                preds = [pl for pl in p["players"] if pl["active"] and pl["id"] in act_pts]
                for pl in preds:
                    b = min(4, int(pl["pPoint"] * 5))
                    e = bins.setdefault(b, [0, 0.0, 0])
                    e[0] += 1
                    e[1] += pl["pPoint"]
                    e[2] += 1 if act_pts[pl["id"]] > 0 else 0
                top3 = sorted(preds, key=lambda x: -x["pPoint"])[:3]
                rows.append({
                    "id": g["id"], "date": d, "home": H, "away": A,
                    "winP": p["win"]["home"], "homeWon": hs > as_, "score": f"{hs}:{as_}", "end": g["end"],
                    "expShots": p["shots"]["total"]["exp"], "shots": g["home"]["sog"] + g["away"]["sog"],
                    "expPens": p["pens"]["total"]["exp"], "pens": (ph_pen + pa_pen) if ph_pen is not None and pa_pen is not None else None,
                    "expGoals": p["goals"]["total"], "goals": hs + as_,
                    "over55": p["goals"]["totalLines"][1]["over"],
                    "top3": [{"name": t["name"], "team": t["team"], "p": t["pPoint"], "hit": act_pts[t["id"]] > 0} for t in top3],
                })
        if not rows:
            return None
        n = len(rows)
        fav_ok = sum(1 for r in rows if (r["winP"] >= 0.5) == r["homeWon"])
        conf = [r for r in rows if max(r["winP"], 1 - r["winP"]) >= 0.6]
        conf_ok = sum(1 for r in conf if (r["winP"] >= 0.5) == r["homeWon"])
        shots_mae = sum(abs(r["expShots"] - r["shots"]) for r in rows) / n
        pr = [r for r in rows if r["pens"] is not None]
        pens_mae = sum(abs(r["expPens"] - r["pens"]) for r in pr) / max(1, len(pr))
        goals_mae = sum(abs(r["expGoals"] - r["goals"]) for r in rows) / n
        tops = [t for r in rows for t in r["top3"]]
        top1 = [r["top3"][0] for r in rows if r["top3"]]
        brier = sum((r["winP"] - (1 if r["homeWon"] else 0)) ** 2 for r in rows) / n
        return {
            "from": dates[0], "to": dates[-1], "games": n,
            "winner": {"ok": fav_ok, "pct": round(fav_ok / n, 4), "confN": len(conf),
                       "confPct": round(conf_ok / len(conf), 4) if conf else None, "brier": round(brier, 4)},
            "shotsMae": round(shots_mae, 2), "pensMae": round(pens_mae, 2), "goalsMae": round(goals_mae, 2),
            "avgShots": round(sum(r["shots"] for r in rows) / n, 1),
            "top3": {"n": len(tops), "hit": sum(1 for t in tops if t["hit"]), "expected": round(sum(t["p"] for t in tops), 1)},
            "top1": {"n": len(top1), "hit": sum(1 for t in top1 if t["hit"]), "expected": round(sum(t["p"] for t in top1), 1)},
            "calib": [{"bin": b, "n": v[0], "pred": round(v[1] / v[0], 4), "actual": round(v[2] / v[0], 4)} for b, v in sorted(bins.items())],
            "rows": rows[::-1][:60],
        }

    # -- přehled ---------------------------------------------------------------
    def overview(self):
        out = []
        for g in self.upcoming:
            p = self.preds.get(g["id"])
            item = dict(g)
            if p:
                item.update({
                    "win": p["win"], "goals": {k: p["goals"][k] for k in ("home", "away", "total")},
                    "shots": {k: p["shots"][k]["exp"] for k in ("home", "away", "total")},
                    "pens": {k: p["pens"][k]["exp"] for k in ("home", "away", "total")},
                    "top": [{k: pl[k] for k in ("id", "name", "team", "pos", "num", "pPoint", "pGoal", "expShots", "p3Shots", "streak", "last5", "form5", "g", "a", "pts", "gp")}
                            for pl in p["players"] if pl["active"]][:10],
                    "value": p.get("value"),
                })
            out.append(item)
        return {"games": out, "league": self.lg, "updated": self.updated}


LEAGUE = League()


def _norm(name):
    n = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", n).strip()


OUT_STATUSES = ("injured reserve", "out", "long-term injured reserve", "ltir", "suspension", "suspended")


def load_injuries(info):
    """Marodka z ESPN: {abbr: {normalizované jméno: {...}}}"""
    try:
        d = json.loads(http_get("https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"))
    except Exception as e:
        print("Marodka nedostupná:", e, flush=True)
        return {}
    by_name = {_norm(v["name"]): ab for ab, v in info.items()}
    out = {}
    for t in d.get("injuries") or []:
        ab = by_name.get(_norm(t.get("displayName")))
        if not ab:
            continue
        for inj in t.get("injuries") or []:
            a = inj.get("athlete") or {}
            st = str(inj.get("status") or "")
            out.setdefault(ab, {})[_norm(a.get("displayName"))] = {
                "name": a.get("displayName"), "status": st, "out": st.lower() in OUT_STATUSES,
                "comment": inj.get("shortComment") or "", "date": (inj.get("date") or "")[:10],
            }
    return out


def load_special(season, info):
    """Přesilovky, oslabení, vhazování z NHL stats API."""
    if not season:
        return {}
    by_name = {_norm(v["name"]): ab for ab, v in info.items()}
    try:
        d = get_json(f"team/summary?cayenneExp=seasonId={season}%20and%20gameTypeId=2", base=STATS_REST)
    except Exception:
        return {}
    out = {}
    for r in d.get("data", []):
        ab = by_name.get(_norm(r.get("teamFullName")))
        if ab:
            out[ab] = {"pp": r.get("powerPlayPct"), "pk": r.get("penaltyKillPct"), "fo": r.get("faceoffWinPct"),
                       "gf": r.get("goalsForPerGame"), "ga": r.get("goalsAgainstPerGame"),
                       "sf": r.get("shotsForPerGame"), "sa": r.get("shotsAgainstPerGame"), "pts": r.get("points"),
                       "gp": r.get("gamesPlayed"), "w": r.get("wins"), "l": r.get("losses"), "otl": r.get("otLosses")}
    return out


def compact_game(box):
    def side(key):
        t = box[key]
        pst = (box.get("playerByGameStats") or {}).get(key) or {}
        players = []
        for p in (pst.get("forwards") or []) + (pst.get("defense") or []):
            players.append({
                "id": p.get("playerId"), "name": _abbr(p.get("name")), "pos": p.get("position"),
                "num": p.get("sweaterNumber"), "g": p.get("goals") or 0, "a": p.get("assists") or 0, "pts": p.get("points") or 0,
                "sog": p.get("sog") or 0, "pim": p.get("pim") or 0, "toi": _toi(p.get("toi")),
                "hits": p.get("hits") or 0, "ppg": p.get("powerPlayGoals") or 0,
                "blk": p.get("blockedShots") or 0,
            })
        starter = None
        for gk in pst.get("goalies") or []:
            if gk.get("starter"):
                starter = {"id": gk.get("playerId"), "name": _abbr(gk.get("name")),
                           "sa": gk.get("shotsAgainst") or 0, "ga": gk.get("goalsAgainst") or 0}
        return {"abbr": t.get("abbrev"), "id": t.get("id"), "score": t.get("score") or 0,
                "sog": t.get("sog") or 0, "players": players, "starter": starter}

    return {"id": box["id"], "date": box.get("gameDate"),
            "end": (box.get("gameOutcome") or {}).get("lastPeriodType", "REG"),
            "home": side("homeTeam"), "away": side("awayTeam")}


def build_team_records(games, pens):
    teams = {}
    for g in sorted(games.values(), key=lambda g: (g["date"] or "", g["id"]), reverse=True):
        for me_key, op_key in (("home", "away"), ("away", "home")):
            me, op = g[me_key], g[op_key]
            pt = pens.get((g["id"], me["id"]))
            pd = pens.get((g["id"], op["id"]))
            if pt is None:
                pt = sum(_pens_from_pim(p["pim"]) for p in me["players"])
            if pd is None:
                pd = sum(_pens_from_pim(p["pim"]) for p in op["players"])
            teams.setdefault(me["abbr"], []).append({
                "id": g["id"], "date": g["date"], "home": me_key == "home", "opp": op["abbr"],
                "gf": me["score"], "ga": op["score"], "sf": me["sog"], "sa": op["sog"],
                "pt": pt, "pd": pd, "end": g["end"],
                "hits": sum(p["hits"] for p in me["players"]),
                "hitsA": sum(p["hits"] for p in op["players"]),
                "blk": sum(p.get("blk", 0) for p in me["players"]),
                "blkA": sum(p.get("blk", 0) for p in op["players"]),
                "ppgf": sum(p.get("ppg", 0) for p in me["players"]),
                "players": me["players"], "starter": me["starter"],
            })
    return teams


def league_averages(games, pens):
    n = len(games) * 2
    if not n:
        return {"shots": 28.5, "goals": 3.05, "pens": 3.4, "hits": 22, "blk": 14, "ppg": 0.7, "ppConv": 0.2}
    shots = sum(g["home"]["sog"] + g["away"]["sog"] for g in games.values()) / n
    goals = sum(g["home"]["score"] + g["away"]["score"] for g in games.values()) / n
    pv = list(pens.values())
    p = sum(pv) / len(pv) if pv else 3.4
    allp = [pl for g in games.values() for side in ("home", "away") for pl in g[side]["players"]]
    hits = sum(pl["hits"] for pl in allp) / n
    blk = sum(pl.get("blk", 0) for pl in allp) / n
    ppg = sum(pl.get("ppg", 0) for pl in allp) / n
    return {"shots": round(shots, 2), "goals": round(goals, 3), "pens": round(p, 3),
            "hits": round(max(hits, 1), 2), "blk": round(max(blk, 1), 2), "ppg": round(ppg, 3),
            "ppConv": round(ppg / max(p, 0.1), 4)}


REG_BOOKS = {"doxxbet"}  # tyto sázkovky v NHL datech uvádějí kurz na základní hrací dobu (1X2)


def _dec(v):
    v = str(v or "").strip()
    try:
        if v.startswith(("+", "-")) and "." not in v:
            a = int(v)
            return round(1 + (a / 100 if a > 0 else 100 / abs(a)), 2)
        x = float(v)
        return x if x > 1 else None
    except Exception:
        return None


def load_upcoming(info):
    d = get_json("schedule/now")
    tip_id = 3
    partners = {}
    for p in d.get("oddsPartners") or []:
        partners[p.get("partnerId")] = {"name": p.get("name"), "country": p.get("country"), "url": p.get("siteUrl")}
        if str(p.get("name", "")).lower().startswith("tipsport"):
            tip_id = p.get("partnerId", tip_id)
    out = []
    days_used = 0
    for day in d.get("gameWeek", []):
        added = False
        for g in day.get("games", []):
            if g.get("gameType") not in (2, 3) or g.get("gameState") in ("FINAL", "OFF"):
                continue
            ht, at = g["homeTeam"], g["awayTeam"]

            def odd(t):
                for o in t.get("odds") or []:
                    if o.get("providerId") == tip_id:
                        v = _num(o.get("value"))
                        if v and v > 1:
                            return v
                return None

            books = []
            hmap = {o.get("providerId"): _dec(o.get("value")) for o in ht.get("odds") or []}
            amap = {o.get("providerId"): _dec(o.get("value")) for o in at.get("odds") or []}
            for pid, info_p in partners.items():
                if hmap.get(pid) and amap.get(pid):
                    nm = info_p["name"] or str(pid)
                    books.append({"name": nm, "country": info_p["country"], "home": hmap[pid], "away": amap[pid],
                                  "reg": nm.lower() in REG_BOOKS})

            out.append({
                "id": g["id"], "start": g.get("startTimeUTC"), "state": g.get("gameState"),
                "date": day.get("date"), "venue": _abbr(g.get("venue")),
                "home": ht["abbrev"], "away": at["abbrev"],
                "homeName": (info.get(ht["abbrev"]) or {}).get("name") or ht["abbrev"],
                "awayName": (info.get(at["abbrev"]) or {}).get("name") or at["abbrev"],
                "homeLogo": ht.get("darkLogo") or ht.get("logo"),
                "awayLogo": at.get("darkLogo") or at.get("logo"),
                "score": [ht.get("score"), at.get("score")],
                "odds": {"home": odd(ht), "away": odd(at), "provider": "Tipsport", "books": books},
            })
            added = True
        if added:
            days_used += 1
        if days_used >= 3:
            break
    return out


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------
def _wmean(recs, key, venue=None):
    sw = sv = 0.0
    for i, r in enumerate(recs):
        v = r.get(key)
        if v is None or (venue is not None and r["home"] != venue):
            continue
        w = 0.5 ** (i / HALF_LIFE)
        sw += w
        sv += v * w
    return (sv / sw if sw else None), sw


def _shrunk(recs, key, prior, venue=None, k=3.0):
    m, sw = _wmean(recs, key, venue)
    return prior if m is None else (m * sw + prior * k) / (sw + k)


def team_profile(recs, lg):
    pri = {"sf": lg["shots"], "sa": lg["shots"], "gf": lg["goals"], "ga": lg["goals"],
           "pt": lg["pens"], "pd": lg["pens"], "hits": lg.get("hits", 22), "hitsA": lg.get("hits", 22),
           "blk": lg.get("blk", 14), "blkA": lg.get("blk", 14)}
    prof = {}
    for key, prior in pri.items():
        allv = _shrunk(recs, key, prior, k=3)
        prof[key] = {"all": allv,
                     "home": _shrunk(recs, key, allv, venue=True, k=4),
                     "away": _shrunk(recs, key, allv, venue=False, k=4)}
    sf = [r["sf"] for r in recs]
    if len(sf) >= 5:
        m = sum(sf) / len(sf)
        sd = math.sqrt(sum((x - m) ** 2 for x in sf) / (len(sf) - 1))
    else:
        sd = 6.0
    prof["sf_sd"] = _clamp(sd, 4.5, 8.5)
    ppg_sum = sum(r.get("ppgf", 0) for r in recs)
    pd_sum = sum(r["pd"] or 0 for r in recs)
    conv0 = lg.get("ppConv", 0.2)
    prof["ppConv"] = (ppg_sum + conv0 * 25) / (pd_sum + 25)

    def rec_str(rs):
        w = sum(1 for r in rs if r["gf"] > r["ga"])
        l_reg = sum(1 for r in rs if r["gf"] < r["ga"] and r["end"] == "REG")
        return f"{w}-{l_reg}-{len(rs) - w - l_reg}"

    last5 = recs[:5]
    n5 = max(1, len(last5))
    prof["form"] = {
        "last5": rec_str(last5), "home": rec_str([r for r in recs if r["home"]]),
        "away": rec_str([r for r in recs if not r["home"]]), "all": rec_str(recs),
        "gf5": round(sum(r["gf"] for r in last5) / n5, 2), "ga5": round(sum(r["ga"] for r in last5) / n5, 2),
        "sf5": round(sum(r["sf"] for r in last5) / n5, 1), "gp": len(recs),
        "results5": ["V" if r["gf"] > r["ga"] else ("P" if r["end"] == "REG" else "PP") for r in last5],
    }
    prof["trend"] = [{"date": r["date"], "opp": r["opp"], "home": r["home"], "sf": r["sf"], "sa": r["sa"],
                      "gf": r["gf"], "ga": r["ga"], "pt": r["pt"], "pd": r["pd"]} for r in reversed(recs[:10])]
    return prof


def player_projections(team, recs, prof, exp_goals, exp_shots, venue_home, roster_names):
    by_id = {}
    for i, r in enumerate(recs):
        w = 0.5 ** (i / HALF_LIFE)
        for p in r["players"]:
            by_id.setdefault(p["id"], []).append((i, w, r["home"], p))
    ctx_g = _clamp(exp_goals / max(0.5, prof["gf"]["all"]), 0.7, 1.4)
    ctx_s = _clamp(exp_shots / max(10, prof["sf"]["all"]), 0.75, 1.3)
    out = []
    for pid, apps in by_id.items():
        if roster_names and pid not in roster_names:
            continue  # už v týmu není
        last = apps[0][3]
        pos = last["pos"] or "C"
        is_d = pos == "D"
        pri_p, pri_g, pri_s = (0.27, 0.06, 1.4) if is_d else (0.42, 0.16, 1.9)
        sw = sum(a[1] for a in apps)

        def wm(fn, sub=None):
            s = ws = 0.0
            for _, w, h, p in apps:
                if sub is not None and h != sub:
                    continue
                s += fn(p) * w
                ws += w
            return (s / ws if ws else None), ws

        def shr(m, w, prior, k):
            return prior if m is None else (m * w + prior * k) / (w + k)

        ppg = shr(wm(lambda p: p["pts"])[0], sw, pri_p, 3)
        gpg = shr(wm(lambda p: p["g"])[0], sw, pri_g, 3)
        spg = shr(wm(lambda p: p["sog"])[0], sw, pri_s, 3)
        prate = shr(wm(lambda p: 1 if p["pts"] > 0 else 0)[0], sw, 1 - math.exp(-pri_p), 3)
        mv, wv = wm(lambda p: p["pts"], venue_home)
        vf = _clamp(shr(mv, wv, ppg, 6) / ppg, 0.8, 1.25) if ppg > 0 else 1.0
        msv, wsv = wm(lambda p: p["sog"], venue_home)
        vfs = _clamp(shr(msv, wsv, spg, 6) / spg, 0.8, 1.25) if spg > 0 else 1.0
        toi_all = wm(lambda p: p["toi"])[0] or 0
        recent = [a[3]["toi"] for a in apps[:3]]
        toi_rec = sum(recent) / len(recent) if recent else toi_all
        tf = _clamp(math.sqrt(toi_rec / toi_all), 0.85, 1.15) if toi_all > 0 else 1.0
        p_base = 0.5 * (prate + (1 - math.exp(-ppg)))
        p_point = 1 - (1 - p_base) ** (vf * tf * ctx_g)
        lam_g = gpg * vf * tf * ctx_g
        lam_s = spg * vfs * tf * ctx_s
        streak = 0
        for a in apps:
            if a[3]["pts"] > 0:
                streak += 1
            else:
                break
        home_apps = [a[3]["pts"] for a in apps if a[2]]
        away_apps = [a[3]["pts"] for a in apps if not a[2]]
        e = math.exp(-lam_s)
        out.append({
            "id": pid, "name": roster_names.get(pid) or last["name"], "team": team, "pos": pos, "num": last.get("num"),
            "gp": len(apps), "g": sum(a[3]["g"] for a in apps), "a": sum(a[3]["a"] for a in apps),
            "pts": sum(a[3]["pts"] for a in apps),
            "pPoint": round(p_point, 4), "expPoints": round(-math.log(max(1e-6, 1 - p_point)), 3),
            "p2Points": round(1 - (1 - p_point) * (1 - math.log(max(1e-6, 1 - p_point))), 4),
            "pGoal": round(1 - math.exp(-lam_g), 4), "expGoals": round(lam_g, 3),
            "expShots": round(lam_s, 2),
            "p2Shots": round(1 - e * (1 + lam_s), 4),
            "p3Shots": round(1 - e * (1 + lam_s + lam_s ** 2 / 2), 4),
            "p4Shots": round(1 - e * (1 + lam_s + lam_s ** 2 / 2 + lam_s ** 3 / 6), 4),
            "form5": sum(a[3]["pts"] for a in apps[:5]), "last5": [a[3]["pts"] for a in apps[:5]],
            "streak": streak,
            "homePpg": round(sum(home_apps) / len(home_apps), 2) if home_apps else None,
            "awayPpg": round(sum(away_apps) / len(away_apps), 2) if away_apps else None,
            "toi": round(toi_rec), "active": apps[0][0] <= 2, "lastGameAgo": apps[0][0],
        })
    return out


def goalie_info(recs, roster_goalies, game_date):
    starts = [r["starter"] for r in recs[:5] if r.get("starter")]
    counts = {}
    for s in starts:
        counts[s["id"]] = counts.get(s["id"], 0) + 1
    order = [s["id"] for s in starts]
    likely = max(counts, key=lambda k: (counts[k], -order.index(k))) if counts else None
    rest = None
    if recs and game_date:
        try:
            rest = (datetime.date.fromisoformat(game_date) - datetime.date.fromisoformat(recs[0]["date"])).days - 1
        except Exception:
            pass
    goalies = []
    for g in roster_goalies:
        goalies.append({
            "id": g.get("playerId"),
            "name": f"{_abbr(g.get('firstName')) or ''} {_abbr(g.get('lastName')) or ''}".strip(),
            "gp": g.get("gamesPlayed"), "gs": g.get("gamesStarted"), "svp": g.get("savePercentage"),
            "gaa": g.get("goalsAgainstAverage"), "wins": g.get("wins"),
            "starts5": counts.get(g.get("playerId"), 0), "likely": g.get("playerId") == likely,
        })
    goalies.sort(key=lambda g: (-g["likely"], -(g["starts5"] or 0), -(g["gs"] or 0)))
    return {"goalies": goalies[:3], "b2b": rest == 0, "restDays": rest}


def predict_game(g, teams, lg, roster, info, injuries=None, special=None):
    injuries = injuries or {}
    special = special or {}
    home, away, date = g["home"], g["away"], g["date"]
    rh, ra = teams.get(home, []), teams.get(away, [])
    ph, pa = team_profile(rh, lg), team_profile(ra, lg)
    rosH, rosA = roster.get(home, {}), roster.get(away, {})

    sH = ph["sf"]["home"] * pa["sa"]["away"] / lg["shots"]
    sA = pa["sf"]["away"] * ph["sa"]["home"] / lg["shots"]
    gH = ph["gf"]["home"] * pa["ga"]["away"] / lg["goals"]
    gA = pa["gf"]["away"] * ph["ga"]["home"] / lg["goals"]
    nH = ph["pt"]["home"] * pa["pd"]["away"] / lg["pens"]
    nA = pa["pt"]["away"] * ph["pd"]["home"] / lg["pens"]

    gi_h = goalie_info(rh, rosH.get("goalies", []), date)
    gi_a = goalie_info(ra, rosA.get("goalies", []), date)
    if gi_h["b2b"]:
        gH *= 0.96; sH *= 0.98; gA *= 1.03
    if gi_a["b2b"]:
        gA *= 0.96; sA *= 0.98; gH *= 1.03

    grid = [[_pmf(i, gH) * _pmf(j, gA) for j in range(12)] for i in range(12)]
    p_home = sum(grid[i][j] for i in range(12) for j in range(12) if i > j)
    p_away = sum(grid[i][j] for i in range(12) for j in range(12) if i < j)
    p_draw = 1 - p_home - p_away
    share = gH / (gH + gA)
    win_home = p_home + p_draw * share
    win_away = 1 - win_home
    best = max(((i, j) for i in range(8) for j in range(8)), key=lambda t: grid[t[0]][t[1]])
    sd_tot = math.sqrt(ph["sf_sd"] ** 2 + pa["sf_sd"] ** 2)

    def shots_block(lam, sd, step=2):
        return {"exp": round(lam, 1), "sd": round(sd, 3), "mean": round(lam, 3),
                "lines": [{"line": l, "over": round(_nover(l, lam, sd), 4)} for l in _lines(lam, step)]}

    def pois_block(lam):
        return {"exp": round(lam, 2),
                "lines": [{"line": l, "over": round(_pover(l, lam), 4)} for l in _lines(lam, 1)]}

    players = (player_projections(home, rh, ph, gH, sH, True, rosH.get("skaters", {})) +
               player_projections(away, ra, pa, gA, sA, False, rosA.get("skaters", {})))
    inj_list = {home: [], away: []}
    for p in players:
        inj = (injuries.get(p["team"]) or {}).get(_norm(p["name"]))
        if inj:
            p["injury"] = inj
            if inj["out"]:
                p["active"] = False
            inj_list[p["team"]].append({"name": p["name"], "status": inj["status"], "comment": inj["comment"],
                                        "out": inj["out"], "ppg": round(p["pts"] / max(1, p["gp"]), 2), "pts": p["pts"]})
    for t in (home, away):  # zranění, kteří v našich datech nejsou (dlouhodobě mimo)
        known = {_norm(x["name"]) for x in inj_list[t]}
        for k, inj in (injuries.get(t) or {}).items():
            if k not in known:
                inj_list[t].append({"name": inj["name"], "status": inj["status"], "comment": inj["comment"], "out": inj["out"], "ppg": None, "pts": None})
    players.sort(key=lambda p: (-p["active"], -p["pPoint"]))

    # další trhy: hity, bloky, góly v přesilovce, první gól
    lh, lb = lg.get("hits", 22), lg.get("blk", 14)
    hH = ph["hits"]["home"] * pa["hitsA"]["away"] / lh
    hA = pa["hits"]["away"] * ph["hitsA"]["home"] / lh
    bH = ph["blk"]["home"] * pa["blkA"]["away"] / lb
    bA = pa["blk"]["away"] * ph["blkA"]["home"] / lb
    ppH = ph["ppConv"] * nA
    ppA = pa["ppConv"] * nH
    tot = gH + gA
    p_none = math.exp(-tot)
    first = {"home": round(gH / tot * (1 - p_none), 4), "away": round(gA / tot * (1 - p_none), 4), "none": round(p_none, 4)}
    for side_team, gx in ((home, gH), (away, gA)):
        act = [p for p in players if p["team"] == side_team and p["active"]]
        sg = sum(p["expGoals"] for p in act) or 1
        for p in act:
            p["pFirstGoal"] = round((p["expGoals"] / sg) * (gx / tot) * (1 - p_none), 4)
    for p in players:
        p.setdefault("pFirstGoal", 0.0)

    def norm_block(lam, k=1.6, step=2):
        sd = math.sqrt(max(1.0, k * lam))
        return {"exp": round(lam, 1), "sd": round(sd, 3), "mean": round(lam, 3),
                "lines": [{"line": l, "over": round(_nover(l, lam, sd), 4)} for l in _lines(lam, step)]}

    extra = {
        "hits": {"home": norm_block(hH), "away": norm_block(hA), "total": norm_block(hH + hA, step=3)},
        "blocks": {"home": norm_block(bH), "away": norm_block(bA), "total": norm_block(bH + bA, step=2)},
        "ppGoals": {"home": {"exp": round(ppH, 2), "p1": round(1 - math.exp(-ppH), 4)},
                    "away": {"exp": round(ppA, 2), "p1": round(1 - math.exp(-ppA), 4)},
                    "total": {"exp": round(ppH + ppA, 2), "lines": [{"line": l, "over": round(_pover(l, ppH + ppA), 4)} for l in (0.5, 1.5, 2.5)]}},
        "firstGoal": first,
    }

    h2h = [{"date": r["date"], "home": r["home"], "gf": r["gf"], "ga": r["ga"], "sf": r["sf"],
            "sa": r["sa"], "pt": r["pt"], "pd": r["pd"], "end": r["end"]} for r in rh if r["opp"] == away]

    def why(team, opp, p, po, is_home, s, n, gx):
        v, ov = ("home", "away") if is_home else ("away", "home")
        vz, ovz = ("doma", "venku") if is_home else ("venku", "doma")
        return [
            f"{team} {vz} střílí v průměru {p['sf'][v]:.1f} střel, {opp} {ovz} povoluje {po['sa'][ov]:.1f} (liga {lg['shots']:.1f}) → čekáme {s:.1f}.",
            f"{team} {vz} dává {p['gf'][v]:.2f} gólu na zápas, {opp} {ovz} dostává {po['ga'][ov]:.2f} → čekáme {gx:.2f}.",
            f"{team} {vz} jde na trestnou lavici {p['pt'][v]:.1f}× za zápas, soupeři {opp} {ovz} vynucují {po['pd'][ov]:.1f}× → čekáme {n:.1f} vyloučení.",
            f"Forma posledních 5 zápasů: {p['form']['last5']} (V-P-PP), skóre {p['form']['gf5']}:{p['form']['ga5']}, střely {p['form']['sf5']}.",
        ] + ([f"Marodka {team}: " + ", ".join(f"{x['name']} ({x['status']})" for x in (injuries.get(team) or {}).values()) + "."]
             if injuries.get(team) else [])

    odds = g.get("odds") or {}
    value = {}
    for side, prob in (("home", win_home), ("away", win_away)):
        o = odds.get(side)
        if o:
            value[side] = round(prob * o - 1, 4)
    for b in odds.get("books") or []:
        ph_, pa_ = (p_home, p_away) if b.get("reg") else (win_home, win_away)
        b["pHome"], b["pAway"] = round(ph_, 4), round(pa_, 4)
        b["evHome"], b["evAway"] = round(ph_ * b["home"] - 1, 4), round(pa_ * b["away"] - 1, 4)

    return {
        "id": g["id"], "home": home, "away": away, "date": date, "start": g.get("start"),
        "venue": g.get("venue"), "homeName": g.get("homeName"), "awayName": g.get("awayName"),
        "homeLogo": g.get("homeLogo"), "awayLogo": g.get("awayLogo"), "state": g.get("state"),
        "league": lg, "odds": odds, "value": value,
        "win": {"home": round(win_home, 4), "away": round(win_away, 4), "ot": round(p_draw, 4),
                "homeReg": round(p_home, 4), "awayReg": round(p_away, 4)},
        "goals": {"home": round(gH, 2), "away": round(gA, 2), "total": round(gH + gA, 2),
                  "totalLines": [{"line": l, "over": round(_pover(l, gH + gA), 4)} for l in (4.5, 5.5, 6.5, 7.5)],
                  "homeLines": pois_block(gH)["lines"], "awayLines": pois_block(gA)["lines"],
                  "likelyScore": f"{best[0]}:{best[1]}"},
        "shots": {"home": shots_block(sH, ph["sf_sd"]), "away": shots_block(sA, pa["sf_sd"]),
                  "total": shots_block(sH + sA, sd_tot)},
        "pens": {"home": pois_block(nH), "away": pois_block(nA), "total": pois_block(nH + nA)},
        "teams": {
            home: {"form": ph["form"], "trend": ph["trend"], "why": why(home, away, ph, pa, True, sH, nH, gH), **gi_h},
            away: {"form": pa["form"], "trend": pa["trend"], "why": why(away, home, pa, ph, False, sA, nA, gA), **gi_a},
        },
        "h2h": h2h, "players": players, "games": {"home": len(rh), "away": len(ra)},
        "extra": extra,
        "injuries": inj_list,
        "special": {home: special.get(home), away: special.get(away)},
    }


# ---------------------------------------------------------------------------
#  Mapa střel (play-by-play s pozicemi střel)
# ---------------------------------------------------------------------------
_pbp = {}
_pbp_lock = threading.Lock()
_shotmaps = {}


def load_shots(gid):
    """Vrátí [(teamId, x, y, gól)] – souřadnice otočené tak, že útočný tým střílí vždy doprava."""
    with _pbp_lock:
        if gid in _pbp:
            return _pbp[gid]
    try:
        d = json.loads(http_get(NHL_API + f"gamecenter/{gid}/play-by-play"))
    except Exception:
        return []
    home_id = (d.get("homeTeam") or {}).get("id")
    out = []
    for p in d.get("plays", []):
        if p.get("typeDescKey") not in ("shot-on-goal", "goal"):
            continue
        if (p.get("periodDescriptor") or {}).get("periodType") == "SO":
            continue
        det = p.get("details") or {}
        x, y, tid = det.get("xCoord"), det.get("yCoord"), det.get("eventOwnerTeamId")
        if x is None or y is None or tid is None:
            continue
        side = p.get("homeTeamDefendingSide")
        is_home = tid == home_id
        attacking_left = (is_home and side == "right") or (not is_home and side == "left")
        if attacking_left:
            x, y = -x, -y
        out.append((tid, x, y, p.get("typeDescKey") == "goal"))
    if d.get("gameState") in ("FINAL", "OFF"):
        with _pbp_lock:
            _pbp[gid] = out
    return out


def shotmap(team, n_games=10):
    hit = _shotmaps.get(team)
    if hit and time.time() - hit[0] < REFRESH_SECONDS:
        return hit[1]
    recs = LEAGUE.teams.get(team, [])[:n_games]
    gids = [r["id"] for r in recs]
    team_id = None
    for r in recs:
        g = LEAGUE.games.get(r["id"])
        if g:
            team_id = g["home"]["id"] if g["home"]["abbr"] == team else g["away"]["id"]
            break
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(load_shots, gids))
    sf, sa = [], []
    for shots in res:
        for tid, x, y, goal in shots:
            (sf if tid == team_id else sa).append([x, y, 1 if goal else 0])
    data = {"team": team, "games": len(gids), "for": sf, "against": sa}
    _shotmaps[team] = (time.time(), data)
    return data


def player_log(team, pid):
    recs = LEAGUE.teams.get(team, [])
    games = []
    for r in recs:
        for p in r["players"]:
            if p["id"] == pid:
                games.append({"date": r["date"], "opp": r["opp"], "home": r["home"], "g": p["g"], "a": p["a"], "pts": p["pts"], "num": p.get("num"),
                              "sog": p["sog"], "toi": p["toi"], "pim": p["pim"], "hits": p["hits"], "res": "V" if r["gf"] > r["ga"] else "P",
                              "score": f"{r['gf']}:{r['ga']}"})
    games = games[:20]
    name = (LEAGUE.roster.get(team, {}).get("skaters", {}) or {}).get(pid)
    inj = (LEAGUE.injuries.get(team) or {}).get(_norm(name)) if name else None
    nxt = None
    for gp_ in LEAGUE.preds.values():
        for pl in gp_["players"]:
            if pl["id"] == pid:
                nxt = {"game": f"{gp_['home']} – {gp_['away']}", "gid": gp_["id"], "start": gp_.get("start"),
                       "pPoint": pl["pPoint"], "pGoal": pl["pGoal"], "expShots": pl["expShots"], "p2Points": pl.get("p2Points")}
                break
        if nxt:
            break
    n = len(games) or 1
    vs = {}
    for g in games:
        v = vs.setdefault(g["opp"], [0, 0])
        v[0] += 1
        v[1] += g["pts"]
    return {"id": pid, "team": team, "name": name, "injury": inj, "games": games[::-1], "next": nxt,
            "num": next((g["num"] for g in games if g.get("num") is not None), None),
            "avg": {"pts": round(sum(g["pts"] for g in games) / n, 2), "sog": round(sum(g["sog"] for g in games) / n, 2),
                    "toi": round(sum(g["toi"] for g in games) / n), "pointRate": round(sum(1 for g in games if g["pts"]) / n, 3)},
            "home": {"gp": sum(1 for g in games if g["home"]), "pts": sum(g["pts"] for g in games if g["home"])},
            "away": {"gp": sum(1 for g in games if not g["home"]), "pts": sum(g["pts"] for g in games if not g["home"])}}


# ---------------------------------------------------------------------------
#  LIVE zápasy
# ---------------------------------------------------------------------------
_live_cache = {}
_live_lock = threading.Lock()
LIVE_TYPES = ("faceoff", "shot-on-goal", "goal", "missed-shot", "blocked-shot", "hit", "penalty", "takeaway", "giveaway")


def _live_get(path, ttl):
    now = time.time()
    with _live_lock:
        hit = _live_cache.get(path)
        if hit and now - hit[0] < ttl:
            return hit[1]
    data = json.loads(http_get(NHL_API + path))
    with _live_lock:
        _live_cache[path] = (now, data)
        if len(_live_cache) > 80:
            for k in sorted(_live_cache, key=lambda k: _live_cache[k][0])[:40]:
                del _live_cache[k]
    return data


def live_games():
    out = []
    today = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=10)
    for day in (today.date(), today.date() - datetime.timedelta(days=1)):
        try:
            d = _live_get(f"score/{day.isoformat()}", 20)
        except Exception:
            continue
        for g in d.get("games", []):
            if g.get("gameType") not in (2, 3):
                continue
            ht, at = g["homeTeam"], g["awayTeam"]
            out.append({
                "id": g["id"], "state": g.get("gameState"), "start": g.get("startTimeUTC"),
                "home": ht.get("abbrev"), "away": at.get("abbrev"),
                "homeName": _abbr(ht.get("name")) or ht.get("abbrev"), "awayName": _abbr(at.get("name")) or at.get("abbrev"),
                "homeLogo": ht.get("darkLogo") or ht.get("logo"), "awayLogo": at.get("darkLogo") or at.get("logo"),
                "score": [ht.get("score"), at.get("score")], "sog": [ht.get("sog"), at.get("sog")],
                "period": (g.get("periodDescriptor") or {}).get("number"),
                "periodType": (g.get("periodDescriptor") or {}).get("periodType"),
                "clock": g.get("clock"),
            })
    order = {"LIVE": 0, "CRIT": 0, "PRE": 1, "FUT": 2, "FINAL": 3, "OFF": 3}
    seen, res = set(), []
    for g in sorted(out, key=lambda g: (order.get(g["state"], 4), g["start"] or "")):
        if g["id"] not in seen:
            seen.add(g["id"])
            res.append(g)
    return {"games": res}


def live_game(gid):
    pbp = _live_get(f"gamecenter/{gid}/play-by-play", 5)
    try:
        box = _live_get(f"gamecenter/{gid}/boxscore", 8)
    except Exception:
        box = {}
    ht, at = pbp.get("homeTeam") or {}, pbp.get("awayTeam") or {}
    ids = {ht.get("id"): ht.get("abbrev"), at.get("id"): at.get("abbrev")}
    names = {}
    for r in pbp.get("rosterSpots") or []:
        names[r.get("playerId")] = (f"{_abbr(r.get('firstName')) or ''} {_abbr(r.get('lastName')) or ''}".strip(),
                                     r.get("sweaterNumber"), ids.get(r.get("teamId")))

    def nm(pid):
        n = names.get(pid)
        return {"id": pid, "name": n[0], "num": n[1]} if n else None

    periods = {}
    events = []
    last_sit, last_side = "1551", None
    for p in pbp.get("plays") or []:
        t = p.get("typeDescKey")
        pd = p.get("periodDescriptor") or {}
        per = pd.get("number") or 0
        if pd.get("periodType") == "SO":
            continue
        det = p.get("details") or {}
        team = ids.get(det.get("eventOwnerTeamId"))
        if p.get("situationCode"):
            last_sit = p["situationCode"]
        if p.get("homeTeamDefendingSide"):
            last_side = p["homeTeamDefendingSide"]
        if t not in LIVE_TYPES:
            continue
        if team:
            st = periods.setdefault(per, {}).setdefault(team, {"g": 0, "sog": 0, "miss": 0, "blk": 0, "hits": 0, "pen": 0, "pim": 0})
            if t == "goal":
                st["g"] += 1; st["sog"] += 1
            elif t == "shot-on-goal":
                st["sog"] += 1
            elif t == "missed-shot":
                st["miss"] += 1
            elif t == "blocked-shot":
                st["blk"] += 1  # zblokovaná střela tohoto týmu
            elif t == "hit":
                st["hits"] += 1
            elif t == "penalty":
                st["pen"] += 1; st["pim"] += det.get("duration") or 0
        who = None
        extra = None
        if t == "goal":
            who = nm(det.get("scoringPlayerId"))
            extra = [x for x in (nm(det.get("assist1PlayerId")), nm(det.get("assist2PlayerId"))) if x]
        elif t in ("shot-on-goal", "missed-shot", "blocked-shot"):
            who = nm(det.get("shootingPlayerId"))
        elif t == "hit":
            who = nm(det.get("hittingPlayerId")); extra = [x for x in (nm(det.get("hitteePlayerId")),) if x]
        elif t == "penalty":
            who = nm(det.get("committedByPlayerId"))
        elif t == "faceoff":
            who = nm(det.get("winningPlayerId"))
        elif t in ("takeaway", "giveaway"):
            who = nm(det.get("playerId"))
        events.append({
            "id": p.get("eventId"), "type": t, "per": per, "time": p.get("timeInPeriod"), "team": team,
            "x": det.get("xCoord"), "y": det.get("yCoord"), "who": who, "extra": extra,
            "desc": det.get("descKey"), "dur": det.get("duration"), "shotType": det.get("shotType"),
            "reason": det.get("reason"), "sit": p.get("situationCode"),
            "score": [det.get("homeScore"), det.get("awayScore")] if t == "goal" else None,
            "clip": det.get("highlightClipSharingUrl") if t == "goal" else None,
            "side": p.get("homeTeamDefendingSide"),
        })
    players = {}
    for key in ("homeTeam", "awayTeam"):
        pst = (box.get("playerByGameStats") or {}).get(key) or {}
        ab = (box.get(key) or {}).get("abbrev")
        sk = []
        for p in (pst.get("forwards") or []) + (pst.get("defense") or []):
            sk.append({"id": p.get("playerId"), "name": (names.get(p.get("playerId")) or (_abbr(p.get("name")),))[0],
                       "num": p.get("sweaterNumber"), "pos": p.get("position"), "g": p.get("goals", 0), "a": p.get("assists", 0),
                       "pts": p.get("points", 0), "sog": p.get("sog", 0), "pim": p.get("pim", 0), "hits": p.get("hits", 0),
                       "blk": p.get("blockedShots", 0), "pm": p.get("plusMinus", 0), "toi": _toi(p.get("toi"))})
        gk = []
        for p in pst.get("goalies") or []:
            if _toi(p.get("toi")) > 0:
                gk.append({"id": p.get("playerId"), "name": (names.get(p.get("playerId")) or (_abbr(p.get("name")),))[0],
                           "num": p.get("sweaterNumber"), "sa": p.get("shotsAgainst", 0), "sv": p.get("saves", 0),
                           "ga": p.get("goalsAgainst", 0), "toi": _toi(p.get("toi"))})
        players[ab] = {"skaters": sk, "goalies": gk}
    pd = pbp.get("periodDescriptor") or {}
    pred = LEAGUE.preds.get(gid)
    pre = None
    if pred:
        pre = {"shots": {k: pred["shots"][k]["exp"] for k in ("home", "away", "total")},
               "pens": {k: pred["pens"][k]["exp"] for k in ("home", "away", "total")},
               "goals": {k: pred["goals"][k] for k in ("home", "away", "total")},
               "win": pred["win"],
               "players": {pl["id"]: {"pPoint": pl["pPoint"], "expShots": pl["expShots"]} for pl in pred["players"]}}
    return {
        "id": gid, "state": pbp.get("gameState"), "start": pbp.get("startTimeUTC"), "venue": _abbr(pbp.get("venue")),
        "period": pd.get("number"), "periodType": pd.get("periodType"), "clock": pbp.get("clock"),
        "home": {"abbr": ht.get("abbrev"), "name": f"{_abbr(ht.get('placeName')) or ''} {_abbr(ht.get('commonName')) or ''}".strip(),
                 "logo": ht.get("darkLogo") or ht.get("logo"), "score": ht.get("score"), "sog": ht.get("sog")},
        "away": {"abbr": at.get("abbrev"), "name": f"{_abbr(at.get('placeName')) or ''} {_abbr(at.get('commonName')) or ''}".strip(),
                 "logo": at.get("darkLogo") or at.get("logo"), "score": at.get("score"), "sog": at.get("sog")},
        "situation": last_sit, "homeSide": last_side, "events": events, "periods": periods, "players": players, "pre": pre,
        "end": (pbp.get("gameOutcome") or {}).get("lastPeriodType"),
    }


# ---------------------------------------------------------------------------
#  AI chatbot (Google Gemini – bezplatná verze API)
# ---------------------------------------------------------------------------
_gem_models = None
_rate = {}
_rate_lock = threading.Lock()


def gemini_key():
    return os.environ.get("GEMINI_API_KEY", "").strip()


def gemini_models(key):
    global _gem_models
    if _gem_models:
        return _gem_models
    cands = []
    if os.environ.get("GEMINI_MODEL"):
        cands.append(os.environ["GEMINI_MODEL"].strip())
    try:
        d = json.loads(http_get(GEMINI_API + "models?pageSize=200&key=" + urllib.parse.quote(key)))
        names = []
        for m in d.get("models", []):
            n = m.get("name", "").split("/", 1)[-1]
            low = n.lower()
            if "generateContent" not in (m.get("supportedGenerationMethods") or []):
                continue
            if "flash" not in low or any(x in low for x in ("image", "tts", "live", "audio", "embed", "exp", "native")):
                continue
            names.append(n)

        def score(n):
            nums = re.findall(r"\d+(?:\.\d+)?", n)
            ver = float(nums[0]) if nums else 0.0
            return ("preview" not in n, "lite" not in n, ver, "latest" in n)

        names.sort(key=score, reverse=True)
        cands += names
    except Exception as e:
        print("Nelze načíst seznam Gemini modelů:", e, flush=True)
    cands += ["gemini-flash-latest", "gemini-flash-lite-latest"]
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    _gem_models = out
    return out


def rate_ok(ip):
    now = time.time()
    with _rate_lock:
        lst = [t for t in _rate.get(ip, []) if now - t < 600]
        if len(lst) >= 20:
            _rate[ip] = lst
            return False
        lst.append(now)
        _rate[ip] = lst
        glob = [t for t in _rate.get("*", []) if now - t < 86400]
        if len(glob) >= 400:
            _rate["*"] = glob
            return False
        glob.append(now)
        _rate["*"] = glob
    return True


def _local(iso):
    try:
        dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(PRAGUE)
        return dt.strftime("%d.%m. %H:%M")
    except Exception:
        return iso or ""


def _p(v):
    return f"{round(v * 100)} %"


def _fair(v):
    return f"{1 / v:.2f}" if v and v > 0.01 else "–"


def chat_context(gid=None):
    L = LEAGUE
    bt = L.backtest
    lines = [f"Ligový průměr na tým a zápas: střely {L.lg['shots']:.1f}, góly {L.lg['goals']:.2f}, vyloučení {L.lg['pens']:.2f}.",
             (f"Přesnost modelu (zpětný test {bt['games']} zápasů): vítěz trefen v {round(bt['winner']['pct'] * 100)} %, "
              f"odchylka střel ±{bt['shotsMae']}, vyloučení ±{bt['pensMae']}, top tip na bod vyšel {bt['top1']['hit']}/{bt['top1']['n']}." if bt else ""),
             "NADCHÁZEJÍCÍ ZÁPASY (čas Praha; domácí první):"]
    for g in L.upcoming:
        p = L.preds.get(g["id"])
        if not p:
            continue
        o = g.get("odds") or {}
        tops = ", ".join(f"{pl['name']} ({pl['team']}) bod {_p(pl['pPoint'])}, gól {_p(pl['pGoal'])}, střely {pl['expShots']}"
                         for pl in [x for x in p["players"] if x["active"]][:6])
        lines.append(
            f"- [{g['id']}] {_local(g['start'])} {g['home']} ({g['homeName']}) – {g['away']} ({g['awayName']}): "
            f"výhra {g['home']} {_p(p['win']['home'])} / {g['away']} {_p(p['win']['away'])}"
            f"{f', Tipsport kurzy {o.get('home')} / {o.get('away')}' if o.get('home') else ''}; "
            f"oček. góly {p['goals']['home']}:{p['goals']['away']}, střely {p['shots']['home']['exp']}:{p['shots']['away']['exp']} "
            f"(celkem {p['shots']['total']['exp']}), vyloučení {p['pens']['home']['exp']}:{p['pens']['away']['exp']} "
            f"(celkem {p['pens']['total']['exp']}). Top šance na bod: {tops}.")
    if gid and gid in L.preds:
        p = L.preds[gid]
        H, A = p["home"], p["away"]
        lines.append(f"\nDETAIL VYBRANÉHO ZÁPASU {H} – {A} ({_local(p['start'])}):")
        lines.append(f"Výhra (vč. prodl.): {H} {_p(p['win']['home'])} (férový kurz {_fair(p['win']['home'])}), "
                     f"{A} {_p(p['win']['away'])} (férový kurz {_fair(p['win']['away'])}); prodloužení {_p(p['win']['ot'])}; "
                     f"nejpravděpodobnější skóre {p['goals']['likelyScore']}.")
        if p["odds"].get("home"):
            lines.append(f"Tipsport: {H} {p['odds']['home']}, {A} {p['odds']['away']}; hodnota sázky (EV): "
                         + ", ".join(f"{k} {v * 100:+.1f} %" for k, v in p["value"].items()))
        for label, blk in (("Góly celkem", {"lines": p["goals"]["totalLines"]}),
                           (f"Střely {H}", p["shots"]["home"]), (f"Střely {A}", p["shots"]["away"]),
                           ("Střely celkem", p["shots"]["total"]), (f"Vyloučení {H}", p["pens"]["home"]),
                           (f"Vyloučení {A}", p["pens"]["away"]), ("Vyloučení celkem", p["pens"]["total"])):
            exp = f"očekáváno {blk['exp']}; " if "exp" in blk else ""
            lines.append(f"{label}: {exp}" + "; ".join(f"přes {l['line']} {_p(l['over'])}" for l in blk["lines"]))
        for t in (H, A):
            sp = (p.get("special") or {}).get(t) or {}
            if sp:
                lines.append(f"{t} sezóna: přesilovka {_p(sp.get('pp') or 0)}, oslabení {_p(sp.get('pk') or 0)}, vhazování {_p(sp.get('fo') or 0)}.")
            inj = (p.get("injuries") or {}).get(t) or []
            if inj:
                lines.append(f"Marodka {t}: " + "; ".join(f"{i['name']} – {i['status']}{' (' + i['comment'] + ')' if i['comment'] else ''}" for i in inj))
        books = (p.get("odds") or {}).get("books") or []
        if books:
            lines.append("Kurzy sázkovek (domácí/hosté): " + "; ".join(f"{b['name']}{' (základní doba)' if b['reg'] else ''} {b['home']}/{b['away']}" for b in books))
        x = p.get("extra")
        if x:
            lines.append(f"Hity: {H} {x['hits']['home']['exp']}, {A} {x['hits']['away']['exp']}, celkem {x['hits']['total']['exp']}; "
                         f"zblokované střely: {H} {x['blocks']['home']['exp']}, {A} {x['blocks']['away']['exp']}; "
                         f"gól v přesilovce: {H} {_p(x['ppGoals']['home']['p1'])}, {A} {_p(x['ppGoals']['away']['p1'])}; "
                         f"první gól: {H} {_p(x['firstGoal']['home'])}, {A} {_p(x['firstGoal']['away'])}.")
        for t in (H, A):
            ti = p["teams"][t]
            gk = next((x for x in ti["goalies"] if x["likely"]), None)
            lines.append(f"{t}: forma {ti['form']['last5']}, " + " ".join(ti["why"]) +
                         (f" Pravděpodobný brankář {gk['name']} (%Z {gk['svp']}, GAA {gk['gaa']})." if gk else "") +
                         (" Hraje druhý den po sobě." if ti["b2b"] else ""))
        if p["h2h"]:
            lines.append("Vzájemné zápasy letos: " + "; ".join(
                f"{r['date']} {H if r['home'] else A} doma {r['gf'] if r['home'] else r['ga']}:{r['ga'] if r['home'] else r['gf']}"
                for r in p["h2h"]))
        lines.append("Hráči (bod %, gól %, oček. střely, 2+/3+ střely %, body v posl. 5, série, čas na ledě):")
        for pl in [x for x in p["players"] if x["active"]][:30]:
            lines.append(f"  {pl['name']} {pl['team']} {pl['pos']}: bod {_p(pl['pPoint'])} (fér. kurz {_fair(pl['pPoint'])}), "
                         f"gól {_p(pl['pGoal'])}, 2+ body {_p(pl.get('p2Points', 0))}, první gól {_p(pl.get('pFirstGoal', 0))}, střely {pl['expShots']} (2+ {_p(pl['p2Shots'])}, 3+ {_p(pl['p3Shots'])}), "
                         f"forma {pl['form5']} b., série {pl['streak']}, sezóna {pl['g']}+{pl['a']}={pl['pts']} v {pl['gp']} z.")
    return "\n".join(lines)


SYSTEM_PROMPT = """Jsi hokejový analytik NHL na české webové stránce. Odpovídáš česky, stručně a věcně,
jako kamarád, který rozumí statistikám. Vycházej POUZE z dat níže (předpovědi statistického modelu
z odehraných zápasů sezóny a kurzy Tipsportu). Pokud něco v datech není (zranění, sestavy, jiné kurzy),
řekni to na rovinu a nic si nevymýšlej. Čísla uváděj přesně tak, jak jsou v datech.
Férový kurz = 1 / pravděpodobnost. Sázka má hodnotu, když je kurz sázkovky vyšší než férový kurz.
Když se uživatel ptá na sázky, pomoz mu s čísly, ale připomeň jednou větou, že jde o odhad, ne jistotu,
a ať sází jen to, co si může dovolit prohrát. Používej odrážky a tučné písmo (markdown) pro přehlednost.
Dnešní datum a čas (Praha): {now}.

DATA:
{context}"""


_goal_ai = {}


def goal_ai(gid, eid):
    key = (gid, eid)
    if key in _goal_ai:
        return _goal_ai[key]
    if not gemini_key():
        return {"error": "AI není zapnutá."}
    d = live_game(gid)
    evs = d["events"]
    idx = next((i for i, e in enumerate(evs) if e["id"] == eid and e["type"] == "goal"), None)
    if idx is None:
        return {"error": "Gól nenalezen."}
    e = evs[idx]
    home = d["home"]["abbr"]
    # vzdálenost od branky (útočí se na branku na straně soupeře)
    goal_x = 89
    is_home = e["team"] == home
    attack_right = (is_home and e.get("side") == "left") or (not is_home and e.get("side") == "right")
    gx = goal_x if attack_right else -goal_x
    dist = round(math.hypot((e["x"] or 0) - gx, e["y"] or 0) * 0.3048, 1) if e["x"] is not None else None

    def tsec(ev):
        m, s_ = (ev["time"] or "0:0").split(":")
        return (ev["per"] - 1) * 1200 + int(m) * 60 + int(s_)

    t0 = tsec(e)
    before = [x for x in evs[max(0, idx - 8):idx] if t0 - tsec(x) <= 25]
    sit = e.get("sit") or "1551"
    sk_h, sk_a = int(sit[2]), int(sit[1])
    if is_home:
        situation = "přesilovka" if sk_h > sk_a else ("oslabení" if sk_h < sk_a else "plný počet hráčů")
        empty = sit[0] == "0"
    else:
        situation = "přesilovka" if sk_a > sk_h else ("oslabení" if sk_a < sk_h else "plný počet hráčů")
        empty = sit[3] == "0"
    pre = (d.get("pre") or {}).get("players", {})
    scorer = e.get("who") or {}
    sp = pre.get(scorer.get("id")) if scorer else None
    lines = [
        f"Zápas {d['home']['abbr']} – {d['away']['abbr']}, {e['per']}. třetina, čas {e['time']}, gól dal tým {e['team']}.",
        f"Střelec: {scorer.get('name')} (#{scorer.get('num')}), typ střely: {e.get('shotType') or 'neuvedeno'}, vzdálenost od branky asi {dist} m.",
        f"Asistence: {', '.join(a['name'] for a in e.get('extra') or []) or 'bez asistence'}.",
        f"Herní situace: {situation}{', do prázdné branky' if empty else ''}. Stav po gólu {e['score'][0]}:{e['score'][1]} (domácí:hosté)." if e.get("score") else f"Herní situace: {situation}.",
        "Akce posledních 25 sekund před gólem: " + ("; ".join(f"{x['time']} {x['type']} {x['team'] or ''} {(x.get('who') or {}).get('name', '')}" for x in before) or "žádné zaznamenané"),
    ]
    if sp:
        lines.append(f"Model před zápasem dával střelci šanci na bod {_p(sp['pPoint'])} a čekal {sp['expShots']} střel.")
    prompt = ("Vysvětli česky, jednoduše a zajímavě (4–6 vět) jak a proč tento gól padl: odkud se střílelo, jestli šlo o dorážku "
              "(střela těsně před gólem), protiútok (zisk puku/vhazování těsně předtím), přesilovku, kdo se na něm podílel a co to "
              "znamenalo pro zápas. Video nevidíš, vycházej jen z těchto dat a na konci jednou krátkou větou řekni, že popis je z dat zápasu.\n\n"
              + "\n".join(lines))
    res = chat([{"role": "user", "content": prompt}], None, system_override=True)
    if "reply" in res:
        _goal_ai[key] = res
    return res


def chat(messages, gid=None, system_override=False):
    key = gemini_key()
    if not key:
        return {"error": "AI zatím není zapnutá – chybí klíč GEMINI_API_KEY v nastavení Renderu."}
    if not LEAGUE.ready and not system_override:
        return {"error": "Data se ještě načítají, zkus to za chvilku."}
    now = datetime.datetime.now(PRAGUE).strftime("%d.%m.%Y %H:%M")
    system = ("Jsi hokejový komentátor NHL. Odpovídáš česky, přesně podle dat, nic si nevymýšlíš." if system_override
              else SYSTEM_PROMPT.format(now=now, context=chat_context(gid)))
    contents = []
    for m in messages[-12:]:
        role = "model" if m.get("role") == "assistant" else "user"
        text = str(m.get("content", ""))[:2000]
        if text:
            contents.append({"role": role, "parts": [{"text": text}]})
    if not contents or contents[-1]["role"] != "user":
        return {"error": "Prázdná zpráva."}
    body = json.dumps({
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 8192},
    }).encode()
    last_err = None
    for model in gemini_models(key)[:5]:
        url = f"{GEMINI_API}models/{model}:generateContent?key={urllib.parse.quote(key)}"
        try:
            d = json.loads(http_get(url, data=body, headers={"Content-Type": "application/json"}, timeout=60))
            cand = (d.get("candidates") or [{}])[0]
            text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
            if text.strip():
                if cand.get("finishReason") == "MAX_TOKENS":
                    text = text.rstrip() + "\n\n_(odpověď byla moc dlouhá a zkrátila se – zeptej se konkrétněji)_"
                return {"reply": text.strip(), "model": model}
            last_err = "AI vrátila prázdnou odpověď."
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", {}).get("message", "")
            except Exception:
                msg = ""
            last_err = f"{e.code} {msg[:200]}"
            if e.code in (400, 403) and "API key" in msg:
                return {"error": "Klíč GEMINI_API_KEY je neplatný. Zkontroluj ho v nastavení Renderu."}
            continue  # 404 model neexistuje, 429 limit -> zkus další model
        except Exception as e:
            last_err = str(e)
    return {"error": f"AI teď neodpovídá ({last_err}). Zkus to za minutu."}


# ---------------------------------------------------------------------------
#  Tikety z Tipsportu – přečtení (AI), spárování se zápasy, vyhodnocení
# ---------------------------------------------------------------------------
def gemini_generate(contents, system, as_json=False, max_tokens=8192, temperature=0.3):
    key = gemini_key()
    if not key:
        return {"error": "AI zatím není zapnutá – chybí klíč GEMINI_API_KEY."}
    cfg = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if as_json:
        cfg["responseMimeType"] = "application/json"
    body = json.dumps({"systemInstruction": {"parts": [{"text": system}]}, "contents": contents, "generationConfig": cfg}).encode()
    last = None
    for model in gemini_models(key)[:5]:
        url = f"{GEMINI_API}models/{model}:generateContent?key={urllib.parse.quote(key)}"
        try:
            d = json.loads(http_get(url, data=body, headers={"Content-Type": "application/json"}, timeout=90))
            cand = (d.get("candidates") or [{}])[0]
            text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
            if text.strip():
                return {"text": text.strip(), "model": model}
            last = "prázdná odpověď"
        except urllib.error.HTTPError as e:
            try:
                last = f"{e.code} {json.loads(e.read()).get('error', {}).get('message', '')[:160]}"
            except Exception:
                last = str(e.code)
        except Exception as e:
            last = str(e)
    return {"error": f"AI teď neodpovídá ({last})."}


TICKET_SCHEMA = """Vrať POUZE JSON tohoto tvaru:
{"stake": číslo nebo null, "totalOdds": číslo nebo null, "possibleWin": číslo nebo null, "type": "AKO"|"SOLO"|"SYSTEM"|null, "date": "YYYY-MM-DD" nebo null,
 "legs": [{"match": "Domácí - Hosté jak je na tiketu", "home": "domácí tým", "away": "hostující tým", "date": "YYYY-MM-DD" nebo null,
   "market": "název sázky z tiketu", "selection": "tip z tiketu", "odds": číslo,
   "kind": jedno z ["winner_ot","1x2","total_goals","team_goals","total_shots","team_shots","total_pens","total_pim","total_hits",
            "player_point","player_goal","player_assist","player_2points","player_shots","first_goal_team","first_goal_player","pp_goal_team","other"],
   "pick": jedno z ["home","away","1","0","2","10","02","12","over","under","yes","no"],
   "line": číslo nebo null, "team": "home"|"away"|null, "player": "jméno hráče" nebo null,
   "status": "won"|"lost"|"open"|null}]}
Pravidla: "Zápas"/"Výsledek" s tipy 1/0/2/10/02/12 = 1x2 (základní hrací doba). "Vítěz vč. prodloužení a nájezdů"/"Postup" = winner_ot (pick home/away).
"Počet gólů"/"Celkem gólů" = total_goals (pick over pro „více než“/„+“, under pro „méně než“/„-“, line = hranice). Góly jednoho týmu = team_goals + team.
"Střely na branku" celkem = total_shots, jednoho týmu = team_shots. "Počet vyloučení" = total_pens, "Trestné minuty" = total_pim, "Hity" = total_hits.
Hráč „kanadské body/bodový zisk“ = player_point (pick yes, line 0.5), „2+ body“ = player_2points, „gól v zápase/střelec gólu“ = player_goal, „asistence“ = player_assist,
„počet střel hráče“ = player_shots (line), „první gól“ týmu = first_goal_team, hráče = first_goal_player, „gól v přesilovce“ = pp_goal_team.
Jména týmů piš tak, jak jsou na tiketu. Když něco nevidíš, dej null. Nevymýšlej si."""


def _team_index():
    idx = {}
    for ab, v in LEAGUE.info.items():
        full = _norm(v.get("name"))
        words = full.split()
        keys = {ab.lower(), full}
        if len(words) >= 2:
            keys.add(" ".join(words[:-1]))  # „Edmonton“, „Vegas Golden“
            keys.add(words[-1])              # „Oilers“
            keys.add(words[0])
        if full.startswith("new york"):
            keys |= {"ny " + words[-1], "nyr" if "rangers" in full else "nyi"}
        if "golden knights" in full:
            keys |= {"vegas", "golden knights"}
        if "st louis" in full or "st. louis" in full:
            keys |= {"st louis", "st. louis", "louis"}
        if "montr" in full:
            keys |= {"montreal", "canadiens"}
        if "utah" in full:
            keys |= {"utah", "mammoth"}
        if "tampa" in full:
            keys |= {"tampa", "tampa bay"}
        for k in keys:
            if len(k) >= 3:
                idx.setdefault(k, set()).add(ab)
    return idx


def _find_team(name, idx):
    n = _norm(name).replace(".", "")
    if not n:
        return None
    if n in idx and len(idx[n]) == 1:
        return next(iter(idx[n]))
    best = None
    for k, abs_ in idx.items():
        if len(abs_) == 1 and (k in n or n in k) and len(k) >= 4:
            if not best or len(k) > len(best[0]):
                best = (k, next(iter(abs_)))
    return best[1] if best else None


def _candidate_games():
    """Zápasy, ke kterým může sázka patřit: nadcházející, dnešní/včerejší a z posledních 7 dní."""
    games = {}
    for g in LEAGUE.upcoming:
        games[g["id"]] = {"id": g["id"], "home": g["home"], "away": g["away"], "date": g["date"], "state": g["state"], "start": g.get("start")}
    try:
        for g in live_games()["games"]:
            games.setdefault(g["id"], {"id": g["id"], "home": g["home"], "away": g["away"], "date": (g["start"] or "")[:10], "state": g["state"], "start": g.get("start")})
    except Exception:
        pass
    if LEAGUE.games:
        last = max(g["date"] for g in LEAGUE.games.values())
        since = (datetime.date.fromisoformat(last) - datetime.timedelta(days=7)).isoformat()
        for g in LEAGUE.games.values():
            if g["date"] >= since:
                games.setdefault(g["id"], {"id": g["id"], "home": g["home"]["abbr"], "away": g["away"]["abbr"], "date": g["date"], "state": "OFF", "start": None})
    return list(games.values())


def _match_player(name, players):
    n = _norm(name)
    if not n:
        return None
    toks = [t for t in n.split() if len(t) > 1]
    best, score = None, 0
    for p in players:
        pn = _norm(p["name"]).split()
        last = pn[-1] if pn else ""
        sc = 0
        if last and last in toks:
            sc += 2
        if pn and pn[0] in toks:
            sc += 1
        if pn and any(t[0] == pn[0][0] for t in toks if len(t) <= 2):
            sc += 0.5
        if sc > score:
            best, score = p, sc
    return best if score >= 2 else None


def _leg_probability(leg, p):
    """Šance podle modelu pro jednu sázku. Vrací (šance, popis, hráč)."""
    kind, pick, line = leg.get("kind"), str(leg.get("pick") or ""), leg.get("line")
    try:
        line = float(line) if line is not None else None
    except (TypeError, ValueError):
        line = None
    side = leg.get("team") or ("home" if pick == "home" else "away" if pick == "away" else None)
    w = p["win"]
    over = pick in ("over", "yes")

    def ov(prob):
        return prob if over else 1 - prob

    if kind == "winner_ot":
        return (w["home"] if pick == "home" else w["away"] if pick == "away" else None), "vítěz vč. prodloužení", None
    if kind == "1x2":
        m = {"1": w["homeReg"], "0": w["ot"], "2": w["awayReg"], "10": w["homeReg"] + w["ot"], "02": w["awayReg"] + w["ot"],
             "12": 1 - w["ot"], "home": w["homeReg"], "away": w["awayReg"]}
        return m.get(pick), "výsledek po 60 minutách", None
    if line is None and kind in ("total_goals", "team_goals", "total_shots", "team_shots", "total_pens", "total_pim", "total_hits", "player_shots"):
        return None, "chybí hranice", None
    if kind == "total_goals":
        return ov(_pover(line, p["goals"]["total"])), f"čekáme {p['goals']['total']} gólů", None
    if kind == "team_goals" and side:
        return ov(_pover(line, p["goals"][side])), f"čekáme {p['goals'][side]} gólů týmu", None
    if kind in ("total_shots", "team_shots"):
        b = p["shots"]["total" if kind == "total_shots" else (side or "total")]
        return ov(_nover(line, b.get("mean", b["exp"]), b.get("sd", 6))), f"čekáme {b['exp']} střel", None
    if kind == "total_pens":
        return ov(_pover(line, p["pens"]["total"]["exp"])), f"čekáme {p['pens']['total']['exp']} vyloučení", None
    if kind == "total_pim":
        lam = p["pens"]["total"]["exp"] * 2.3
        return ov(_nover(line, lam, math.sqrt(lam * 3.2))), f"čekáme asi {lam:.1f} trestných minut (přibližně)", None
    if kind == "total_hits" and p.get("extra"):
        b = p["extra"]["hits"]["total"]
        return ov(_nover(line, b["mean"], b["sd"])), f"čekáme {b['exp']} hitů", None
    if kind == "first_goal_team" and p.get("extra"):
        fg = p["extra"]["firstGoal"]
        return (fg["home"] if side == "home" or pick == "home" else fg["away"]), "první gól", None
    if kind == "pp_goal_team" and p.get("extra"):
        pp = p["extra"]["ppGoals"]["home" if side == "home" else "away"]["p1"]
        return (pp if pick != "no" else 1 - pp), "gól v přesilovce", None
    if kind and kind.startswith(("player_", "first_goal_player")):
        pl = _match_player(leg.get("player") or leg.get("selection"), p["players"])
        if not pl:
            return None, "hráče se nepodařilo najít", None
        lam_s = pl["expShots"]
        if kind == "player_point":
            pr = pl["pPoint"]
        elif kind == "player_goal":
            pr = pl["pGoal"]
        elif kind == "player_2points":
            pr = pl.get("p2Points")
        elif kind == "player_assist":
            pr = 1 - math.exp(-max(0.01, pl.get("expPoints", 0) - pl.get("expGoals", 0)))
        elif kind == "player_shots":
            pr = ov(_pover(line, lam_s))
        else:
            pr = pl.get("pFirstGoal")
        if kind in ("player_point", "player_goal", "player_2points", "player_assist", "first_goal_player") and pick == "no":
            pr = 1 - pr
        return pr, f"{pl['name']}: bod {round(pl['pPoint'] * 100)} %, střely {lam_s}", pl
    return None, "tento typ sázky model neumí spočítat", None


def _leg_result(leg, live, pl):
    """Vyhodnocení odehrané sázky ze skutečných dat. True/False/None."""
    kind, pick, line = leg.get("kind"), str(leg.get("pick") or ""), leg.get("line")
    try:
        line = float(line) if line is not None else None
    except (TypeError, ValueError):
        line = None
    H, A = live["home"]["abbr"], live["away"]["abbr"]
    hs, as_ = live["home"]["score"] or 0, live["away"]["score"] or 0
    evs = live["events"]
    reg = [sum(1 for e in evs if e["type"] == "goal" and e["per"] <= 3 and e["team"] == t) for t in (H, A)]
    side = leg.get("team") or ("home" if pick == "home" else "away" if pick == "away" else None)
    over = pick in ("over", "yes")

    def cmp(v):
        return None if line is None else ((v > line) if over else (v < line))

    if kind == "winner_ot":
        return (hs > as_) if pick == "home" else (as_ > hs) if pick == "away" else None
    if kind == "1x2":
        r = "1" if reg[0] > reg[1] else "2" if reg[1] > reg[0] else "0"
        return r in {"1": "1", "0": "0", "2": "2", "10": "10", "02": "02", "12": "12", "home": "1", "away": "2"}.get(pick, "")
    if kind == "total_goals":
        return cmp(hs + as_ - (1 if live.get("end") == "SO" else 0))
    if kind == "team_goals" and side:
        v = (hs if side == "home" else as_) - (1 if live.get("end") == "SO" and ((side == "home") == (hs > as_)) else 0)
        return cmp(v)
    sog = {H: live["home"]["sog"] or 0, A: live["away"]["sog"] or 0}
    if kind == "total_shots":
        return cmp(sog[H] + sog[A])
    if kind == "team_shots" and side:
        return cmp(sog[H if side == "home" else A])
    pens = [e for e in evs if e["type"] == "penalty" and (e.get("desc") or "") not in ("misconduct", "game-misconduct")]
    if kind == "total_pens":
        return cmp(len(pens))
    if kind == "total_pim":
        return cmp(sum(e.get("dur") or 0 for e in evs if e["type"] == "penalty"))
    if kind == "total_hits":
        return cmp(sum(1 for e in evs if e["type"] == "hit"))
    if kind == "first_goal_team":
        fg = next((e for e in evs if e["type"] == "goal"), None)
        return None if not fg else (fg["team"] == (H if (side or pick) == "home" else A))
    if pl is not None:
        box = next((x for t in live["players"].values() for x in t["skaters"] if x["id"] == pl["id"]), None)
        if not box:
            return False  # nehrál
        if kind == "player_point":
            return (box["pts"] > 0) == (pick != "no")
        if kind == "player_goal":
            return (box["g"] > 0) == (pick != "no")
        if kind == "player_assist":
            return (box["a"] > 0) == (pick != "no")
        if kind == "player_2points":
            return (box["pts"] >= 2) == (pick != "no")
        if kind == "player_shots":
            return cmp(box["sog"])
        if kind == "first_goal_player":
            fg = next((e for e in evs if e["type"] == "goal"), None)
            return bool(fg and fg.get("who") and fg["who"]["id"] == pl["id"])
    return None


def analyze_ticket(image_b64=None, mime="image/jpeg", text=None):
    if not LEAGUE.ready:
        return {"error": "Data se ještě načítají, zkus to za minutu."}
    parts = [{"text": "Přečti tento tiket z Tipsportu (NHL hokej) a převeď ho do JSON. " + TICKET_SCHEMA}]
    if image_b64:
        parts.append({"inline_data": {"mime_type": mime, "data": image_b64}})
    if text:
        parts.append({"text": "Text tiketu:\n" + text[:4000]})
    r = gemini_generate([{"role": "user", "parts": parts}], "Jsi přesný čtenář sázkových tiketů. Odpovídáš jen validním JSON.", as_json=True, temperature=0.1)
    if "error" in r:
        return r
    try:
        raw = r["text"].strip()
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
        tk = json.loads(raw)
    except Exception:
        return {"error": "AI tiket nepřečetla. Zkus ostřejší screenshot, nebo text tiketu vlož ručně."}
    legs = tk.get("legs") or []
    if not legs:
        return {"error": "Na obrázku jsem nenašel žádné sázky na NHL. Je to opravdu tiket?"}
    idx = _team_index()
    cands = _candidate_games()
    out = []
    for leg in legs[:30]:
        H = _find_team(leg.get("home") or (leg.get("match") or "").split("-")[0], idx)
        A = _find_team(leg.get("away") or ((leg.get("match") or "") + "-").split("-")[1], idx)
        game = None
        opts = [g for g in cands if {g["home"], g["away"]} == {H, A}] if H and A else []
        if opts:
            d = leg.get("date") or tk.get("date")
            opts.sort(key=lambda g: (0 if d and g["date"] == d else 1, 0 if g["state"] not in ("OFF", "FINAL") else 1, g["date"] or ""))
            game = opts[0]
            if game["home"] != H:  # tiket má týmy obráceně
                leg["pick"] = {"home": "away", "away": "home", "1": "2", "2": "1", "10": "02", "02": "10"}.get(str(leg.get("pick")), leg.get("pick"))
                if leg.get("team") in ("home", "away"):
                    leg["team"] = "away" if leg["team"] == "home" else "home"
        item = {"leg": leg, "gameId": game["id"] if game else None, "home": game["home"] if game else H, "away": game["away"] if game else A,
                "state": game["state"] if game else None, "date": game["date"] if game else None}
        if game:
            pred = LEAGUE.preds.get(game["id"])
            if not pred:
                try:
                    pred = LEAGUE.retro(game["id"])
                except Exception:
                    pred = None
            if pred:
                prob, note, pl = _leg_probability(leg, pred)
                item.update(prob=prob, note=note, player=pl and {"id": pl["id"], "name": pl["name"], "team": pl["team"]})
                odds = leg.get("odds")
                if prob and odds:
                    item["fair"] = round(1 / prob, 2) if prob > 0.01 else None
                    item["ev"] = round(prob * float(odds) - 1, 4)
                if pl and pl.get("injury"):
                    item["warn"] = f"{pl['name']}: {pl['injury']['status']} – {pl['injury']['comment']}"
                if pl and not pl.get("active"):
                    item["warn"] = (item.get("warn") or "") + f" Hráč v posledních zápasech nehrál."
            if game["state"] in ("OFF", "FINAL"):
                try:
                    lv = live_game(game["id"])
                    if lv.get("state") in ("OFF", "FINAL"):
                        item["result"] = _leg_result(leg, lv, item.get("player") and {"id": item["player"]["id"]})
                        item["finalScore"] = f"{lv['home']['score']}:{lv['away']['score']}" + (f" {lv.get('end')}" if lv.get("end") not in (None, "REG") else "")
                except Exception as e:
                    print("výsledek tiketu:", e, flush=True)
        else:
            item["note"] = "zápas se nepodařilo najít v datech NHL (jen posledních 7 dní a nadcházející zápasy)"
        out.append(item)
    probs = [x.get("prob") for x in out]
    p_all = math.prod(probs) if all(probs) else None
    odds_all = tk.get("totalOdds") or (math.prod(float(x["leg"]["odds"]) for x in out if x["leg"].get("odds")) if all(x["leg"].get("odds") for x in out) else None)
    results = [x.get("result") for x in out]
    status = "lost" if any(r is False for r in results) else "won" if results and all(r is True for r in results) else "open"
    # návrhy: nejsilnější sázky z modelu ve stejných zápasech
    sugg = []
    for gid in {x["gameId"] for x in out if x["gameId"]}:
        pred = LEAGUE.preds.get(gid) or (LEAGUE.history.get(gid) or (0, None))[1]
        if not pred or pred.get("retro"):
            continue
        H, A = pred["home"], pred["away"]
        for pl in [x for x in pred["players"] if x["active"]][:3]:
            sugg.append({"game": f"{H} – {A}", "text": f"{pl['name']} ({pl['team']}) – kanadský bod", "prob": pl["pPoint"]})
        fav = "home" if pred["win"]["home"] >= pred["win"]["away"] else "away"
        sugg.append({"game": f"{H} – {A}", "text": f"Vítěz vč. prodloužení – {H if fav == 'home' else A}", "prob": pred["win"][fav],
                     "odds": (pred.get("odds") or {}).get(fav)})
        tl = max(pred["shots"]["total"]["lines"], key=lambda l: abs(l["over"] - 0.5))
        sugg.append({"game": f"{H} – {A}", "text": f"Střely celkem – {'přes' if tl['over'] > 0.5 else 'pod'} {tl['line']}", "prob": max(tl["over"], 1 - tl["over"])})
    sugg.sort(key=lambda x: -x["prob"])
    summary = {"stake": tk.get("stake"), "totalOdds": odds_all, "possibleWin": tk.get("possibleWin"), "type": tk.get("type"),
               "pAll": p_all, "ev": (p_all * float(odds_all) - 1) if p_all and odds_all else None, "status": status}
    # AI komentář
    lines = []
    for i, x in enumerate(out, 1):
        lg = x["leg"]
        lines.append(f"{i}. {lg.get('match')} | {lg.get('market')}: {lg.get('selection')} @ {lg.get('odds')} | model: "
                     + (f"{round(x['prob'] * 100)} % (férový kurz {x.get('fair')}, hodnota {round((x.get('ev') or 0) * 100)} %)" if x.get("prob") else f"nelze ({x.get('note')})")
                     + (f" | {x['note']}" if x.get('prob') and x.get('note') else "")
                     + (f" | VAROVÁNÍ: {x['warn']}" if x.get("warn") else "")
                     + (f" | výsledek: {'VYŠLO' if x['result'] else 'NEVYŠLO'} (skóre {x.get('finalScore')})" if x.get("result") is not None else ""))
    ctx = (f"Tiket: typ {summary['type']}, vklad {summary['stake']}, celkový kurz {summary['totalOdds']}, šance podle modelu "
           f"{round(p_all * 100, 1) if p_all else '?'} %, hodnota {round(summary['ev'] * 100) if summary['ev'] is not None else '?'} %, stav {status}.\n"
           + "\n".join(lines) + "\nSilné sázky z modelu ve stejných zápasech: "
           + "; ".join(f"{s_['game']}: {s_['text']} ({round(s_['prob'] * 100)} %)" for s_ in sugg[:6]))
    com = gemini_generate([{"role": "user", "parts": [{"text": ctx}]}],
                          "Jsi zkušený, ale poctivý hokejový analytik. Česky a stručně (max ~180 slov) zhodnoť tiket: které sázky jsou podle modelu "
                          "dobré a které slabé (s čísly), na co pozor (zranění, sázky ze stejného zápasu spolu souvisí), a navrhni konkrétní úpravu tiketu "
                          "(co vyhodit / čím nahradit – jen z uvedených dat). Pokud už je tiket vyhodnocený, okomentuj co vyšlo a co ne. "
                          "Použij odrážky a tučné písmo. Na konci jednou větou připomeň, že jde o odhad a ať sází jen to, co si může dovolit prohrát.",
                          temperature=0.4, max_tokens=4096)
    return {"ticket": {k: tk.get(k) for k in ("stake", "totalOdds", "possibleWin", "type", "date")}, "legs": out, "summary": summary,
            "suggestions": sugg[:8], "comment": com.get("text"), "commentError": com.get("error")}


# ---------------------------------------------------------------------------
#  Webový server
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code=200):
        self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if u.path == "/model/status":
            LEAGUE.ensure()
            self._json({**LEAGUE.status(), "ai": bool(gemini_key())})
            return
        if u.path == "/model/overview":
            LEAGUE.ensure()
            if not LEAGUE.ready:
                self._json({"loading": True, **LEAGUE.status()})
                return
            self._json({**LEAGUE.overview(), "ai": bool(gemini_key()), "status": LEAGUE.status()})
            return
        if u.path == "/model/livegames":
            try:
                self._json(live_games())
            except Exception as e:
                self._json({"error": f"Nepodařilo se načíst zápasy: {e}"}, 502)
            return
        if u.path == "/model/live":
            try:
                self._json(live_game(int(q.get("id", ["0"])[0])))
            except Exception as e:
                self._json({"error": f"Zápas se nepodařilo načíst: {e}"}, 502)
            return
        if u.path == "/model/backtest":
            LEAGUE.ensure()
            self._json(LEAGUE.backtest or {"loading": True, **LEAGUE.status()})
            return
        if u.path == "/manifest.webmanifest":
            self._send(200, MANIFEST.encode(), "application/manifest+json")
            return
        if u.path in ("/icon-192.png", "/icon-512.png", "/apple-touch-icon.png"):
            self._send(200, ICONS[u.path], "image/png")
            return
        if u.path == "/sw.js":
            self._send(200, SW_JS.encode(), "application/javascript")
            return
        if u.path == "/ping":
            LEAGUE.ensure()
            self._send(200, b"ok", "text/plain")
            return
        if u.path == "/model/player":
            try:
                pid = int(q.get("id", ["0"])[0])
            except ValueError:
                pid = 0
            team = q.get("team", [""])[0].upper()
            self._json(player_log(team, pid))
            return
        if u.path == "/img":
            src = q.get("u", [""])[0]
            if not src.startswith("https://assets.nhle.com/"):
                self._send(400, b"bad", "text/plain")
                return
            try:
                data = nhl_get(src[len("https://assets.nhle.com/"):], base="https://assets.nhle.com/")
                ctype = "image/svg+xml" if src.endswith(".svg") else "image/png"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "public, max-age=86400")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception:
                self._send(404, b"", "text/plain")
            return
        if u.path == "/model/injuries":
            self._json(LEAGUE.injuries)
            return
        if u.path == "/model/shotmap":
            team = q.get("team", [""])[0].upper()
            if not (team.isalpha() and len(team) == 3) or not LEAGUE.ready:
                self._json({"error": "Data nejsou připravená."}, 400)
                return
            self._json(shotmap(team))
            return
        if u.path == "/model/predict":
            LEAGUE.ensure()
            try:
                gid = int(q.get("id", ["0"])[0])
            except ValueError:
                gid = 0
            p = LEAGUE.preds.get(gid)
            if not p and LEAGUE.ready:
                try:
                    p = LEAGUE.retro(gid)
                except Exception as e:
                    print("retro selhalo", gid, e, flush=True)
            if p:
                self._json(p)
            elif not LEAGUE.ready:
                self._json({"loading": True, **LEAGUE.status()})
            else:
                self._json({"error": "Pro tento zápas se nepodařilo spočítat předpověď."}, 404)
            return
        if u.path.startswith("/api/"):
            path = self.path[len("/api/"):]
            if not all(c.isalnum() or c in "/-_." for c in path):
                self._json({"error": "bad path"}, 400)
                return
            try:
                self._send(200, nhl_get(path), "application/json; charset=utf-8")
            except urllib.error.HTTPError as e:
                text = ("Pro tuto volbu nejsou žádná data (třeba play-off ještě nezačalo)."
                        if e.code == 404 else f"NHL API vrátilo chybu {e.code}")
                self._json({"error": text}, 502)
            except Exception as e:
                hint = " (Na Macu spusť 'Install Certificates.command' ze složky Pythonu.)" if "CERTIFICATE" in str(e).upper() else ""
                self._json({"error": f"Nepodařilo se stáhnout data: {e}{hint}"}, 502)
            return
        self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        if self.path not in ("/model/chat", "/model/goalai", "/model/ticket"):
            self._send(404, b"Not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = 0
        limit = 9_000_000 if self.path == "/model/ticket" else 60000
        if n <= 0 or n > limit:
            self._json({"error": "Zpráva nebo obrázek je moc velký (max. 6 MB)."}, 400)
            return
        try:
            body = json.loads(self.rfile.read(n))
        except Exception:
            self._json({"error": "Špatný formát."}, 400)
            return
        ip = (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()
        if not rate_ok(ip):
            self._json({"error": "Moc zpráv za krátkou dobu. Zkus to za pár minut."}, 429)
            return
        gid = body.get("gameId")
        try:
            gid = int(gid) if gid else None
        except (TypeError, ValueError):
            gid = None
        if self.path == "/model/ticket":
            img, mime = body.get("image"), "image/jpeg"
            if img and img.startswith("data:"):
                head, img = img.split(",", 1)
                mime = head[5:].split(";")[0] or mime
            try:
                res = analyze_ticket(img, mime, body.get("text"))
            except Exception as e:
                res = {"error": f"Vyhodnocení se nepovedlo: {e}"}
            self._json(res, 200 if "legs" in res else 503)
            return
        if self.path == "/model/goalai":
            try:
                res = goal_ai(gid, int(body.get("eventId")))
            except Exception as e:
                res = {"error": f"Rozbor se nepovedl: {e}"}
            self._json(res, 200 if "reply" in res else 503)
            return
        res = chat(body.get("messages") or [], gid)
        self._json(res, 200 if "reply" in res else 503)


# Ikony aplikace (PNG, base64)
import base64 as _b64
ICONS = {
    "/icon-192.png": _b64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAMAAAADACAIAAADdvvtQAAAItElEQVR42u2dWYwURRyHq6bHhSB4xA0bmUEJiMPyJIiiKKCAkSheKGp480IBQQQPlkV2ORdQUUDk8EB98oLFI5JoEKOoICrEKLBEJVwaUEkwCi7MTPswiOzsHH3OVHV/X/qJXaZmu7/51b+6qmtkm9MrRano9e4+ASVhy43x0jQkfRWo1zsYo4ZPN8V1Eqg33qjKt16b5LFAqBM2jTwT6OI1qKMZ39wcV0Ig1AmzRq4E6oM6QeFrpxo5F+gS7AkWmx055ESgSxpRJ7ga3RL3V6BLsSfofGXHoQj2gJurbCOB+mJPmNhkLYesCtR3NfaEz6HhcW8Eugx7wsrGYg4VF+hy7Ak3XxZ0KII94MaBQgnUbxX2wAm+uDVuL4GwB6z4EM0bTZwzEE5vJF5B/IA1K3LUQFe+jT2Qlw23xYt0YXRe4LwL60/8QEGyDIkSPyC8WpE44C3iByzx6Yh4jgQifsB5Al1F/IAdPhkRb5lAJucEnCbQ1W8SP2Cb9bfHI5wFcEOU8hncdmGD3qD/AhcJRPwAAoFQaDkHgJ0E4g4QOC6ih733D2cBnCcQBRBQA0EZE4gCCOjCgC4MSCAI430glWqgef1Pu/Ds3KGYNsW49cd++avI231taEWHihwfihkbj285mLbV6G9HzVEfHSv6ngd1Nsb1yv18Zr5GXbaoVhcmhVDnKPxGR/aIOn4FZ426fNv+tajOoVYXVvjN9OsUaTxT7jpsOntlab9R6frPkaVtkSK6yPUY2SNK2cF9IOf0ropUnx3ZcShtWz7T/pSftf8iPWzU1G9eUr9h/Mhqg889w3jn9KyM9OoY2Xow7W2B5VNFIkveYqiH8Ra5szq69YD94a6Dv9R0/Tum/2+SBLJLt7PkZZ0im35JezIKc5kH0tNGJV1YaQL/juro5l+PmWb5h/EOBGIYX2qSrbIm3kEO6Ew1zTDeGut3p4Z0MWTLj+dtPYzP96ZSab9GyNLiYl/Ts0alYBjvD/v+NDfsTWX9Y8d2cnAXQogayNpH8+3tqX5xw2gp/PCE8enuVHOqeA3hoJ6tbCdfv6WNH1MZghqo9Bw8Yn68OzuEzmorr+1GCFEDWagnpCkat6cGnmdUtBTmhu7RdT+njhx3MZXh2xnI22jBP5PlHN4v58j89PA/5oc/ZYdQ+woxrHtU+rOcQ3gxG2+rRe2Wc2g2F/bezmTrsBl6gXFGG1ZWkkAWIuTvY2Ltj8msH7WNipsShpsPtzohpN+CMj0mX8z/64a1O1PXdM2OnMFdjQ92pg4dNa28gkV+P2JOWFt8xm3A+caoPlGvGhXUQP7FT+ZoTor3m7IrodMiYni1IcuxpDXsNZB2Akkh1v2cI2z6dzHObS9ZE00RXZzjKfHO9uwQMqQY3pMFr8yFWbtB8tmu1HUXGlUtI6dv50ja01ssLpe0OmiUubASkTJF47Zk62tpMJxnLszipNKmPenrE2bnM6V/01I+PdbjX4skkL0B7+ofkmQANZCFeiLPpNLW/emf/kh3Oyfi5rEe6bIGMm03KgNUA2nchWVY9X3qsYGRMq6J9rDRc9rJlSNsLCAZ3djcnKQLc8eOg+lt9h/xgZAt5yi4NnT1d8meQyqcL2k1/erChP3lHL4/bksCtWbXIXPLfkKIGshFMbHm++RFnSqk1P65MKHbsD8gW9ztP2xu2pMiD8rw2b73dTYaB/ZIBLowIIGAXVoBSCCgBgISCJgLA6ALA7owIIGAGgiALgzowoAEAubCAEggoAaCMicQGQTcBwK6MKCIBobxACQQUAMBCQQM44PN0ocTXr3U6GeawnkO5SOvNiOKfwReLPlo4AR63rIulVVVXjX6+4EDFn9zTLCUCohABaTx0BLP3QqATPKxV3QVaMnEhJrGOPNp7AItZZKP6ybQc628UdwYuz49qJVJOgmUpY7W3hQ1SReN5GQdBFp8ijoB86awSeOU10hOXqm0QIsnhUidvBo9ra5GskZhgRb9Z0+o1Mmp0XhVHYoobk9lVVVo7Tn1z180KcF3pto4Ttoj4L/zsGhSQsnvTDWFasdC7Mnj0MJJCdUulnJd2LOPYE8hhzLnR7CcA7z/VjWKaBCsB4ISf4E1CQQsaRXe30Cjjs53U1HSheGQm5kNujDvl/mFxx66MCfnjrkwZYfxUY3OY2hn4xnGo5GX6ih1yfS7E33y/AZ7RaIud6KjATjjAVsTLbgPVN5roOlTGULPubAAbu+imk/eZww1kCLXryxPpopgzcbLWSuUWxPdMDkhIA81c5tY0lrkmDK3CVFyMmVuE0taLR21DTiUTW1Dk4JXKqLy+UIa9c+Gok9lZI6pOCSEEGJqQ5O6l6lhuQaPNs+qSYRWHcXfoZy7TJvNFWZOCZFGT8zRZHMFjQQKiUa6qHNCoHnLdN1gakawTJo2R88NpuYvC8IWd9P1lKlujv5b3M1fGrRNNqfXKi1T3exgbbL55NLgb/NbXz6l6mcHfZvfMAhUArECL0pegZ56PrwCgeDJVBCsBwIEAr6xEIAEAopoIIGAHcoA6MKALgwUT6BHH49xFsDV5gqUQUANBGVMIE4DIBCUBdnm9EohxIKG/ZwLsMvEmlhUwR1DgCIaQtaFCSGepRcDO0yoibXcZJO7QeAmgYQQC+cQQmCJh6bEcmzzSykNrhJICLFoNiEERRhfG8u70TghBK6G8eNqmZyHQmQZElV8G2LQrAbKsGQWlRDkYOzUmKU70a1/DyCnFVEt9tMH/ebCxhBCYMGH3DXQSZbOpBgCMfqJmMPZ+AL/E7CneAJlWEYOhZUHiiWIJYGEEMtxKHzcb6H/sSqQEGL5DBwKkz3TYs5vJOZjBQ6Fg1HTYr4sabX+uhAGe2wnUIYXyKHgcp/NjHAi0AmNpqNRsNSpi3k2mWqRF3EoKNxbF/NyNt4WL6GRztxTF/N+OQcaoU6pBcrwcj0aacDd9TF/F5ShEeqUU6CTrMQkNbir3q8beP8COS5WNiCwR5wAAAAASUVORK5CYII="),
    "/icon-512.png": _b64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAAB7GkOtAAATz0lEQVR42u3dabRdZXkH8Gfvc5LIEJYpicHclCIhBAhWWgqRFoMCam0pZdDAov3QWjsICiIgo4rKIA5lEMWhxQ4fVAiTlFJ0CbRVo1RwykBknkpCAZlJcu+5px9wxQAJuefcc+7Z735/v08Oa92cu+8+z38/77PfvYspW00POvE733zQQYBq+vHBsx2EsSsEwCYL/TUKPdQlGP5UMAgAFR+QBwJA0QcyD4McA0DRB4RBXgHwu+o+MDa35ZEE9Q8AdR+QBHkFgLoPSILsAkDpB8RAXgGg7gOSILsA2FPpBwbh1sRjIO0A2PNqpR8YdAwcMlsAKP2AGBAASj8gBgRAD/2e0g+k4EeJxECp+gPkWa8S6ACUfkArkF0AKP2AGIgMl4BUfyCsCOXWAeyl9AN19D8VawVK1R8gz/pWrQ5gr6tUf6DufcChswWA0g+Igch9CUj1B3JThbo3+A5gb9UfyNUtA+0DBhkASj/AAGOgVP0BIsuFkFL1B8gzAwawBLRA9QfYmB9O7FpQqfoDVMQEV8hS9QfIMwMmbglowZWqP8CY/PCw2fXpAFR/gKrVzInoAN6o+gN07gd97gNK1R+gmvpdP0vVHyDPDChVf4A8M6BfM4B9VH+A3lnSh3lAqfoDVF8/6mqp+gPkmQGl6g+QZwb0cgawzxWqP0DfLTm8N/OAZi/DxJ8FIB09WwL6fZf/ABOiV/W2VP0B8syAUvUHyDMDSgcRIE/jvQvoD1z+AwzO98ZxR1Cp+gOkazx1uFT9AfLMADMAgDAD6MC+Lv8BquS7nQ8DugmAfRer/gDVy4B3zrYEBEAfAsDlP0A1dVqfdQAAYQi8eW9y+Q9Qbf895klAqfoD1MnYa7UlIIAwBHb5D5BRE9D0ti8AHcAmLXT5D5CUsdTtsXUAbQcTIL8OYOHlLv8BEmwCNle93QUEEDaCbcR+Lv8BUvaf75qtAwBgzAHg8h8gda9QyXUAADoAAMIQOOLN1n8A6uLmjY2CdQAAOoANL/8vc/kPUK8mYNFsHQAAEZt6FpBnfwJEhncBvcX6D0DtvLy2WwIC0AEAkHMAWP8BiDxWgZrGvwA6AAByDYD9rf8A1NqGdb7p3b8AOgAAItOdwCbAADl2AAd8wwAAoP7WV3tLQABmAABEljMAAwCAHDuAAw0AALLxQs23BARgBgCAAAAgMhkCFx4CAZCZYspW09/6dRNggLAEBIAAACBqPQOwBQxABwCAAABAAAAgAACoieLga9c4CgCR411A9gADhCUgAAQAAAIAgBpq2gYMoAMAQAAAIAAACDMAAHQAANgJDIAOAAABAEAYAgNQyXcCMxCfXjh5zqt7k7/Do3H0d9Y99nwf5zmf23/y0Nbdf9on1rbffcO6KhyxY76z7uFn+zv4mlTGNw6aMs4fktURIywBMZ6Kc8TODccBEAA52n/7xqytrecBYQaQX5IXcdQuzc/+aLhPP7+owE+ozu8yMT8/nyOGDoDx2mdW2auhAiAASEkR8We7muoDYSdwjt4wo9x923Lpo6NV/HDtynyMdiK/aT5HDB0APfHnu7kdCDAEjqqv2PTD3GnlgteWtzw8Wr/ft+jRx8hnCJzKEUMHQM8ctWuz8JUFBECGZk8t9pttIQgQAFk6YpdG098WEAAZmrFl8fYdNAFAGALn6PCdGzfe31ozEhWZAFfkTEtlCJzVEUMHQI9tM6U4aI4mABAAWTp4p+bUya7eAAGQny2acZjHRAOxqRmATd619vYdGtfd2erru2LGqCJnWtHu+ycpHDF0AFTBpEYs2kUTAAiALO23fWNoqkkAIACyfFfMkR4TDQiAPC3wrhggbATL1VHzm2d9bziyf61gVk8DtREMHQAREa+fUe4+w58bEAB5NgG7mQQAAiBLc6YVe8/yFwfCDCBHR+7avPXhdaNtMwAzADMAvBQ+M0NTi4W/2bj5vlZ4KbyXwnspfFgCIk3PDccvHu/mxb/v3KUxyZ8dEABJ+9qybi7kp29ZvHVHD4cABEDKVjw6+tPV3TQBh85rbOGGIAhDYMcgUt0YVUR8bdnIb8/s+JH/UycXB81tLl4xktzvawic87vJ0AHwIvc92V7yYDdNwB/t1PCuGBAApO3y5SOtdjfvijl0nkkARN63gZK4Vc+2b763dcDrOq7mB+7Y+Pe7Wo8+l9ctfue/bbJzBnQA9bH49ta6zm8ImlTG4d4VA4bARMpj5CfXtG+4u/Unczuu5gu3b1x3R+uhp9upDCTdJpD5J8FOYOLlO0W/efvIATs0tpzU8btiFu3WPP8HwxOxK9WZlu4R87cLS0BU2LPDce0vurmtc69Z5ZxprvBAAJCy/7iz9cSabi7VjtzdvQAQZgBEumvE61px9e2tv9ij42o+f0b5+teUSx8ZtY5sBmAjmA6AVN14b2v1s900AUfM1wSAACBlrdFYvLybScCO04q9h5wMIABI2ZIHRu9/spsmYNFuzVKrDwKASPnWwcuWddMEvHZqsfC37AuDMAQm0h0S/mTV6MrHRudt23G6H7Zr4/sPtIZbhsCGwIbAOgCSddnSbqr4tlsUB3pXDNgJTFR/d+Wmf8LKR0d/smp0j+06DviDd27cfE/r+eHEfl9nSB+PmHcC6wBIsAkYaXf+vZ06pfjjuW4JBQFAyu5/sv39B7rZ2/WHcxvbTLHqC2EITKQ7JLxy+cgbZ09udJjyr2rGIbs0/vWnI4bAhsCGwF4IQ6oeebZ90z2tA+d0PNfdf8fG9XfU810xJ96wbvUz/f29JjXi0kOmOP0IS0AM1tVdvSumWcZhu7kdCAQAKXtyTfuGO7u5JXTf7RtD22j9IcwAiAqu8I7xJ1y3cmT/1zW26vBVuEUR75rfvHDJcHK/78BXtAtHDB0AFfHccPzbym4munvOKuf8hjMEBAAp+/ZdrV929a6YRbubBIAAIGXrWnH1im4mAbvOKHef6SSB8CgIIt1HBfzX3a13zG1st3XHy7mL5jeXrVpXn8cJTMAnadfrAQweBWEITFRvhtzR3260HVcuGzl6waRO/6EdphV7zy5ta0pxCGwjGGEJiBfc8sDofU90cy13+PxmQw0AAUCkvGJ0+dJubgfabmrxmq0lAAgAUvbzVaO3/9+o4wCEGUDkt5Vs8dLWGW8ps11HTmUGkNURQwfABLnzsdEf/68mABAAWVrc1btiAAFA8h56qr3k/pbjAAKAHF25vDViHQgi8yGwpYBkjedv99gz7Zvvbh24UyOJTxs9vRm235+kVz8/nyOGDoCJdu3y1toRhwEEAPl5am37W3eYBIAAIEvXrxx5Zp32HsJL4cnO88Nx3YrWEW9wGrBJ571j8kD+3X+5beSmu3SodgITfdxxeuNdrbft3Ji2RRGJPCPT00Cd4YQlIHpiuBXXLHOdBQKALH333taqp00CQACQn9F2XLXUDaEQZgBEKuubPfzb3frg6L2/bO8wraj975vVDKDI6Z13eCdweCdw1z/pip+NnLDfpCQ+bRbvBI7s38TrXcRhCYiJsnz16IpHPB4IBABZuuJnJgEgAMjSPY+3b3tIEwBhCExkuE3mqp+P7DFrclnYCGYjmCGwDoDMPPxUe8l99oWBACBL1yz1rhgQAGTp8efaN92pCYAwAyAyfFTW9Stab9qx8aqmGYAZgIfB6QDIzNNr299eqQkAO4GJ+u8Efrlv3T7y5p3KqVMKO4HtBLYTuLY91nu+vsZRAAhLQACEITAAOgAABAAAAgAAAQBAGAIDoAMAQAAAEB4FAYAZAABhCQgAAQCAAABAAAAQhsAA6AAAEAAACAAAwk5gAAyBAYgIS0AAAgAAAQBAmAEAoAMAQAAAIAAAEAAAREpDYDuBAXQAAAgAAAQAAGEjGAA6AAAEAAACAAABAEAYAgPgncAAhCUgAAQAAAIAgDAEBkAHAIAAAEAAABBmAADoAADY7E7g0AMAeBQEAGEJCABDYAB0AAAIAAAEAABhBgCADgAAAQCAAAAg7AQGwBAYgLAEBIAAAEAAACAAAAhDYAB0AAAIAAAEAACxsRmAncAk7pLj5w3qn37v+Ssdf9JVnPjPax0FlHVRgQAAJV48IABAoRcMCABQ7kUCNQyAkwQA4/AF5X7QjhYJCACyrfjTZ84c1D/96OrV8gABgIpfk5pes7SQBwgAKlf0a1niKx4PwoCXBsCHBAAREfH5/hR9hb6awXCMMCCi+NA/CYCMi/4H56n1UuGYvxcGAgB1X7nPOxIkgQBA3Vfxc88DSSAAyL3uq/jyQBLUPABOFgA1dXFXdV/RFwYb9T5JIACoZd1X9IWBJBAA5FX61X26TgIxIABIr/Qr+vQwDMSAACCB0q/u078kEAMJB8ApAiBNnxtD6Vf3mbAkeL8YSDIAvioAUiv9Jyj9VDUGPisGBAADKv3qPlVIAjEgAFD6EQNUOwBOFQCVd9ErVn+ln8rGwLEyoNpKhyDd6j995kzVnxj069he4SS86ARvDNUB0IfS7/iQUDegFdABoPpT825AKyAA6Ff1t+ZDuitCMqCCitMsAVXMhZuu/g4Okfhy0HHWgnQAdFT9XfhTm1bgQn2AAKCj6u/IUKepgAyICi0BXWoJqBrV/0TVn8hnOei4z1gL0gFghxfOZAbVAZyuA6iAC152+e87Q+37gA9oAnQAXHCiJVGc+QgAXP7jrEYAANBXxRlmAAN1vtV/IutJwPEmAToAAAQAAAIAgL5pRttBAAZHCRpgABSOATA4SlBYAgJAAIT75BwEnNsIAADCDCCvCyXbwcjk8l8J0gEAIACwWorzGQHgO+Mg4ExGAPjmgHOY6PUQ2DY8A2EYXPVXggb6KAgS+RaJAVz4E5aAfKPAuYoOQCsASj86AN8xcGZiJ7BWAFIp/UqQJSDEAK76EQCIAZR+BABiAKUfMwA6+2ZKAlKp+0qQdwKjISDXS34lyBIQGgIs9SAAkASo+wgA+vndFgYo+obAaAskgbo/MEqQDgBtAS72EQAIAxR9BACVqhryQMXHDAA1RR6o+GEGoANAxREJyj0CAFVpY1VJKqj1eBQE6pdgUOjDoyB0ALCZelfLeHAtT9UUZ315raMwWOeeMs9BGI8BpoWaPk6nfnKlg6ADAFUYwkvhARAAumBw5iMAfBPAOU/YCQykTuXRAfBrp7kgwtmOAPCtAOc5E9GHnf0l+wCq5ZxTbQugvtX/XNW/SgFwjgConrNlAHV0uuofloDwPcFZjQDAtwXnM2EJiLAchNKPDgDfH5y99L0DOFcHkIKz9AGk5gzVXwAgBlD6EQCIAZR+BABiAKWfwQfAFwVAyjFwmhhg0KX/HKU/2QD4pABI3yfEAIPwYaVfACAGUPoRAIgBlH4EAJIAdZ+KB8B5AqDuPi4J6NZH1H0BgCRA3UcAIAlQ9xEASALUfQQAwgBFn/QC4FOXCABe6mOnC4Ma+ujZij4CAGGg6IMAQB6o+AgAkAcqPpkFwKcFAD11pkjo6+FV7hEAiATlHgQAgkGhBwGAeFDiQQAgKpR12EwAfEYAAGSpdAgA8tSMtoMAoAMAQAAAUG/NwjEA0AEAIAAAEAAAhBkAADoAAAQAAGEnMAA6AAAMgQHQAQAgAAAQAACEGQAAOgAABAAAAgCAsBMYgO4UU7aa/qnzHnIgAMISEAACAAABAIAAACDqcBdQhM3AADoAAAQAALUPgJNOHnIgALJy0slDzRf+U2E/MIAlIAAEAAACAIBaBsCJp5gDA+TihZrfXP/fbQcDsAQEgAAAoPYBcIIxAEAG1lf75ob/qzEAgCUgAKL+j4P+NQ+EAMizA/jgqcYAAHW2YZ23BAQQloAizIEBItch8PFWgQBq6iUV3hIQgA4AgMwDwCoQQNR9/SdePgQOo2AAS0AA1FgxZavpG/0/LjjnIUcHoB4+cNqQDgCAzXUAEXGhJgAgfcedNmQGAIAAABAAXXQNAETi6z86AIAwBN6ki4yCAdJ07Csu5OgAAMIMoJsAASDFy//Y1LOAXsq7ggHyvA302NM1AQBJXf6PoW43x/izPB8UINONYO/XBAAkYowV211AADoATQBANpf/MZaNYC9x8dn2hQFU1Ps6uVK3BAQQloB6Hy8AVPPyXwcAEB4G15mLzzIJAKjS5f8ZQxMUABHxeRkAUA3HnDHkjWAA9D8AugscAKpw+T/eDkAGACRa/WM8M4D1vmAYADAIR4/vKtwMACDMAAYTQQAMpPaWFfkcAExw1S0r9WkAmLB62+zhZ/LWMICo/aMgNuWST7gjCKCP3vvhoQoNgfv0yQDoa40tK/75AOhTdS2T+JQAqn9UfAawoS+aBwD0wt/156q6TO4TA6j+VQ8AGQBQ5SpaJv3pAVT/qOAMYENfMg8A6MTf9v/quazNbwKg+lexA/hVH/BxfQDA5qr/R4aSeRx0BX8rANW/ci+EkQEAFamQE7oEtN6XrQUBvNjfTPj1cZnJ7wmg+lflncAyAGCw9XAwS0BhOQhg0JfCgw+AiPiKDADy89eDXggpHQWAPOteJToArQCg9OceABHxlY/JAKC+1f+jFVrwKB0dgDzrW+U6gPX+QSsA1MV7KnlpWzpeAHlWs+p2AFoBQOnPPQDEAKD0R1ZLQFaEANVfB/Ai/6gVAKrqr5K6VE0vAMQAoPRnHQBiAFD6sw6AX8XAmWIAGETpPzPt2WQdAuAFl4oBYKK8+8w63JZSnwCQBIC6n3sAiAFA6c86ACQBoO7nHgCSAFD3cw+A9b4qCYCN+csz83riQI4BIAyAbIu+ABAGoOgjAOQBqPi5+n+xJs/ExHiFFQAAAABJRU5ErkJggg=="),
    "/apple-touch-icon.png": _b64.b64decode("iVBORw0KGgoAAAANSUhEUgAAALQAAAC0CAIAAACyr5FlAAAIFUlEQVR42u2dW2wUVRjHv7OzsCCgiEWQ3YY0oMvNKCAgqAHUBzGGm6Co8UFE5VaD1Au00BboTVTulxZU1Acjckf0wWsMoAJajBBgI0hQ0DYBTDBByu7s+NBa9jI7ndmZ7Z4z5//LPHU7O7Ozv/m+b745Z5b5OuRQhhm4+yyBDHB4bCCj788yJMfAXRCidUUZF+BdjkFwItvUOmeJM3LACVdaYleOwTuhBdf8ND6QHTnughki8GO6fqQpB7SQQZF05BgCM8TkkEU/rMkxZAe0EF+RCQHn5RgKM9zCQXN+mJVjGMxwFwdM+GFKjmHbYYYb/ZgYsCvH3TDDvfxg6EcLcgyHGW7n+9R+eGCG5Bh8yykjx4htMEMivns0YC1yAMnRjxz3IGzIx/6k4KEjx70wQ1b2xfvh1fkXDUcJ6EWO+7YibEjN3kkB/YIUZoBYB+LSCkNCAbppZeQWhA3QxLeTA+hzACOupRWGgwF008roj5FTQBzfPBZAWgEtpRXkFKCfVu7fjJwCUkQOhA0AOYA1UJCC1DXHI59cwVEAiBzAYs2BggNADpDG1Qpu0wNEDgA5ANIKkCFytPPShw/7Ur1atC98/GLUYPX8gd7RuUry35/8rOFKxMK2Ht/TEI6muZ+Ob4vQ5zDD0/0UnMHyRg7jHejTxTO0u+dQXdTq6kzvJWa4Gyzd/XR8W9zIke2ao8UdeKqvt7bualSz/LbJ72ywLd3/t7OunW0hrZgltxMbmYvkgrSSgil9lP1nVUtFHNKKKy5lTezATe3ZmDxl90nVwuqa3kuajT3RDF9ydluIHJaGr064zfvVGfVyGAVpK9Yc7P99zeJiho5tafytXvPrWt2Wnf3MxLZ4WPgtSNWkwDuml3JjOyFOOdcUpLxeyh76M9q/K+vU9poNPoUm91E2Ho7gUlb2tPJvRNsVSqxAR/VUenRkZlZHWrG/eHk29/Pf1Id6KTnXXTvOCqMp/b3LD4Qd39YHY33II8KkFaZRRKWtx9Xpg+MMHtrD07szO/W3ZnxNaDXU2/wI6JBmgb2/q39cSjyQTwzw4rSW/VKWEZFGm48mVqD9unru6OZxtuaw2apx56Us53Iwop/roqELiZ3zKQO8HkeLRMih1+fQKPuLcVtao4+OJF629LyBDQ94zKxrdltk7w5AWp+L84Xf9nnsyXfyQrT2r+igW+IqpEn9vQfPXXWqpT11Z0NYJeORYBvH+SQbz8HxvZXYl7YcjdzZva0n5k83d2AP5Cm4t4LpkHTukrbvTOKpPa6v4sP4eRlHgiU1A3YcU4fnKm1ixv1c72ODejC0zzGRmi5e1r48lRg8FAQOCWsO3cS850RkVJ7Svg1GgsmRVkizEMkvN9CnJyKTbveaeVumWdsWszESzOFt8faQWlH44lf1wd5K5/YCp5MNE83e5Cv7OnzqQhTtc7NtxLBKu46pWelatn43FiPBLLP3tFr3D+b4yj2pKdVLmkbbj0Rmjmgj6EgwSwVZFr8gYTqkCdSejZ6+qOV1YSJerdicfYM+R8ts+yVCIKPn7bTNeNQk4LbPAQjtc4D2OcAzwQAiB4AcAKAgBbiUBYgcADUHQFoBSCsAaQVADoD2OUDkAAAFKTAXORA7APocAGkFoCAFuJQFiBwAcgCx0opMVM8J2n+T6StCkhwu9vL7DS77SOtfCmZr0zOWu8ob9or4cqwzbUNOt272N3e+vt7kf84U3BVR5TAQwhEDHPdGRFHYq+8JI8fauUHebEjPlVnLQpAjg1pwK4R5UfhXhL3GtxxrYrQQTggzoszmWBE2j1c5VrtRi1SK5HOpCJu3iTs5VhcEXeyEgSX5b4XwS01Gi2xmxH7S1QVB/OhwSlYVBBsPljxmNPvR+JFXFQQROXSWZjNIVpr94OY33jTiYVnJ0xmTdVYWBHn4Ujx8Zl/Jgwfhlj1I75m4GGAsvRwaBhine8PT9T0PpBXAdVrx4NTBZxep5mg8RlJduSRrwcP3wm9aOV9fL9u9Fe6uVoQ4aq6/K0uYK4tEY14LpJX0j6kLRoIJ0ARzwckn3BhSwoy3bH0HfI4+JxHb5+6bSG3wDbXyvBXKzA91txqsbAMvwwQr5+GufRPzq0IY7BO3FFaFoAURFVaFMExQ/7jADOJomCAfI8Gal6JKef0oqgzx9XVU1HA6b6V8flAqLTjcK1ZZw/WMtzK3K7KA40jJqqoFmCu7pNCFiiys4H6urBByuEwR/rVokuP1aiGfz7FYNFGKKwR8PsfSauGf7LOIV1FKKgR/ss/S9W57Jtiioqy5UlLurmeCveE6OQwodcKb0nJpniYolRyAMKkJYDwHIPykBkDkAPhJDYDIASAHQEEK0OcAAGkFIK0AOzBfh5w3q87hQABEDmBxIjXKDgA5gPWag4iWVaLsAHHMne/38jPhH6AgBaKlFSJaUYHMApqYU+iPe+wTMgtAWgHW0woRrUJmAUQvFvop+WmCuEMLUqaV/CI/jojkxDoQl1YaWVOO5CIps+OjAwpSYK4gbWZtGYKHdMxa4DclBxGtgx8yMXOBH30OYDutNLJ+CYKHFMxY6LfcIU21DpDBjBYiRyPViB/uZbrh+d+yHERUAz/cyAstZQZTchBRzWL44S4ziv22CtIENsAPt/B8sd/u1UoyG+GH+DxX7HfgUjalIougiJhalPgd63MY8Db8EI1pJX4nm2BQRFotHJCDiN6BH3zzbIk/I+1z87xbCkW4Y2qpP4P3VmCJnE5kSo5mNsGS1uWZUudvhP0H3teicfdfPBYAAAAASUVORK5CYII="),
}

MANIFEST = json.dumps({
    "name": "NHL Predikce", "short_name": "NHL Predikce",
    "description": "Předpovědi NHL – body, střely, vyloučení, kurzy, live hřiště",
    "start_url": "/", "display": "standalone", "background_color": "#070b14", "theme_color": "#070b14",
    "icons": [{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
              {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"}],
}, ensure_ascii=False)

SW_JS = """self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => self.clients.claim());
self.addEventListener('fetch', e => {});"""


def keep_alive():
    """Na Renderu zdarma server po 15 min nečinnosti usne – každých 10 min se sám „šťouchne“ přes veřejnou adresu."""
    url = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("KEEPALIVE_URL")
    if not url:
        return

    def run():
        while True:
            time.sleep(600)
            try:
                http_get(url.rstrip("/") + "/ping", timeout=30)
            except Exception as e:
                print("keep-alive:", e, flush=True)
    threading.Thread(target=run, daemon=True).start()


def main():
    LEAGUE.ensure()  # začni stahovat data hned po startu
    keep_alive()
    if os.environ.get("PORT"):
        port = int(os.environ["PORT"])
        server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        print(f"NHL statistiky běží na portu {port}", flush=True)
        server.serve_forever()
        return
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else PORT
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        print(f"Port {port} je obsazený. Zkus: python nhl_statistiky.py 8800")
        sys.exit(1)
    url = f"http://localhost:{port}"
    print(f"NHL statistiky běží na {url}  (Ctrl+C = konec)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nKonec.")


PAGE = r"""<!DOCTYPE html>
<html lang="cs">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NHL Predikce – body, střely, vyloučení, kurzy</title>
<link rel="manifest" href="/manifest.webmanifest">
<meta name="theme-color" content="#120822">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<link rel="icon" href="/icon-192.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="NHL Predikce">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Anton&family=Caveat:wght@600&family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #070b14; --bg2: #0c1322; --card: rgba(18, 27, 46, .78); --card2: #152038; --line: rgba(148, 163, 184, .14);
  --line2: rgba(148, 163, 184, .26); --text: #e8eef8; --muted: #8796ae; --cyan: #38bdf8; --blue: #3b82f6;
  --violet: #a78bfa; --gold: #fbbf24; --green: #34d399; --red: #f87171; --orange: #fb923c;
  --home: #38bdf8; --away: #a78bfa; --radius: 16px;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: var(--text);
  background: radial-gradient(1200px 600px at 10% -10%, rgba(56,189,248,.16), transparent 60%),
              radial-gradient(900px 500px at 100% 0%, rgba(167,139,250,.14), transparent 60%), var(--bg);
  background-attachment: fixed; min-height: 100vh; -webkit-font-smoothing: antialiased; }
a { color: var(--cyan); }
button, input, select { font-family: inherit; }
body { zoom: 1.15; }
.help { font-size: 13.5px; line-height: 1.6; color: #9fb0c8; margin: 8px 0 0; font-weight: 400; }
.help b { color: #d6e2f3; }
.helpbox { background: rgba(56,189,248,.07); border: 1px solid rgba(56,189,248,.25); border-radius: 14px; padding: 14px 18px; margin: 16px 0; }
.helpbox h3 { margin: 0 0 6px; font-size: 16px; }
.helpbox ul { margin: 6px 0 0; padding-left: 20px; }
.helpbox li { margin: 4px 0; }
details.gloss { margin-top: 30px; }
details.gloss summary { cursor: pointer; font-weight: 800; font-size: 18px; padding: 6px 0; }
details.gloss dl { display: grid; grid-template-columns: max-content 1fr; gap: 8px 18px; font-size: 14.5px; margin: 12px 0 0; }
details.gloss dt { font-weight: 800; color: var(--gold); }
details.gloss dd { margin: 0; color: #c3cee0; line-height: 1.55; }
.wrap { max-width: 1280px; margin: 0 auto; padding: 0 20px 80px; }

/* cinematic */
.display { font-family: Anton, Impact, "Arial Narrow", sans-serif; font-weight: 400; text-transform: uppercase; letter-spacing: .01em; line-height: .92; }
.cine { position: relative; margin: 0 -20px; padding: 70px 40px 46px; overflow: hidden; border-bottom: 1px solid var(--line);
  background: radial-gradient(900px 420px at 80% 10%, rgba(56,189,248,.22), transparent 60%),
              radial-gradient(700px 380px at 0% 100%, rgba(167,139,250,.2), transparent 60%), linear-gradient(180deg, #0a1324, #070b14); }
.cine .rink { position: absolute; inset: 0; width: 100%; height: 100%; opacity: .16; pointer-events: none; }
.cine .kicker { color: var(--cyan); font-weight: 800; letter-spacing: .22em; font-size: 13px; text-transform: uppercase; position: relative; }
.cine h1 { position: relative; font-size: clamp(56px, 9vw, 136px); margin: 14px 0 0; line-height: 1.12; padding-top: .08em;
  background: linear-gradient(180deg, #ffffff 30%, #9fc3ea); -webkit-background-clip: text; background-clip: text; color: transparent; }
.cine h1 > span:not(.ln) { display: block; color: transparent; -webkit-text-stroke: 1.5px rgba(255,255,255,.55); background: none; }
.hero2 h1 .ln:nth-child(2) > i { color: transparent; -webkit-text-stroke: 1.5px rgba(255,255,255,.6); }
.hero2 h1 { background: none; color: #fff; text-shadow: 0 3px 14px rgba(0,0,0,.35); }
.cine .lead { position: relative; max-width: 720px; font-size: 18px; color: #b9c7dc; line-height: 1.55; margin: 18px 0 0; }
.counters { position: relative; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0; margin-top: 40px;
  border-top: 1px solid var(--line2); }
.counters div { padding: 18px 18px 0 0; }
.counters div + div { padding-left: 18px; border-left: 1px solid var(--line); }
.counters b { display: block; font-family: Anton, Impact, sans-serif; font-weight: 400; font-size: clamp(40px, 5vw, 68px); line-height: 1; }
.counters small { color: var(--muted); font-size: 12px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
.cine .daytabs { position: relative; margin-top: 28px; }
.marquee { margin: 0 -20px; overflow: hidden; border-bottom: 1px solid var(--line); background: rgba(251,191,36,.06); }
.marquee .track { display: inline-flex; gap: 46px; padding: 14px 0; white-space: nowrap; animation: mq 60s linear infinite; }
.marquee span { font-family: Anton, Impact, sans-serif; font-size: 22px; text-transform: uppercase; letter-spacing: .03em; color: #e9eef7; }
.marquee span em { font-style: normal; color: var(--gold); margin-left: 8px; }
@keyframes mq { from { transform: translateX(0); } to { transform: translateX(-50%); } }
.sechead { display: flex; align-items: flex-end; gap: 18px; margin: 64px 0 16px; flex-wrap: wrap; }
.sechead .no { font-family: Anton, Impact, sans-serif; font-size: 22px; color: var(--cyan); }
.sechead h2 { margin: 0; font-family: Anton, Impact, sans-serif; font-weight: 400; text-transform: uppercase; font-size: clamp(36px, 5vw, 64px); line-height: 1.1; letter-spacing: .01em; }
.sechead .hint { color: var(--muted); font-size: 14px; padding-bottom: 8px; }
/* hráčské karty */
.pcards { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 14px; }
.pcard { position: relative; overflow: hidden; border-radius: 20px; padding: 18px; cursor: pointer; min-height: 300px;
  display: flex; flex-direction: column; border: 1px solid var(--line2); transition: transform .2s, box-shadow .2s;
  background: linear-gradient(160deg, color-mix(in srgb, var(--tc) 55%, #0b1426) 0%, #0b1426 62%); }
.pcard:hover { transform: translateY(-4px); box-shadow: 0 18px 40px rgba(0,0,0,.45); }
.pcard .jn { position: absolute; right: -6px; top: -18px; font-family: Anton, Impact, sans-serif; font-size: 150px; line-height: 1;
  color: rgba(255,255,255,.1); -webkit-text-stroke: 1px rgba(255,255,255,.22); }
.pcard .wm { position: absolute; left: -30px; bottom: -30px; width: 170px; height: 170px; opacity: .1; }
.pcard .top { display: flex; align-items: center; gap: 8px; position: relative; }
.pcard .top img { width: 34px; height: 34px; }
.pcard .top small { color: #cdd8ea; font-weight: 700; font-size: 12px; letter-spacing: .06em; }
.pcard .pname { position: relative; font-family: Anton, Impact, sans-serif; text-transform: uppercase; font-size: 30px; line-height: 1.1; margin: 16px 0 6px; }
.pcard .pname span { display: block; font-size: 17px; color: #d5deeb; letter-spacing: .04em; }
.pcard .big { position: relative; margin-top: auto; font-family: Anton, Impact, sans-serif; font-size: 64px; line-height: 1; color: var(--gold); }
.pcard .big small { font-family: Inter, sans-serif; font-size: 12px; font-weight: 700; color: #cdd8ea; letter-spacing: .08em; display: block; text-transform: uppercase; }
.pcard .pst { position: relative; display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,.12); }
.pcard .pst b { display: block; font-size: 17px; font-weight: 800; }
.pcard .pst small { color: #aebbd0; font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; }
.pcard .streak { position: absolute; right: 14px; top: 14px; }
.reveal { opacity: 0; transform: translateY(26px); transition: opacity .7s ease, transform .7s ease; }
.reveal.in { opacity: 1; transform: none; }
.mcard .wmL, .mcard .wmR { position: absolute; width: 190px; height: 190px; opacity: .07; top: 20px; pointer-events: none; }
.mcard .wmL { left: -60px; } .mcard .wmR { right: -60px; }
/* interaktivita */
#progress { position: fixed; left: 0; top: 0; height: 3px; width: 0; z-index: 100; background: linear-gradient(90deg, var(--cyan), var(--violet), var(--gold)); }
.cine .spot { position: absolute; inset: 0; pointer-events: none; opacity: .9;
  background: radial-gradient(420px circle at var(--mx, 70%) var(--my, 30%), rgba(56,189,248,.22), transparent 60%); transition: background .08s; }
.cine h1, .cine .kicker, .cine .lead { will-change: transform; }
.tilt { transform-style: preserve-3d; transition: transform .18s ease-out, box-shadow .2s; will-change: transform; }
.tilt .glare { position: absolute; inset: 0; pointer-events: none; border-radius: inherit; opacity: 0; transition: opacity .2s;
  background: radial-gradient(300px circle at var(--gx, 50%) var(--gy, 50%), rgba(255,255,255,.16), transparent 60%); }
.tilt:hover .glare { opacity: 1; }
.pcard.reveal.in .big, .mcard.reveal.in .xg { animation: pop .7s cubic-bezier(.2,1.4,.4,1) both; }
@keyframes pop { from { transform: scale(.6); opacity: 0; } to { transform: none; opacity: 1; } }
.wbar i { transition: width 1.1s cubic-bezier(.2,.8,.2,1); }
.cmp .bar i { transition: width 1.1s cubic-bezier(.2,.8,.2,1); }
.countdown { font-variant-numeric: tabular-nums; color: var(--cyan); font-weight: 800; }
.slider { margin-top: 6px; }
.slider .srow { display: grid; grid-template-columns: 1fr auto; gap: 18px; align-items: center; }
.slider input[type=range] { width: 100%; accent-color: var(--gold); height: 28px; cursor: pointer; }
.slider .lineval { font-family: Anton, Impact, sans-serif; font-size: 54px; line-height: 1; color: #fff; min-width: 120px; text-align: right; }
.slider .split { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px; }
.slider .side { border-radius: 16px; padding: 16px; border: 1px solid var(--line2); background: rgba(255,255,255,.04); position: relative; overflow: hidden; }
.slider .side .fill { position: absolute; left: 0; bottom: 0; top: 0; opacity: .18; transition: width .35s; }
.slider .side.over .fill { background: var(--green); } .slider .side.under .fill { background: var(--red); }
.slider .side > *:not(.fill) { position: relative; }
.slider .side small { color: var(--muted); font-weight: 700; text-transform: uppercase; letter-spacing: .08em; font-size: 12px; }
.slider .side b { display: block; font-family: Anton, Impact, sans-serif; font-weight: 400; font-size: 48px; line-height: 1.05; }
.slider .side span { color: #c7d3e6; font-size: 14px; }
.slider .dist { margin-top: 14px; }
.slider .dist svg { width: 100%; height: auto; display: block; }
/* hřiště */
.rinkwrap { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(260px, 1fr); gap: 18px; align-items: start; }
.rinkwrap svg { width: 100%; height: auto; display: block; border-radius: 18px; }
.shot { animation: dropin .45s cubic-bezier(.2,1.6,.4,1) both; transform-box: fill-box; transform-origin: center; }
.shot.goal { animation: dropin .45s cubic-bezier(.2,1.6,.4,1) both, glow 1.6s ease-in-out infinite alternate; }
@keyframes dropin { from { transform: scale(0); opacity: 0; } to { transform: scale(1); opacity: 1; } }
@keyframes glow { from { filter: drop-shadow(0 0 0 #fbbf24); } to { filter: drop-shadow(0 0 6px #fbbf24); } }
.zones div { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px dashed var(--line); }
.zones b { font-family: Anton, Impact, sans-serif; font-weight: 400; font-size: 30px; }
/* simulace */
.simtop { display: grid; grid-template-columns: auto 1fr; gap: 24px; align-items: center; }
.scoreboard { background: #04070d; border: 2px solid #1f2a40; border-radius: 18px; padding: 16px 26px; text-align: center; min-width: 300px;
  box-shadow: inset 0 0 30px rgba(56,189,248,.08), 0 10px 30px rgba(0,0,0,.4); }
.scoreboard .digits { font-family: Anton, Impact, sans-serif; font-size: 84px; line-height: 1; color: #ffcf4d; text-shadow: 0 0 18px rgba(255,190,60,.55); letter-spacing: .04em; }
.scoreboard .tm { display: flex; justify-content: space-between; font-weight: 800; color: #9fb0c8; letter-spacing: .1em; font-size: 13px; }
.scoreboard small { color: #6f7f99; font-size: 12px; }
.simstats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.simstats div { background: rgba(255,255,255,.04); border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; }
.simstats b { display: block; font-family: Anton, Impact, sans-serif; font-weight: 400; font-size: 38px; line-height: 1.05; }
.simstats small { color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; }
.simgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; margin-top: 16px; }
.hbar { display: grid; grid-template-columns: 110px 1fr 54px; gap: 8px; align-items: center; margin: 5px 0; font-size: 13.5px; }
.hbar .b { height: 12px; background: rgba(255,255,255,.06); border-radius: 99px; overflow: hidden; }
.hbar .b i { display: block; height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--cyan), var(--violet)); transition: width .25s; }
.hbar .n { text-align: right; font-weight: 800; font-variant-numeric: tabular-nums; }
/* porovnání */
.cmpwrap { display: grid; grid-template-columns: minmax(260px, 420px) 1fr; gap: 22px; align-items: center; }
.cmpwrap svg { width: 100%; height: auto; }
.radar-poly { transform-origin: 200px 200px; animation: grow .9s cubic-bezier(.2,1.2,.4,1) both; }
@keyframes grow { from { transform: scale(0); opacity: 0; } to { transform: scale(1); opacity: 1; } }
.cmprow { margin: 10px 0; }
.cmprow .lab { font-size: 13px; color: var(--muted); font-weight: 700; margin-bottom: 4px; }
.cmprow .r { display: grid; grid-template-columns: 1fr 60px; gap: 8px; align-items: center; margin: 3px 0; }
.cmprow .b { height: 12px; background: rgba(255,255,255,.06); border-radius: 99px; overflow: hidden; }
.cmprow .b i { display: block; height: 100%; border-radius: 99px; width: 0; transition: width 1s cubic-bezier(.2,.8,.2,1); }
.cmprow .v { text-align: right; font-weight: 800; font-size: 13.5px; }
.cmpsel { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
.cmpsel select { border-width: 2px; }
/* tiket */
#slipbtn { position: fixed; left: 22px; bottom: 22px; z-index: 50; border: none; cursor: pointer; color: #1a1300; font-weight: 900;
  font-size: 15px; padding: 14px 20px; border-radius: 999px; background: linear-gradient(135deg, #fde68a, var(--gold)); box-shadow: 0 10px 30px rgba(251,191,36,.4); }
#slipbtn .cnt { background: #1a1300; color: var(--gold); border-radius: 99px; padding: 1px 8px; margin-left: 6px; }
#slipbtn.bump { animation: bump .5s; }
@keyframes bump { 30% { transform: scale(1.18) rotate(-4deg); } 60% { transform: scale(.95); } }
#slip { position: fixed; left: 22px; bottom: 22px; z-index: 60; width: 430px; max-width: calc(100vw - 24px); max-height: calc(100vh - 40px);
  display: flex; flex-direction: column; background: #0d1527; border: 1px solid rgba(251,191,36,.4); border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,.6); overflow: hidden; }
#slip header { display: flex; align-items: center; gap: 10px; padding: 14px 16px; background: linear-gradient(135deg, rgba(251,191,36,.25), rgba(251,146,60,.15)); border-bottom: 1px solid var(--line); }
#slip header b { flex: 1; font-family: Anton, Impact, sans-serif; font-weight: 400; font-size: 24px; letter-spacing: .03em; text-transform: uppercase; }
#slip header button { background: none; border: none; color: var(--muted); font-size: 20px; cursor: pointer; }
#slipitems { overflow-y: auto; padding: 10px 14px; flex: 1; }
.sitem { display: grid; grid-template-columns: 1fr auto auto; gap: 8px; align-items: center; padding: 10px 0; border-bottom: 1px dashed var(--line); }
.sitem small { display: block; color: var(--muted); font-size: 12px; }
.sitem input { width: 74px; padding: 6px 8px; text-align: right; }
.sitem button { background: none; border: none; color: var(--red); font-size: 18px; cursor: pointer; }
#slipsum { padding: 14px 16px; border-top: 1px solid var(--line); background: rgba(255,255,255,.03); }
#slipsum .row { display: flex; justify-content: space-between; align-items: center; margin: 5px 0; font-size: 14px; }
#slipsum .row b { font-size: 17px; }
#slipsum .big { font-family: Anton, Impact, sans-serif; font-weight: 400; font-size: 34px; color: var(--gold); }
.addbtn { background: rgba(251,191,36,.14); border: 1px solid rgba(251,191,36,.45); color: #ffe08a; border-radius: 9px; padding: 5px 10px;
  font-weight: 800; font-size: 12.5px; cursor: pointer; white-space: nowrap; }
.addbtn:hover { background: rgba(251,191,36,.28); }
.addbtn.done { background: rgba(52,211,153,.15); border-color: rgba(52,211,153,.5); color: var(--green); }
@media (max-width: 900px) { .rinkwrap, .cmpwrap, .simtop { grid-template-columns: 1fr; } .scoreboard { min-width: 0; } }
@media (max-width: 720px) { #slipbtn { left: 14px; bottom: 14px; } #slip { left: 6px; bottom: 6px; } .scoreboard .digits { font-size: 64px; } }
/* live */
.livedot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: var(--red); box-shadow: 0 0 0 0 rgba(248,113,113,.7); animation: ping 1.6s infinite; vertical-align: 1px; }
@keyframes ping { 0% { box-shadow: 0 0 0 0 rgba(248,113,113,.7); } 70% { box-shadow: 0 0 0 9px rgba(248,113,113,0); } 100% { box-shadow: 0 0 0 0 rgba(248,113,113,0); } }
.lgames { display: flex; gap: 10px; overflow-x: auto; padding: 4px 2px 12px; scrollbar-width: thin; }
.lg { flex: 0 0 auto; min-width: 190px; background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 10px 12px; cursor: pointer; }
.lg.on { border-color: var(--cyan); box-shadow: 0 0 0 2px rgba(56,189,248,.25); }
.lg .r { display: flex; align-items: center; gap: 8px; font-weight: 800; margin: 3px 0; }
.lg .r img { width: 24px; height: 24px; } .lg .r span { flex: 1; } .lg .r b { font-size: 18px; }
.lg small { color: var(--muted); font-size: 12px; font-weight: 600; }
.lscore { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 14px; padding: 22px; border-radius: 22px;
  background: linear-gradient(135deg, color-mix(in srgb, var(--hc) 28%, #0b1426), #070b14 50%, color-mix(in srgb, var(--ac) 28%, #0b1426)); border: 1px solid var(--line2); }
.lscore .t { display: flex; align-items: center; gap: 14px; } .lscore .t.r { flex-direction: row-reverse; text-align: right; }
.lscore .t img { width: 78px; height: 78px; }
.lscore .t b { font-family: Anton, Impact, sans-serif; font-weight: 400; font-size: 30px; text-transform: uppercase; display: block; line-height: 1.1; }
.lscore .t small { color: var(--muted); font-weight: 700; }
.lscore .mid { text-align: center; }
.lscore .sc { font-family: Anton, Impact, sans-serif; font-size: 96px; line-height: 1; color: #fff; letter-spacing: .03em; }
.lscore .sc.flash { animation: goalflash 1.6s ease; }
@keyframes goalflash { 0%, 100% { color: #fff; text-shadow: none; } 20%, 60% { color: var(--gold); text-shadow: 0 0 30px rgba(251,191,36,.9); transform: scale(1.08); } }
.lscore .clk { font-weight: 800; font-size: 18px; color: var(--cyan); font-variant-numeric: tabular-nums; }
.badge { display: inline-block; margin-top: 6px; padding: 4px 10px; border-radius: 99px; font-weight: 800; font-size: 12.5px; }
.badge.pp { background: rgba(251,191,36,.18); color: var(--gold); animation: ppblink 1.2s ease-in-out infinite alternate; }
@keyframes ppblink { from { opacity: .6; } to { opacity: 1; } }
.badge.en { background: rgba(248,113,113,.18); color: var(--red); }
.rinkstage { perspective: 1500px; margin-top: 16px; }
.rinkstage .tilt3d { transition: transform .9s cubic-bezier(.2,.8,.2,1); transform-origin: 50% 60%; }
.rinkstage.d3 .tilt3d { transform: rotateX(46deg) scale(.94); }
.rinkstage svg { width: 100%; height: auto; display: block; filter: drop-shadow(0 30px 40px rgba(0,0,0,.55)); }
#puck { transition: transform .55s cubic-bezier(.3,.8,.3,1); }
.ev { transform-box: fill-box; transform-origin: center; animation: dropin .45s cubic-bezier(.2,1.6,.4,1) both; }
.ev.goal { animation: dropin .45s cubic-bezier(.2,1.6,.4,1) both, glow 1.4s ease-in-out infinite alternate; }
.ripple { transform-box: fill-box; transform-origin: center; animation: ripple 1.2s ease-out forwards; }
@keyframes ripple { from { transform: scale(.3); opacity: .9; } to { transform: scale(3.2); opacity: 0; } }
.feed { max-height: 520px; overflow-y: auto; }
.fe { display: grid; grid-template-columns: 54px 26px 1fr; gap: 8px; padding: 8px 0; border-bottom: 1px dashed var(--line); font-size: 14px; align-items: start; }
.fe .tm { color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12.5px; font-weight: 700; }
.fe.goal { background: rgba(251,191,36,.08); border-radius: 8px; }
.fe small { color: var(--muted); display: block; }
.lfilters { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 12px; }
.lfilters label { display: inline-flex; gap: 5px; align-items: center; font-size: 13px; font-weight: 600; color: #c7d3e6;
  background: rgba(255,255,255,.04); border: 1px solid var(--line); padding: 6px 10px; border-radius: 99px; cursor: pointer; }
.pace { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.pace div { background: rgba(255,255,255,.04); border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; }
.pace b { font-family: Anton, Impact, sans-serif; font-weight: 400; font-size: 36px; display: block; line-height: 1.05; }
.pace small { color: var(--muted); font-size: 12.5px; }
.lgrid { display: grid; grid-template-columns: minmax(0, 2fr) minmax(280px, 1fr); gap: 16px; margin-top: 16px; }
.okc { color: var(--green); font-weight: 800; } .noc { color: var(--red); font-weight: 800; }
.calib svg { width: 100%; height: auto; }
.bigstat b { font-family: Anton, Impact, sans-serif; font-weight: 400; font-size: 56px; line-height: 1; display: block; }
.bigstat small { color: var(--muted); font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: .07em; }
@media (max-width: 900px) { .lgrid { grid-template-columns: 1fr; } .lscore .sc { font-size: 64px; } .lscore .t img { width: 48px; height: 48px; } .lscore .t b { font-size: 20px; } }
@media (max-width: 720px) { .lscore { padding: 14px 8px; gap: 6px; } .lscore .t { flex-direction: column !important; text-align: center !important; gap: 4px; } }
/* pohyblivý hero */
.hero2 { min-height: 86vh; display: flex; flex-direction: column; justify-content: flex-end; cursor: none; }
.hero2 .stage3d { position: absolute; inset: 0; perspective: 1200px; pointer-events: none; }
.hero2 #icecv { position: absolute; inset: -6% -4%; width: 108%; height: 112%; transform-origin: 50% 80%; will-change: transform; }
.hero2 #frost { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; mix-blend-mode: screen; }
.hero2 .backword { position: absolute; left: 50%; top: 44%; transform: translate(-50%, -50%); font-family: Anton, Impact, sans-serif;
  font-size: clamp(160px, 30vw, 460px); line-height: 1; color: transparent; -webkit-text-stroke: 2px rgba(255,255,255,.09); letter-spacing: .02em;
  pointer-events: none; white-space: nowrap; will-change: transform; user-select: none; }
.hero2 > *:not(.stage3d):not(#frost):not(.backword):not(.spot):not(.sky):not(.sun):not(.skyline) { position: relative; z-index: 3; }
.hero2 h1 .ln { display: block; overflow: hidden; padding-top: .08em; margin-top: -.08em; }
.hero2 h1 .ln > i { display: block; font-style: normal; transform: translateY(110%); animation: lineup 1s cubic-bezier(.2,.9,.2,1) forwards; }
.hero2 h1 .ln:nth-child(2) > i { animation-delay: .15s; }
@keyframes lineup { to { transform: none; } }
.hero2 .kicker, .hero2 .lead, .hero2 .counters, .hero2 .daytabs, .hero2 .hint2 { opacity: 0; animation: fadeup .9s ease forwards .45s; }
@keyframes fadeup { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: none; } }
.hero2 .hint2 { color: #9fb0c8; font-size: 13px; font-weight: 600; margin-top: 10px; letter-spacing: .02em; }
.hero2 .hint2 b { color: #fff; }
#cur { position: fixed; left: 0; top: 0; width: 34px; height: 34px; margin: -17px 0 0 -17px; border: 2px solid rgba(255,255,255,.85); border-radius: 50%;
  pointer-events: none; z-index: 200; mix-blend-mode: difference; transition: width .2s, height .2s, margin .2s, background .2s; display: none; }
#cur.big { width: 64px; height: 64px; margin: -32px 0 0 -32px; background: rgba(255,255,255,.9); }
#cur.on { display: block; }
.magnet { transition: transform .25s cubic-bezier(.2,.8,.2,1); }
@media (hover: none) { .hero2 { cursor: auto; } }
@media (prefers-reduced-motion: reduce) { .hero2 h1 .ln > i, .hero2 .kicker, .hero2 .lead, .hero2 .counters, .hero2 .daytabs { animation: none; opacity: 1; transform: none; } }
/* top bar */
.topbar { position: sticky; top: 0; z-index: 30; backdrop-filter: blur(14px); background: rgba(7,11,20,.72);
  border-bottom: 1px solid var(--line); }
.topbar .wrap { display: flex; align-items: center; gap: 18px; padding-top: 12px; padding-bottom: 12px; flex-wrap: wrap; }
.brand { display: flex; align-items: center; gap: 10px; font-weight: 900; font-size: 19px; letter-spacing: -.02em; text-decoration: none; color: var(--text); }
.brand .puck { width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center; font-size: 18px;
  background: linear-gradient(135deg, var(--cyan), var(--violet)); box-shadow: 0 6px 20px rgba(56,189,248,.35); }
.brand small { display: block; font-size: 11px; font-weight: 600; color: var(--muted); letter-spacing: .02em; }
.nav { display: flex; gap: 6px; margin-left: auto; background: rgba(255,255,255,.04); padding: 4px; border-radius: 12px; border: 1px solid var(--line); }
.nav a { color: var(--muted); text-decoration: none; padding: 8px 14px; border-radius: 9px; font-weight: 600; font-size: 14px; }
.nav a.on { color: #fff; background: linear-gradient(135deg, rgba(56,189,248,.25), rgba(167,139,250,.25)); }
.status { font-size: 12px; color: var(--muted); }

/* common */
h1.page { font-size: 30px; font-weight: 900; letter-spacing: -.03em; margin: 28px 0 4px; }
.sub { color: var(--muted); font-size: 14px; }
h2.sec { font-size: 18px; font-weight: 800; margin: 34px 0 12px; display: flex; align-items: center; gap: 10px; }
h2.sec .hint { font-size: 13px; font-weight: 500; color: var(--muted); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius); padding: 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,.25); }
.grid { display: grid; gap: 14px; }
.g3 { grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
.g2 { grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); }
.hidden { display: none !important; }
.muted { color: var(--muted); }
.pill { display: inline-flex; align-items: center; gap: 4px; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 999px;
  background: rgba(255,255,255,.06); color: var(--muted); white-space: nowrap; }
.pill.hot { background: rgba(251,191,36,.15); color: var(--gold); }
.pill.good { background: rgba(52,211,153,.15); color: var(--green); }
.pill.bad { background: rgba(248,113,113,.14); color: var(--red); }
.pill.live { background: rgba(248,113,113,.18); color: var(--red); }
.btn { background: rgba(255,255,255,.06); color: var(--text); border: 1px solid var(--line2); border-radius: 10px;
  padding: 9px 14px; font-weight: 600; font-size: 14px; cursor: pointer; }
.btn:hover { border-color: var(--cyan); }
.btn.primary { background: linear-gradient(135deg, var(--blue), var(--violet)); border: none; color: #fff; box-shadow: 0 6px 18px rgba(99,102,241,.35); }
.seg { display: inline-flex; background: rgba(255,255,255,.04); border: 1px solid var(--line); border-radius: 11px; padding: 3px; flex-wrap: wrap; }
.seg button { background: none; border: none; color: var(--muted); padding: 7px 12px; border-radius: 8px; font-weight: 600; font-size: 13px; cursor: pointer; }
.seg button.on { background: rgba(56,189,248,.18); color: #fff; }
select, input[type=text], input[type=number], input[type=search] { background: rgba(255,255,255,.05); color: var(--text);
  border: 1px solid var(--line2); border-radius: 10px; padding: 9px 11px; font-size: 14px; }
input:focus, select:focus { outline: none; border-color: var(--cyan); }
select option { background: var(--bg2); }
.loader { padding: 60px 20px; text-align: center; color: var(--muted); }
.spinner { width: 34px; height: 34px; border: 3px solid var(--line2); border-top-color: var(--cyan); border-radius: 50%;
  animation: spin 1s linear infinite; margin: 0 auto 14px; }
@keyframes spin { to { transform: rotate(360deg); } }
.progress { height: 8px; background: rgba(255,255,255,.06); border-radius: 99px; overflow: hidden; max-width: 380px; margin: 14px auto 0; }
.progress i { display: block; height: 100%; background: linear-gradient(90deg, var(--cyan), var(--violet)); transition: width .4s; }
.err { color: var(--red); }

/* overview */
.daytabs { margin: 18px 0 6px; }
.leaders { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
.leader h3 { margin: 0 0 12px; font-size: 14px; color: var(--muted); font-weight: 700; text-transform: uppercase; letter-spacing: .05em; }
.lrow { display: flex; align-items: center; gap: 10px; padding: 7px 0; border-bottom: 1px dashed var(--line); cursor: pointer; }
.lrow:last-child { border-bottom: none; }
.lrow .rk { width: 20px; color: var(--muted); font-weight: 700; font-size: 13px; }
.lrow img { width: 24px; height: 24px; }
.lrow .nm { flex: 1; font-weight: 600; font-size: 14px; line-height: 1.2; }
.lrow .nm small { display: block; color: var(--muted); font-weight: 500; font-size: 11px; }
.lrow .val { font-weight: 800; color: var(--gold); font-variant-numeric: tabular-nums; }
.matches { display: grid; grid-template-columns: repeat(auto-fill, minmax(370px, 1fr)); gap: 14px; }
.mcard { cursor: pointer; transition: transform .15s, border-color .15s; position: relative; overflow: hidden; }
.mcard:hover { transform: translateY(-2px); border-color: rgba(56,189,248,.5); }
.mcard .mtop { display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: var(--muted); font-weight: 600; }
.mteams { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 8px; margin: 12px 0 10px; }
.mteam { display: flex; flex-direction: column; align-items: center; text-align: center; gap: 4px; }
.mteam img { width: 52px; height: 52px; filter: drop-shadow(0 4px 10px rgba(0,0,0,.4)); }
.mteam b { font-size: 15px; }
.mteam small { color: var(--muted); font-size: 11px; }
.mmid { text-align: center; }
.mmid .xg { font-family: Anton, Impact, sans-serif; font-weight: 400; font-size: 44px; letter-spacing: .01em; line-height: 1; }
.mmid small { color: var(--muted); font-size: 11px; display: block; }
.wbar { display: flex; height: 8px; border-radius: 99px; overflow: hidden; background: rgba(255,255,255,.06); }
.wbar i { display: block; height: 100%; }
.wbar .h { background: linear-gradient(90deg, #0ea5e9, var(--home)); }
.wbar .a { background: linear-gradient(90deg, var(--away), #8b5cf6); }
.wlab { display: flex; justify-content: space-between; font-size: 12px; font-weight: 700; margin-top: 5px; }
.wlab .h { color: var(--home); } .wlab .a { color: var(--away); }
.odds { display: flex; justify-content: space-between; align-items: center; margin-top: 10px; font-size: 12px; color: var(--muted); gap: 8px; flex-wrap: wrap; }
.odd { display: inline-flex; align-items: center; gap: 6px; background: rgba(36,151,242,.12); border: 1px solid rgba(36,151,242,.35);
  color: #cfe7ff; font-weight: 800; padding: 3px 9px; border-radius: 8px; font-size: 13px; }
.chips { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; }
.chip { background: rgba(255,255,255,.04); border: 1px solid var(--line); border-radius: 11px; padding: 8px; text-align: center; }
.chip small { display: block; color: var(--muted); font-size: 11px; font-weight: 600; }
.chip b { font-size: 15px; font-variant-numeric: tabular-nums; }
.chip em { display: block; font-style: normal; font-size: 11px; color: var(--gold); font-weight: 700; }
.tops { margin-top: 12px; font-size: 13px; display: flex; flex-direction: column; gap: 4px; }
.tops div { display: flex; justify-content: space-between; gap: 6px; }
.tops span.p { color: var(--gold); font-weight: 800; }

/* detail */
.back { margin-top: 20px; }
.hero { margin-top: 14px; padding: 40px 26px 30px; position: relative; overflow: hidden; border-radius: 26px;
  background: radial-gradient(600px 320px at 0% 50%, color-mix(in srgb, var(--hc) 40%, transparent), transparent 70%),
              radial-gradient(600px 320px at 100% 50%, color-mix(in srgb, var(--ac) 40%, transparent), transparent 70%), linear-gradient(180deg, #0d1830, #0a1222); }
.hero .rink { position: absolute; inset: 0; width: 100%; height: 100%; opacity: .12; pointer-events: none; }
.hero > * { position: relative; }
.hero-grid { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 16px; }
.hteam { text-align: center; }
.hteam img { width: 150px; height: 150px; filter: drop-shadow(0 14px 30px rgba(0,0,0,.55)); }
.hteam h2 { margin: 6px 0 2px; font-family: Anton, Impact, sans-serif; font-weight: 400; text-transform: uppercase; font-size: 34px; letter-spacing: .02em; line-height: 1.15; }
.hteam .form { display: flex; gap: 4px; justify-content: center; margin-top: 8px; }
.res { width: 24px; height: 24px; border-radius: 6px; display: grid; place-items: center; font-size: 10px; font-weight: 800; }
.res.V { background: rgba(52,211,153,.2); color: var(--green); }
.res.P { background: rgba(248,113,113,.18); color: var(--red); }
.res.PP { background: rgba(251,146,60,.18); color: var(--orange); }
.hmid { text-align: center; min-width: 220px; }
.hmid .when { color: var(--muted); font-size: 13px; font-weight: 600; }
.hmid .score { font-family: Anton, Impact, sans-serif; font-weight: 400; font-size: 120px; letter-spacing: .01em; line-height: 1; margin: 8px 0 2px;
  background: linear-gradient(135deg, #fff, #b9d7ff); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hmid .lbl { color: var(--muted); font-size: 12px; }
.bigwin { margin-top: 22px; }
.bigwin .wbar { height: 14px; }
.bigwin .wlab { font-size: 15px; margin-top: 8px; }
.hero-actions { display: flex; gap: 10px; justify-content: center; margin-top: 18px; flex-wrap: wrap; }
.statrow { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; margin-top: 14px; }
.stat h4 { margin: 0; font-size: 13px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 700; }
.stat .total { font-family: Anton, Impact, sans-serif; font-weight: 400; font-size: 64px; letter-spacing: .01em; line-height: 1; margin: 6px 0 2px; color: var(--gold); }
.stat .total small { font-size: 14px; color: var(--muted); font-weight: 600; }
.cmp { margin-top: 10px; }
.cmp .r { display: grid; grid-template-columns: 42px 1fr 48px; align-items: center; gap: 8px; margin: 6px 0; font-size: 13px; font-weight: 700; }
.cmp .bar { height: 10px; border-radius: 99px; background: rgba(255,255,255,.06); overflow: hidden; }
.cmp .bar i { display: block; height: 100%; border-radius: 99px; }
.cmp .num { text-align: right; font-variant-numeric: tabular-nums; }
.mkt table { width: 100%; }
.mkt td input { width: 82px; padding: 6px 8px; text-align: right; }
.ev { font-weight: 800; }
.ev.good { color: var(--green); } .ev.bad { color: var(--red); }
.charts svg { width: 100%; height: auto; display: block; }
.legend { display: flex; gap: 14px; font-size: 12px; color: var(--muted); margin-top: 6px; flex-wrap: wrap; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
ul.why { margin: 0; padding-left: 18px; color: #c3cee0; font-size: 13.5px; line-height: 1.6; }
.gk { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px dashed var(--line); gap: 8px; }
.gk:last-child { border-bottom: none; }
.calc { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
.out { display: flex; gap: 18px; flex-wrap: wrap; margin-top: 12px; }
.out div { background: rgba(255,255,255,.04); border: 1px solid var(--line); padding: 10px 14px; border-radius: 12px; min-width: 130px; }
.out small { display: block; color: var(--muted); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
.out b { font-size: 22px; font-weight: 900; }
.disc { color: var(--muted); font-size: 13.5px; line-height: 1.65; margin-top: 28px; padding-top: 16px; border-top: 1px solid var(--line); }

/* tables */
.tablewrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 14px; background: var(--card); }
table.t { border-collapse: collapse; width: 100%; font-size: 14.5px; white-space: nowrap; }
table.t th, table.t td { padding: 9px 10px; text-align: right; border-bottom: 1px solid var(--line); }
table.t th { background: rgba(21,32,56,.95); position: sticky; top: 0; cursor: pointer; user-select: none; font-weight: 700;
  font-size: 12px; color: #b8c6dc; }
table.t th:hover { color: var(--cyan); }
table.t th.sorted { color: var(--gold); }
table.t .left { text-align: left; }
table.t tbody tr:hover { background: rgba(56,189,248,.06); }
table.t tr.dim td { opacity: .45; }
td.rank { color: var(--muted); width: 30px; }
td.pos { color: var(--muted); }
td.gold { color: var(--gold); font-weight: 800; }
.pbar { display: inline-block; height: 7px; border-radius: 4px; background: linear-gradient(90deg, var(--orange), var(--gold));
  vertical-align: middle; margin-right: 7px; }
.plus { color: var(--green); } .minus { color: var(--red); }
.dots { display: inline-flex; gap: 2px; }
.dots i { width: 7px; height: 7px; border-radius: 2px; background: rgba(255,255,255,.12); }
.dots i.y { background: var(--gold); }

/* stats view */
.controls { display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0; align-items: center; }
.scards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 18px; }
.scard .lbl { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; font-weight: 700; }
.scard .val { font-size: 30px; font-weight: 900; margin: 4px 0 2px; }
.scard .who { font-size: 13px; }
.thead { display: flex; align-items: center; gap: 14px; }
.thead img { width: 56px; height: 56px; }
.legendtxt { color: var(--muted); font-size: 13.5px; margin-top: 10px; line-height: 1.6; }

/* chat */
#chatbtn { position: fixed; right: 22px; bottom: 22px; z-index: 50; border: none; cursor: pointer; color: #fff; font-weight: 800;
  font-size: 15px; padding: 14px 20px; border-radius: 999px; background: linear-gradient(135deg, var(--blue), var(--violet));
  box-shadow: 0 10px 30px rgba(99,102,241,.5); display: flex; align-items: center; gap: 8px; }
#chat { position: fixed; right: 22px; bottom: 22px; z-index: 60; width: 420px; max-width: calc(100vw - 24px);
  height: 620px; max-height: calc(100vh - 40px); display: flex; flex-direction: column; background: #0d1527;
  border: 1px solid var(--line2); border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,.6); overflow: hidden; }
#chat header { display: flex; align-items: center; gap: 10px; padding: 14px 16px; border-bottom: 1px solid var(--line);
  background: linear-gradient(135deg, rgba(59,130,246,.2), rgba(167,139,250,.2)); }
#chat header b { flex: 1; }
#chat header small { display: block; color: var(--muted); font-weight: 500; font-size: 11px; }
#chat header button { background: none; border: none; color: var(--muted); font-size: 20px; cursor: pointer; }
#msgs { flex: 1; overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 10px; }
.msg { max-width: 88%; padding: 10px 13px; border-radius: 14px; font-size: 15px; line-height: 1.5; }
.msg.user { align-self: flex-end; background: linear-gradient(135deg, var(--blue), #6366f1); color: #fff; border-bottom-right-radius: 4px; }
.msg.ai { align-self: flex-start; background: rgba(255,255,255,.06); border: 1px solid var(--line); border-bottom-left-radius: 4px; }
.msg.ai p { margin: 0 0 6px; } .msg.ai p:last-child { margin: 0; }
.msg.ai ul { margin: 4px 0; padding-left: 18px; }
.msg.err { color: var(--red); }
.typing span { display: inline-block; width: 7px; height: 7px; margin: 0 2px; border-radius: 50%; background: var(--muted); animation: bl 1.2s infinite; }
.typing span:nth-child(2) { animation-delay: .2s; } .typing span:nth-child(3) { animation-delay: .4s; }
@keyframes bl { 0%, 80%, 100% { opacity: .2; } 40% { opacity: 1; } }
#sugg { display: flex; gap: 6px; flex-wrap: wrap; padding: 0 14px 10px; }
#sugg button { background: rgba(56,189,248,.1); border: 1px solid rgba(56,189,248,.3); color: #cdeaff; border-radius: 99px;
  padding: 6px 10px; font-size: 12px; cursor: pointer; }
#chatform { display: flex; gap: 8px; padding: 12px; border-top: 1px solid var(--line); }
#chatform input { flex: 1; }

@media (max-width: 720px) {
  body { zoom: 1; }
  .wrap { padding: 0 12px 90px; }
  details.gloss dl { grid-template-columns: 1fr; gap: 2px; } details.gloss dd { margin-bottom: 10px; }
  .nav { margin-left: 0; width: 100%; } .nav a { flex: 1; text-align: center; }
  .matches { grid-template-columns: 1fr; }
  .hero { padding: 18px 12px; }
  .topbar { position: static; }
  .hero-grid { grid-template-columns: 1fr 1fr; }
  .hmid { grid-column: 1 / -1; order: 3; }
  .mkt td input { width: 70px; }
  h1.page { font-size: 24px; }
  .hteam img { width: 80px; height: 80px; } .hteam h2 { font-size: 22px; }
  .hmid { min-width: 0; } .hmid .score { font-size: 76px; }
  .cine { padding: 40px 16px 30px; margin: 0 -12px; } .marquee { margin: 0 -12px; }
  .counters { grid-template-columns: repeat(2, 1fr); } .counters div { padding: 14px 10px 10px 0 !important; border-left: none !important; }
  .cine .lead { font-size: 15.5px; }
  .g2 { grid-template-columns: 1fr; }
  #chat { right: 6px; bottom: 6px; height: calc(100vh - 20px); }
  #chatbtn { right: 14px; bottom: 14px; }
}
/* ================= NHL TÉMA (televizní grafika) ================= */
:root {
  --bg: #05070b; --bg2: #0b0f16; --card: rgba(16, 21, 30, .82); --card2: #141b26;
  --line: rgba(200, 215, 235, .10); --line2: rgba(200, 215, 235, .20); --text: #f4f7fb; --muted: #8d9aae;
  --cyan: #4cc3ff; --blue: #2f7bff; --violet: #b9c6d8; --gold: #ffd166; --orange: #ff8a3d; --green: #35d39a; --red: #ff2d3d;
  --home: #4cc3ff; --away: #e8edf4;
  --sunset: linear-gradient(180deg, #ffffff 0%, #dfe7f1 45%, #9fb0c6 100%);
  --brand: linear-gradient(135deg, #2f7bff, #0d47c9);
}
body { background: radial-gradient(1100px 520px at 50% -12%, rgba(76,195,255,.12), transparent 60%), linear-gradient(180deg, #070a10, #05070b 40%); }
.topbar { background: rgba(5,7,11,.86); border-bottom: 1px solid rgba(255,255,255,.08); }
.brand .puck { background: #fff; color: #000; box-shadow: 0 0 0 2px #0b0f16, 0 0 0 3px rgba(255,255,255,.4); }
.nav a.on { background: #fff; color: #05070b; }
.btn.primary { background: var(--brand); box-shadow: 0 8px 24px rgba(47,123,255,.35); }
.seg button.on { background: #fff; color: #05070b; }
.card { background: linear-gradient(180deg, rgba(20,27,38,.9), rgba(12,16,23,.9)); box-shadow: 0 14px 40px rgba(0,0,0,.45); }
.wbar .h { background: linear-gradient(90deg, #2f7bff, var(--home)); }
.wbar .a { background: linear-gradient(90deg, #c9d3e0, #ffffff); }
.wlab .a { color: #e8edf4; }
.sechead .no { color: var(--cyan); font-size: 18px; letter-spacing: .1em; }
.sechead h2 { color: #fff; position: relative; padding-bottom: 6px; }
.sechead h2::after { content: ""; position: absolute; left: 0; bottom: -4px; width: 64px; height: 5px; background: var(--red); transform: skewX(-20deg); }
.pill.hot { background: rgba(255,45,61,.16); color: #ff6b77; }
.odd { background: rgba(255,255,255,.06); border-color: rgba(255,255,255,.25); color: #fff; }
.helpbox { background: rgba(76,195,255,.06); border-color: rgba(76,195,255,.22); }
table.t th { background: rgba(14,19,27,.98); color: #c9d4e3; text-transform: uppercase; letter-spacing: .04em; font-size: 11.5px; }
td.gold, .gold { color: #fff !important; }
.pbar { background: linear-gradient(90deg, #2f7bff, #4cc3ff); }
#chatbtn { background: var(--brand); box-shadow: 0 10px 30px rgba(47,123,255,.45); }
#slipbtn { background: #fff; color: #05070b; box-shadow: 0 10px 30px rgba(255,255,255,.18); }
#slipbtn .cnt { background: var(--red); color: #fff; }
#chat { background: #0b0f16; } #chat header { background: linear-gradient(135deg, rgba(47,123,255,.22), rgba(255,255,255,.05)); }
.msg.user { background: var(--brand); }
#progress { background: linear-gradient(90deg, #2f7bff, #4cc3ff, #fff); }
.marquee { background: #fff; border-color: #fff; } .marquee span { color: #05070b; } .marquee span em { color: var(--red); }
.counters b { color: #fff; }
.mcard .wmL, .mcard .wmR { opacity: .06; }
.res.V { background: rgba(53,211,154,.18); } .res.P { background: rgba(255,45,61,.18); color: #ff6b77; }
.scoreboard .digits { color: #ff2d3d; text-shadow: 0 0 18px rgba(255,45,61,.6); }
/* ikony */
.ic { display: inline-block; width: 1.05em; height: 1.05em; vertical-align: -0.16em; flex: none; }
.ic svg { width: 100%; height: 100%; display: block; }
/* ---------- hero: aréna ---------- */
.hero2 { background: none !important; padding-top: 80px; }
.hero2 .sky { position: absolute; inset: 0; background:
    radial-gradient(60% 40% at 50% 0%, rgba(255,255,255,.18), transparent 70%),
    radial-gradient(1px 1px at 20% 30%, rgba(255,255,255,.5), transparent), 
    linear-gradient(180deg, #0a0e15 0%, #0d131d 45%, #111a27 62%, #05070b 100%); }
.hero2 .sky::before { content: ""; position: absolute; inset: 0 0 38% 0; opacity: .55;
  background-image: radial-gradient(circle, rgba(255,255,255,.35) 0 1px, transparent 1.5px), radial-gradient(circle, rgba(76,195,255,.25) 0 1px, transparent 1.5px);
  background-size: 14px 12px, 23px 19px; background-position: 0 0, 7px 5px; -webkit-mask: linear-gradient(180deg, transparent, #000 35%, #000 80%, transparent); mask: linear-gradient(180deg, transparent, #000 35%, #000 80%, transparent); }
.hero2 .sun { position: absolute; inset: 0; pointer-events: none; border-radius: 0; width: auto; transform: none; animation: none; filter: none; -webkit-mask: none; mask: none;
  background: conic-gradient(from 180deg at 18% -10%, transparent 0 8deg, rgba(255,255,255,.12) 10deg 14deg, transparent 16deg),
              conic-gradient(from 180deg at 50% -12%, transparent 0 172deg, rgba(255,255,255,.10) 176deg 184deg, transparent 188deg),
              conic-gradient(from 180deg at 82% -10%, transparent 0 344deg, rgba(255,255,255,.12) 346deg 350deg, transparent 352deg);
  animation: beams 7s ease-in-out infinite alternate; }
@keyframes beams { from { opacity: .55; } to { opacity: 1; } }
.hero2 .skyline { position: absolute; left: 0; right: 0; top: 60%; height: 10%; width: 100%; }
.hero2 .stage3d { top: 66%; perspective: 900px; }
.hero2 #icecv { inset: 0 -10%; width: 120%; height: 100%; transform: rotateX(50deg); transform-origin: 50% 0%; }
.hero2 #frost { display: none; }
.hero2 .backword { display: block; top: 38%; -webkit-text-stroke: 2px rgba(255,255,255,.06); }
.hero2 h1 { font-size: clamp(60px, 10.5vw, 165px) !important; line-height: 1.08 !important; text-shadow: none !important; }
.hero2 h1 .ln { padding-top: .2em !important; margin-top: -.2em !important; }
.hero2 h1 .ln > i { background: linear-gradient(180deg, #ffffff 0%, #eef3f9 50%, #a9b8cc 100%); -webkit-background-clip: text; background-clip: text; color: transparent;
  -webkit-text-stroke: 0; filter: drop-shadow(0 6px 0 rgba(0,0,0,.55)); }
.hero2 h1 .ln:nth-child(2) > i { background: linear-gradient(180deg, #bfe9ff, #4cc3ff); -webkit-background-clip: text; background-clip: text; color: transparent; -webkit-text-stroke: 0; }
.hero2 .lead { color: #dfe7f1 !important; text-shadow: 0 2px 12px rgba(0,0,0,.6); }
.hero2 .counters { border-top-color: rgba(255,255,255,.22); }
.hero2 .help, .hero2 .hint2 { color: #c9d4e3 !important; }
.badge-gta { display: inline-flex; align-items: center; gap: 10px; padding: 0; border-radius: 0; background: none; border: none;
  font-family: Anton, Impact, sans-serif; letter-spacing: .1em; font-size: 16px; text-transform: uppercase; position: relative; z-index: 3; color: #fff; }
.hero2 .counters { background: rgba(5,7,11,.86); backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,.12) !important; border-radius: 16px; padding: 4px 18px 14px; margin-top: 34px; }
.hero2 .counters div + div { border-left-color: rgba(255,255,255,.12); }
.hero2 > .help, .hero2 .hint2 { display: inline-block; background: rgba(5,7,11,.82); padding: 5px 12px; border-radius: 8px; margin-right: auto; }
.hero2 .daytabs { background: rgba(5,7,11,.86) !important; }
.hero2 .skyline { z-index: 1; }
.badge-gta::before { content: ""; width: 34px; height: 6px; background: var(--red); transform: skewX(-20deg); }
/* ---------- NHL ID karta ---------- */
.idwrap { display: grid; grid-template-columns: minmax(280px, 360px) 1fr; gap: 26px; align-items: start; margin-top: 18px; }
.idctl .row { margin-bottom: 14px; } .idctl label { display: block; font-size: 12px; font-weight: 800; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; margin-bottom: 6px; }
.idctl select, .idctl input { width: 100%; }
.themes { display: flex; gap: 8px; } .themes button { flex: 1; height: 42px; border-radius: 10px; border: 2px solid transparent; cursor: pointer; }
.themes button.on { border-color: #fff; }
.idstage { perspective: 1600px; display: flex; justify-content: center; padding: 20px 0 40px; }
.idcard { width: min(100%, 640px); aspect-ratio: 1.586; position: relative; transform-style: preserve-3d; transition: transform .9s cubic-bezier(.2,.8,.2,1); cursor: pointer; }
.idcard.flip { transform: rotateY(180deg); }
.idface { position: absolute; inset: 0; border-radius: 22px; overflow: hidden; backface-visibility: hidden; -webkit-backface-visibility: hidden;
  box-shadow: 0 30px 70px rgba(0,0,0,.55), inset 0 0 0 1px rgba(255,255,255,.25); color: #241031; }
.idface.back { transform: rotateY(180deg); }
.idface .bg { position: absolute; inset: 0; background: var(--cardbg); }
.idface .holo { position: absolute; inset: 0; mix-blend-mode: color-dodge; opacity: .35; pointer-events: none;
  background: linear-gradient(115deg, transparent 20%, rgba(180,200,230,.6) 35%, rgba(255,255,255,.75) 45%, rgba(150,210,255,.6) 55%, transparent 70%);
  background-size: 250% 250%; background-position: var(--hx, 50%) var(--hy, 50%); transition: background-position .1s; }
.idface .hdr { position: absolute; left: 0; right: 0; top: 0; height: 17%; display: flex; align-items: center; justify-content: space-between; padding: 0 5%;
  background: linear-gradient(100deg, #05070b 0 62%, var(--tc) 62% 78%, #ffffff 78% 80%, var(--tc) 80%); color: #fff; }
.idface .hdr b { font-family: Anton, Impact, sans-serif; font-size: clamp(18px, 3.4vw, 30px); letter-spacing: .06em; text-shadow: 2px 2px 0 rgba(0,0,0,.25); }
.idface .hdr span { font-family: Anton, Impact, sans-serif; font-size: clamp(11px, 1.7vw, 15px); letter-spacing: .12em; opacity: .9; }
.idface .photo { position: absolute; left: 5%; top: 23%; width: 28%; height: 62%; border-radius: 14px; overflow: hidden;
  background: linear-gradient(160deg, var(--tc), #12081f); display: grid; place-items: center; box-shadow: inset 0 0 0 3px rgba(255,255,255,.6); }
.idface .photo .num { font-family: Anton, Impact, sans-serif; font-size: clamp(60px, 12vw, 120px); color: #fff; line-height: 1; text-shadow: 4px 4px 0 rgba(0,0,0,.3); }
.idface .photo img { position: absolute; width: 44%; right: 6%; bottom: 6%; opacity: .95; filter: drop-shadow(0 3px 6px rgba(0,0,0,.4)); }
.idface .fields { position: absolute; left: 38%; right: 5%; top: 22%; display: grid; grid-template-columns: 1fr 1fr; gap: 3.2% 6%; font-family: "JetBrains Mono", monospace; }
.idface .fields div small { display: block; font-size: clamp(8px, 1.15vw, 11px); font-weight: 700; color: #6b7a90; letter-spacing: .1em; }
.idface .fields div b { font-size: clamp(11px, 1.9vw, 18px); color: #1d0b2a; text-transform: uppercase; }
.idface .fields .wide { grid-column: 1 / -1; }
.idface .fields .wide b { font-family: Anton, Impact, sans-serif; font-weight: 400; font-size: clamp(20px, 4vw, 38px); letter-spacing: .02em; }
.idface .sig { position: absolute; left: 38%; bottom: 7%; font-family: Caveat, cursive; font-size: clamp(20px, 3.6vw, 34px); color: #0d2a6e; transform: rotate(-4deg); }
.idface::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 3.5%; background: linear-gradient(90deg, var(--tc) 0 70%, #fff 70% 72%, #05070b 72%); }
.idface .bar { position: absolute; right: 5%; bottom: 7%; width: 26%; height: 11%; }
.idface .stamp { position: absolute; right: 7%; top: 21%; width: 17%; aspect-ratio: 1; opacity: .9; }
.idface .bstats { position: absolute; inset: 21% 5% 6% 5%; display: grid; grid-template-columns: repeat(4, 1fr); grid-auto-rows: min-content; gap: 10px; font-family: "JetBrains Mono", monospace; }
.idface .bstats div { background: rgba(255,255,255,.55); border-radius: 10px; padding: 8px 10px; }
.idface .bstats small { display: block; font-size: clamp(8px, 1.1vw, 11px); color: #6b7a90; font-weight: 700; letter-spacing: .08em; }
.idface .bstats b { font-family: Anton, Impact, sans-serif; font-weight: 400; font-size: clamp(18px, 3vw, 30px); color: #1d0b2a; }
.idface .bstats .spark { grid-column: 1 / -1; background: rgba(29,11,42,.85); color: #fff; }
.idface .bstats .spark small { color: #9fd3ff; }
.idnote { text-align: center; color: var(--muted); font-size: 13px; margin-top: -18px; }
/* ---------- tikety ---------- */
.drop { border: 2px dashed var(--line2); border-radius: 18px; padding: 30px 20px; text-align: center; cursor: pointer; transition: border-color .2s, background .2s; }
.drop:hover, .drop.over { border-color: var(--cyan); background: rgba(76,195,255,.06); }
.drop img { max-height: 320px; max-width: 100%; border-radius: 12px; margin-top: 12px; box-shadow: 0 10px 30px rgba(0,0,0,.4); }
.tkgrid { display: grid; grid-template-columns: minmax(280px, 420px) 1fr; gap: 18px; align-items: start; }
.verdict { display: inline-flex; align-items: center; gap: 8px; padding: 8px 14px; border-radius: 12px; font-family: Anton, Impact, sans-serif; font-size: 22px; letter-spacing: .04em; text-transform: uppercase; }
.verdict.good { background: rgba(53,211,154,.16); color: var(--green); } .verdict.bad { background: rgba(255,45,61,.16); color: #ff6b77; } .verdict.mid { background: rgba(255,209,102,.14); color: var(--gold); }
.tkleg td small { display: block; color: var(--muted); font-size: 12px; white-space: normal; }
.tkleg td { white-space: normal; vertical-align: top; }
.tkhist { display: flex; gap: 10px; overflow-x: auto; padding-bottom: 6px; }
.tkhist .h { flex: 0 0 auto; min-width: 180px; cursor: pointer; }
@media (max-width: 900px) { .tkgrid { grid-template-columns: 1fr; } }
/* ---------- modal hráče ---------- */
#pmodal { position: fixed; inset: 0; z-index: 120; background: rgba(8,3,16,.72); backdrop-filter: blur(6px); display: grid; place-items: center; padding: 16px; }
#pmodal .box { width: min(860px, 100%); max-height: calc(100vh - 32px); overflow-y: auto; background: linear-gradient(180deg, #24123c, #160b26); border: 1px solid var(--line2);
  border-radius: 22px; padding: 22px; box-shadow: 0 30px 80px rgba(0,0,0,.6); animation: pop .4s cubic-bezier(.2,1.2,.4,1) both; }
#pmodal .x { float: right; background: none; border: none; color: var(--muted); font-size: 24px; cursor: pointer; }
.pname-link { cursor: pointer; border-bottom: 1px dashed rgba(255,255,255,.3); }
.pname-link:hover { color: var(--gold); border-color: var(--gold); }
.injbadge { display: inline-flex; gap: 4px; align-items: center; font-size: 11px; font-weight: 800; padding: 2px 7px; border-radius: 6px; background: rgba(255,84,112,.16); color: var(--red); }
.injbadge.dtd { background: rgba(255,194,75,.14); color: var(--gold); }
.goalrow { display: grid; grid-template-columns: 56px 1fr auto; gap: 10px; align-items: start; padding: 12px 0; border-bottom: 1px dashed var(--line); }
.goalrow .t { font-family: Anton, Impact, sans-serif; font-size: 20px; color: var(--gold); }
.goalrow .acts { display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }
.airesp { grid-column: 1 / -1; background: rgba(45,226,230,.07); border: 1px solid rgba(45,226,230,.25); border-radius: 12px; padding: 10px 12px; font-size: 14px; line-height: 1.55; }
.books td.best { color: var(--green); font-weight: 900; }
.retro { background: linear-gradient(90deg, rgba(255,122,69,.18), rgba(255,61,154,.18)); border: 1px solid rgba(255,122,69,.35); border-radius: 14px; padding: 12px 16px; margin-top: 14px; }
@media (max-width: 900px) { .idwrap { grid-template-columns: 1fr; } }
@media (max-width: 720px) { .nav { overflow-x: auto; flex-wrap: nowrap; scrollbar-width: none; } .nav a { flex: 0 0 auto !important; padding: 8px 12px; } }
@media (max-width: 720px) { .goalrow { grid-template-columns: 48px 1fr; } .goalrow .acts { grid-column: 1 / -1; justify-content: flex-start; } }
@media (max-width: 720px) { .hero2 { padding-top: 50px; } .hero2 .sun { top: 30%; width: 80vw; } .idface .bstats { grid-template-columns: repeat(3, 1fr); gap: 6px; } }
</style>
</head>
<body>
<div id="progress"></div>
<div id="cur"></div>
<div id="pmodal" class="hidden"><div class="box" id="pmbox"></div></div>
<div class="topbar"><div class="wrap">
  <a class="brand" href="#"><span class="puck">🏒</span><span>NHL Predikce<small>body · střely · vyloučení · kurzy</small></span></a>
  <nav class="nav">
    <a href="#" id="nav-over" class="on">🔮 Předpovědi</a>
    <a href="#live" id="nav-live"><span class="livedot"></span> Live</a>
    <a href="#presnost" id="nav-acc">🎯 Přesnost</a>
    <a href="#tikety" id="nav-tk">🎫 Tikety</a>
    <a href="#karta" id="nav-card">🪪 NHL ID</a>
    <a href="#statistiky" id="nav-stats">📊 Statistiky</a>
  </nav>
</div></div>

<div class="wrap">
  <section id="view-over"></section>
  <section id="view-game" class="hidden"></section>
  <section id="view-live" class="hidden"></section>
  <section id="view-acc" class="hidden"></section>
  <section id="view-card" class="hidden"></section>
  <section id="view-tk" class="hidden"></section>
  <section id="view-stats" class="hidden">
    <div class="thead" style="margin-top:26px"><img id="slogo" alt=""><div><h1 class="page" id="stitle" style="margin:0">Statistiky týmu</h1><div class="sub" id="ssub"></div></div></div>
    <div class="controls">
      <select id="team"></select>
      <select id="season"></select>
      <div class="seg" id="gtype"><button data-v="2" class="on">Základní část</button><button data-v="3">Play-off</button></div>
      <div class="seg" id="posf"><button data-v="all" class="on">Všichni</button><button data-v="F">Útočníci</button><button data-v="D">Obránci</button></div>
      <input id="search" type="search" placeholder="Hledat hráče…">
    </div>
    <p class="help" style="margin:-6px 0 14px">Tady vidíš statistiky hráčů vybraného týmu za celou sezónu. Nahoře vyber tým a sezónu. <b>Kliknutím na název sloupce</b> tabulku seřadíš (třeba podle bodů nebo trestných minut). Vysvětlivky zkratek jsou pod tabulkou.</p>
    <div class="scards" id="scards"></div>
    <div class="tablewrap"><table class="t" id="skaters"></table></div>
    <h2 class="sec">Brankáři</h2>
    <div class="tablewrap"><table class="t" id="goalies"></table></div>
    <div class="legendtxt">Z = zápasy · G = góly · A = asistence · <b>B = kanadské body</b> · B/Z = body na zápas · TM = trestné minuty ·
      S = střely na branku · %S = úspěšnost střelby · PPG/SHG = góly v přesilovce/oslabení · GWG = vítězné góly · TOI = čas na ledě ·
      %VB = vhazování. Brankáři: V/P/PP = výhry/prohry/prohry v prodl. · GAA = inkasované góly na zápas · %Z = úspěšnost zásahů · ČK = čistá konta.</div>
  </section>
</div>

<button id="slipbtn">🎫 Můj tiket<span class="cnt" id="slipcnt">0</span></button>
<div id="slip" class="hidden">
  <header><span style="font-size:22px">🎫</span><b>Můj tiket</b><button id="slipclose" title="Zavřít">✕</button></header>
  <div id="slipitems"></div>
  <div id="slipsum"></div>
</div>
<button id="chatbtn">🤖 Zeptej se AI</button>
<div id="chat" class="hidden">
  <header><span style="font-size:22px">🤖</span><b>AI hokejový analytik<small id="chatctx">Všechny nadcházející zápasy</small></b>
    <button id="chatclose" title="Zavřít">✕</button></header>
  <div id="msgs"></div>
  <div id="sugg"></div>
  <form id="chatform"><input id="chatin" type="text" placeholder="Napiš otázku…" autocomplete="off"><button class="btn primary">➤</button></form>
</div>

<script>
// ===================================================================
//  Pomocné funkce
// ===================================================================
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pc = (v) => (v == null ? "–" : Math.round(v * 100) + " %");
const num = (v, d = 2) => (v == null ? "–" : Number(v).toFixed(d));
const fair = (p) => (p > 0.005 ? (1 / p).toFixed(2) : "–");
const toi = (sec) => { if (sec == null) return "–"; const s = Math.round(sec); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
const posName = { C: "C", L: "LW", R: "RW", D: "D", G: "G" };
const fmtTime = (iso) => new Date(iso).toLocaleTimeString("cs-CZ", { hour: "2-digit", minute: "2-digit" });
const fmtDay = (iso) => new Date(iso).toLocaleDateString("cs-CZ", { weekday: "long", day: "numeric", month: "numeric" });
const dayKey = (iso) => { const d = new Date(iso); return `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`; };
const fmtDate = (s) => { const [, m, d] = String(s).split("-"); return `${+d}. ${+m}.`; };
const fmtSeason = (s) => { s = String(s); return `${s.slice(0, 4)}/${s.slice(6, 8)}`; };
const name = (p) => `${p.firstName?.default ?? ""} ${p.lastName?.default ?? ""}`.trim();
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { if (v === "" || v == null) localStorage.removeItem(k); else localStorage.setItem(k, v); } catch (e) {} },
};
async function getJSON(url) {
  const r = await fetch(url);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || "Chyba " + r.status);
  return j;
}
function segment(el, cb) {
  el.querySelectorAll("button").forEach((b) => (b.onclick = () => {
    el.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    cb(b.dataset.v);
  }));
}
function evHtml(p, odd) {
  if (!odd || odd <= 1 || !p) return `<span class="muted">–</span>`;
  const ev = p * odd - 1;
  return `<span class="ev ${ev > 0 ? "good" : "bad"}">${ev > 0 ? "+" : ""}${(ev * 100).toFixed(1)} %${ev > 0.03 ? " ✅" : ""}</span>`;
}

const TEAM_COLOR = { ANA: "#F47A38", BOS: "#FFB81C", BUF: "#2563eb", CGY: "#C8102E", CAR: "#CE1126", CHI: "#CF0A2C", COL: "#8a2a4f",
  CBJ: "#1d4ed8", DAL: "#00a86b", DET: "#CE1126", EDM: "#FF4C00", FLA: "#C8102E", LAK: "#A2AAAD", MIN: "#2e7d4f", MTL: "#AF1E2D",
  NSH: "#FFB81C", NJD: "#CE1126", NYI: "#1d6fd8", NYR: "#2563eb", OTT: "#C52032", PHI: "#F74902", PIT: "#FCB514", SJS: "#00a3ad",
  SEA: "#68c6c6", STL: "#2563eb", TBL: "#2563eb", TOR: "#2563eb", UTA: "#6CACE4", VAN: "#1d6fd8", VGK: "#B4975A", WSH: "#C8102E", WPG: "#3b6fd8" };
const tcol = (t) => TEAM_COLOR[t] || "#38bdf8";
const RINK = `<svg class="rink" viewBox="0 0 1200 520" preserveAspectRatio="xMidYMid slice" fill="none" stroke-width="3">
  <rect x="20" y="20" width="1160" height="480" rx="170" stroke="#cfe6ff"/>
  <line x1="600" y1="20" x2="600" y2="500" stroke="#ff4d5e" stroke-width="6"/>
  <line x1="420" y1="20" x2="420" y2="500" stroke="#4da3ff" stroke-width="7"/><line x1="780" y1="20" x2="780" y2="500" stroke="#4da3ff" stroke-width="7"/>
  <line x1="110" y1="20" x2="110" y2="500" stroke="#ff4d5e"/><line x1="1090" y1="20" x2="1090" y2="500" stroke="#ff4d5e"/>
  <circle cx="600" cy="260" r="90" stroke="#4da3ff"/><circle cx="600" cy="260" r="6" fill="#4da3ff"/>
  <circle cx="260" cy="140" r="70" stroke="#ff4d5e"/><circle cx="260" cy="380" r="70" stroke="#ff4d5e"/>
  <circle cx="940" cy="140" r="70" stroke="#ff4d5e"/><circle cx="940" cy="380" r="70" stroke="#ff4d5e"/>
  <path d="M110 230 a30 30 0 0 1 0 60" stroke="#ff4d5e"/><path d="M1090 230 a30 30 0 0 0 0 60" stroke="#ff4d5e"/></svg>`;
function splitName(n) { const i = String(n).indexOf(" "); return i < 0 ? ["", n] : [n.slice(0, i), n.slice(i + 1)]; }
function playerCard(p, logo, extra = "") {
  const [first, last] = splitName(p.name);
  const dots = (p.last5 || []).slice().reverse().map((x) => `<i class="${x > 0 ? "y" : ""}"></i>`).join("");
  return `<div class="pcard reveal tilt" style="--tc:${tcol(p.team)}" ${extra}><span class="glare"></span>
    <div class="jn">${p.num ?? ""}</div><img class="wm" src="${esc(logo)}" alt="">
    ${p.streak >= 3 ? `<span class="pill hot streak">🔥 ${p.streak} v řadě</span>` : ""}
    <div class="top"><img src="${esc(logo)}" alt=""><small>${esc(p.team)} · ${posName[p.pos] || p.pos}${p.num != null ? " · #" + p.num : ""}</small></div>
    <div class="pname"><span>${esc(first)}</span><span class="pname-link" data-player="${p.id}:${p.team}" style="font-size:inherit;color:inherit;display:inline">${esc(last)}</span></div>
    <div class="big">${pc(p.pPoint)}<small>šance na kanadský bod</small></div>
    <div class="pst"><div><b>${pc(p.pGoal)}</b><small>šance na gól</small></div><div><b>${num(p.expShots, 1)}</b><small>oček. střely</small></div>
      <div><b><span class="dots">${dots}</span></b><small>posl. 5 zápasů</small></div></div>
  </div>`;
}
// ---------- interaktivita ----------
function erf(x) { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; }
const ncdf = (x, m, sd) => 0.5 * (1 + erf((x - m) / (sd * Math.SQRT2)));
function pois(k, l) { let p = Math.exp(-l); for (let i = 1; i <= k; i++) p *= l / i; return p; }
function poisCdf(k, l) { let s = 0; for (let i = 0; i <= k; i++) s += pois(i, l); return s; }
function countUp(root) {
  root.querySelectorAll("[data-count]").forEach((el) => {
    const to = +el.dataset.count, t0 = performance.now(), dur = 1400;
    const step = (t) => { const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3); el.textContent = Math.round(to * e); if (k < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  });
}
function tilt(root) {
  if (matchMedia("(hover: none)").matches) return;
  root.querySelectorAll(".tilt").forEach((c) => {
    c.onmousemove = (e) => {
      const r = c.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
      c.style.transform = `perspective(900px) rotateY(${(x - 0.5) * 10}deg) rotateX(${(0.5 - y) * 10}deg) translateY(-4px)`;
      c.style.setProperty("--gx", x * 100 + "%"); c.style.setProperty("--gy", y * 100 + "%");
    };
    c.onmouseleave = () => { c.style.transform = ""; };
  });
}
function countdowns() {
  document.querySelectorAll(".countdown[data-start]").forEach((el) => {
    const ms = new Date(el.dataset.start) - Date.now();
    if (ms <= 0) { el.textContent = "právě se hraje / brzy začne"; return; }
    const h = Math.floor(ms / 3600000), m = Math.floor((ms % 3600000) / 60000), s = Math.floor((ms % 60000) / 1000);
    el.textContent = `začíná za ${h > 23 ? Math.floor(h / 24) + " d " + (h % 24) + " h" : h + " h " + String(m).padStart(2, "0") + " min " + String(s).padStart(2, "0") + " s"}`;
  });
}
setInterval(countdowns, 1000);
function interactive(root) {
  tilt(root); countUp(root); countdowns();
  const cine = root.querySelector("#cine");
  if (cine) cine.onmousemove = (e) => {
    const r = cine.getBoundingClientRect();
    cine.style.setProperty("--mx", e.clientX - r.left + "px"); cine.style.setProperty("--my", e.clientY - r.top + "px");
  };
  // pruhy naběhnou od nuly
  root.querySelectorAll(".wbar i, .cmp .bar i").forEach((b) => { const w = b.style.width; b.style.width = "0"; requestAnimationFrame(() => setTimeout(() => (b.style.width = w), 60)); });
}
window.addEventListener("scroll", () => {
  const h = document.documentElement, p = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
  $("progress").style.width = p * 100 + "%";
  const cine = document.getElementById("cine");
  if (cine && !cine.closest(".hidden")) {
    const y = Math.min(window.scrollY, 900);
    const h1 = cine.querySelector("h1");
    if (cine.classList.contains("hero2")) {
      const lns = h1.querySelectorAll(".ln");
      if (lns[0]) lns[0].style.transform = `translateX(${-y * 0.5}px)`;
      if (lns[1]) lns[1].style.transform = `translateX(${y * 0.5}px)`;
      h1.style.opacity = 1 - y / 650;
      const bw = document.getElementById("backword");
      if (bw) bw.style.transform = `translate(-50%, -50%) scale(${1 + y / 500})`, (bw.style.opacity = 1 - y / 600);
      const cv = document.getElementById("icecv");
      if (cv) cv.style.transform = `rotateX(${Math.min(75, 52 + y / 25)}deg)`;
      const sun = cine.querySelector(".sun");
      if (sun) sun.style.transform = `translate(-50%, ${-50 + y / 8}%)`;
    } else {
      h1.style.transform = `translateY(${y * 0.35}px)`;
      h1.style.opacity = 1 - y / 700;
    }
    cine.querySelector(".lead").style.transform = `translateY(${y * 0.2}px)`;
  }
}, { passive: true });
let SL = { kind: "shots.total" };
function slCfg() {
  const [grp, side] = SL.kind.split(".");
  const d = GAME;
  if (grp === "shots") { const b = d.shots[side]; return { normal: true, mean: b.mean ?? b.exp, sd: b.sd ?? 6, lab: "střel", step: 1 }; }
  if (grp === "goals") return { normal: false, mean: d.goals.total, lab: "gólů" };
  if (grp === "hits" || grp === "blocks") { const b = d.extra[grp][side]; return { normal: true, mean: b.mean, sd: b.sd, lab: grp === "hits" ? "hitů" : "bloků" }; }
  return { normal: false, mean: d.pens[side].exp, lab: "vyloučení" };
}
function initSlider() {
  segment($("slkind"), (v) => { SL.kind = v; setupRange(); });
  $("slrange").oninput = updateSlider;
  setupRange();
}
function setupRange() {
  const c = slCfg(), r = $("slrange");
  const lo = Math.max(0.5, Math.floor(c.mean - (c.normal ? 3 * c.sd : 3 * Math.sqrt(c.mean) + 1)) + 0.5);
  const hi = Math.floor(c.mean + (c.normal ? 3 * c.sd : 3 * Math.sqrt(c.mean) + 2)) + 0.5;
  r.min = lo; r.max = hi; r.step = 1; r.value = Math.floor(c.mean) + 0.5;
  updateSlider();
}
function updateSlider() {
  const c = slCfg(), line = +$("slrange").value;
  const over = c.normal ? 1 - ncdf(line, c.mean, c.sd) : 1 - poisCdf(Math.floor(line), c.mean);
  $("slval").textContent = line.toFixed(1);
  $("slpo").textContent = pc(over); $("slpu").textContent = pc(1 - over);
  $("slfa").textContent = `${Math.floor(line) + 1} a víc ${c.lab} · férový kurz ${fair(over)}`;
  $("slfb").textContent = `${Math.floor(line)} a míň ${c.lab} · férový kurz ${fair(1 - over)}`;
  $("slfo").style.width = over * 100 + "%"; $("slfu").style.width = (1 - over) * 100 + "%";
  // graf rozdělení
  const lo = +$("slrange").min - 0.5, hi = +$("slrange").max + 0.5, ks = [];
  for (let k = Math.ceil(lo); k <= Math.floor(hi); k++) ks.push(k);
  const pk = (k) => (c.normal ? ncdf(k + 0.5, c.mean, c.sd) - ncdf(k - 0.5, c.mean, c.sd) : pois(k, c.mean));
  const W = 620, H = 150, mx = Math.max(...ks.map(pk)) * 1.1, bw = W / ks.length;
  let svg = `<svg viewBox="0 0 ${W} ${H + 22}" xmlns="http://www.w3.org/2000/svg" font-family="Inter,sans-serif">`;
  ks.forEach((k, i) => {
    const h = (pk(k) / mx) * H, isO = k > line;
    svg += `<rect x="${i * bw + 1}" y="${H - h}" width="${bw - 2}" height="${h}" rx="2" fill="${isO ? "#34d399" : "#f87171"}" opacity=".75"><title>${k} ${c.lab}: ${pc(pk(k))}</title></rect>`;
    if (ks.length < 30 || k % 2 === 0) svg += `<text x="${i * bw + bw / 2}" y="${H + 15}" font-size="10" fill="#8796ae" text-anchor="middle">${k}</text>`;
  });
  const lx = ((line - ks[0] + 0.5) / ks.length) * W;
  svg += `<line x1="${lx}" x2="${lx}" y1="0" y2="${H}" stroke="#fbbf24" stroke-width="3"/></svg>`;
  $("sldist").innerHTML = svg;
}

// ---------- pohyblivý hero: led, puk, námraza ----------
let HERO = null;
function initHero2() {
  if (HERO) { HERO.stop(); HERO = null; }
  const sec = document.getElementById("cine"), cv = document.getElementById("icecv"), fr = document.getElementById("frost");
  if (!sec || !cv || !fr || !sec.classList.contains("hero2")) return;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ctx = cv.getContext("2d"), fx = fr.getContext("2d");
  const DPR = Math.min(1.5, window.devicePixelRatio || 1);
  let W = 0, H = 0, FW = 0, FH = 0, frostTex = null, run = true, visible = true, raf = 0, t0 = performance.now();
  const trail = document.createElement("canvas"), tx = trail.getContext("2d");
  const mouse = { x: 0, y: 0, active: false };
  const puck = { x: 0, y: 0, vx: 0, vy: 0, free: 0 };
  const parts = [], snow = [];
  function size() {
    const r = cv.getBoundingClientRect(), rf = sec.getBoundingClientRect();
    W = Math.max(10, r.width); H = Math.max(10, r.height); FW = rf.width; FH = rf.height;
    for (const [c, w, h] of [[cv, W, H], [trail, W, H], [fr, FW, FH]]) { c.width = w * DPR; c.height = h * DPR; }
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0); tx.setTransform(DPR, 0, 0, DPR, 0, 0); fx.setTransform(DPR, 0, 0, DPR, 0, 0);
    if (!puck.x) { puck.x = W * 0.62; puck.y = H * 0.55; }
    // textura námrazy
    frostTex = document.createElement("canvas"); frostTex.width = FW * DPR; frostTex.height = FH * DPR;
    const f = frostTex.getContext("2d"); f.setTransform(DPR, 0, 0, DPR, 0, 0);
    const g = f.createLinearGradient(0, 0, 0, FH); g.addColorStop(0, "rgba(190,220,255,.10)"); g.addColorStop(1, "rgba(190,220,255,.03)");
    f.fillStyle = g; f.fillRect(0, 0, FW, FH);
    for (let i = 0; i < (FW * FH) / 900; i++) { f.fillStyle = `rgba(255,255,255,${Math.random() * 0.22})`; f.fillRect(Math.random() * FW, Math.random() * FH, 1.4, 1.4); }
    f.strokeStyle = "rgba(255,255,255,.09)"; f.lineWidth = 1;
    for (let i = 0; i < 70; i++) { // krystaly
      let x = Math.random() * FW, y = Math.random() * FH, a = Math.random() * 6.28;
      for (let j = 0; j < 7; j++) { const l = 20 + Math.random() * 60; f.beginPath(); f.moveTo(x, y); x += Math.cos(a) * l; y += Math.sin(a) * l; f.lineTo(x, y); f.stroke();
        for (let k = -1; k <= 1; k += 2) { f.beginPath(); f.moveTo(x, y); f.lineTo(x + Math.cos(a + k * 0.7) * l * 0.35, y + Math.sin(a + k * 0.7) * l * 0.35); f.stroke(); }
        a += (Math.random() - 0.5) * 1.2; }
    }
    fx.clearRect(0, 0, FW, FH); fx.drawImage(frostTex, 0, 0, FW, FH);
    for (let i = 0; i < 70; i++) snow.push({ x: Math.random() * W, y: Math.random() * H, r: Math.random() * 1.8 + 0.4, v: Math.random() * 0.4 + 0.1 });
    snow.length = Math.min(snow.length, 90);
  }
  function drawRink(t) {
    const hc = "#0ea5e9";
    ctx.clearRect(0, 0, W, H);
    const g = ctx.createRadialGradient(W * 0.6, H * 0.4, 10, W * 0.5, H * 0.5, Math.max(W, H) * 0.8);
    g.addColorStop(0, "#f7fbff"); g.addColorStop(0.6, "#dfeaf6"); g.addColorStop(1, "#b9c9dc"); ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    // rink lines
    const rx = W * 0.06, ry = H * 0.12, rw = W * 0.88, rh = H * 0.76, rr = rh * 0.32;
    ctx.save(); ctx.globalAlpha = 0.9; ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(150,170,195,.6)"; ctx.beginPath(); ctx.roundRect(rx, ry, rw, rh, rr); ctx.stroke();
    ctx.strokeStyle = "rgba(255,77,94,.7)"; ctx.lineWidth = 6; ctx.beginPath(); ctx.moveTo(W / 2, ry); ctx.lineTo(W / 2, ry + rh); ctx.stroke();
    ctx.strokeStyle = "rgba(77,163,255,.75)"; ctx.lineWidth = 7;
    [0.36, 0.64].forEach((k) => { ctx.beginPath(); ctx.moveTo(rx + rw * k, ry); ctx.lineTo(rx + rw * k, ry + rh); ctx.stroke(); });
    ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(W / 2, H / 2, rh * 0.2, 0, 6.29); ctx.stroke();
    ctx.strokeStyle = "rgba(255,77,94,.55)";
    [[0.2, 0.33], [0.2, 0.67], [0.8, 0.33], [0.8, 0.67]].forEach(([a, b]) => { ctx.beginPath(); ctx.arc(rx + rw * a, ry + rh * b, rh * 0.15, 0, 6.29); ctx.stroke(); });
    ctx.restore();
    // pulzující světla stadionu
    for (let i = 0; i < 5; i++) {
      const lx = W * (0.1 + i * 0.2), a = 0.10 + 0.06 * Math.sin(t / 900 + i);
      const lg = ctx.createRadialGradient(lx, H * 0.2, 0, lx, H * 0.2, H * 0.8); lg.addColorStop(0, `rgba(255,255,255,${a})`); lg.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = lg; ctx.fillRect(0, 0, W, H);
    }
    ctx.drawImage(trail, 0, 0, W, H);
    // sníh / ledová tříšť ve vzduchu
    ctx.fillStyle = "rgba(120,150,190,.35)";
    snow.forEach((p) => { p.y -= p.v; p.x += Math.sin((t / 1000) + p.r) * 0.2; if (p.y < -5) { p.y = H + 5; p.x = Math.random() * W; } ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.29); ctx.fill(); });
    // částice
    for (let i = parts.length - 1; i >= 0; i--) { const p = parts[i]; p.x += p.vx; p.y += p.vy; p.vx *= 0.94; p.vy *= 0.94; p.life -= 0.025;
      if (p.life <= 0) { parts.splice(i, 1); continue; } ctx.fillStyle = `rgba(160,185,215,${p.life})`; ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.29); ctx.fill(); }
    // puk
    const sp = Math.hypot(puck.vx, puck.vy);
    if (sp > 6) { ctx.strokeStyle = "rgba(47,123,255,.35)"; ctx.lineWidth = 10; ctx.lineCap = "round"; ctx.beginPath(); ctx.moveTo(puck.x, puck.y); ctx.lineTo(puck.x - puck.vx * 4, puck.y - puck.vy * 4); ctx.stroke(); }
    ctx.fillStyle = "rgba(0,0,0,.45)"; ctx.beginPath(); ctx.ellipse(puck.x + 4, puck.y + 9, 22, 9, 0, 0, 6.29); ctx.fill();
    ctx.fillStyle = "#0b0f17"; ctx.beginPath(); ctx.ellipse(puck.x, puck.y + 5, 20, 9, 0, 0, 6.29); ctx.fill(); ctx.fillRect(puck.x - 20, puck.y - 1, 40, 6);
    const pg = ctx.createLinearGradient(puck.x - 20, puck.y - 9, puck.x + 20, puck.y + 9); pg.addColorStop(0, "#3a4250"); pg.addColorStop(1, "#111722");
    ctx.fillStyle = pg; ctx.beginPath(); ctx.ellipse(puck.x, puck.y - 1, 20, 9, 0, 0, 6.29); ctx.fill();
  }
  function step() {
    const tx0 = mouse.active ? mouse.x : W * 0.62 + Math.cos(performance.now() / 1700) * W * 0.22;
    const ty0 = mouse.active ? mouse.y : H * 0.55 + Math.sin(performance.now() / 1300) * H * 0.18;
    const px = puck.x, py = puck.y;
    if (puck.free > 0) { puck.free--; } else { puck.vx += (tx0 - puck.x) * 0.012; puck.vy += (ty0 - puck.y) * 0.012; }
    puck.vx *= puck.free > 0 ? 0.985 : 0.88; puck.vy *= puck.free > 0 ? 0.985 : 0.88;
    puck.x += puck.vx; puck.y += puck.vy;
    const m = 24;
    if (puck.x < m || puck.x > W - m) { puck.vx *= -0.8; puck.x = Math.max(m, Math.min(W - m, puck.x)); burst(puck.x, puck.y, 10); }
    if (puck.y < m || puck.y > H - m) { puck.vy *= -0.8; puck.y = Math.max(m, Math.min(H - m, puck.y)); burst(puck.x, puck.y, 10); }
    // stopa bruslí
    tx.globalCompositeOperation = "destination-out"; tx.fillStyle = "rgba(0,0,0,.012)"; tx.fillRect(0, 0, W, H);
    tx.globalCompositeOperation = "source-over"; tx.strokeStyle = "rgba(120,145,175,.28)"; tx.lineWidth = 2; tx.lineCap = "round";
    const nx = -(puck.y - py), ny = puck.x - px, nl = Math.hypot(nx, ny) || 1;
    [-7, 7].forEach((o) => { tx.beginPath(); tx.moveTo(px + (nx / nl) * o, py + (ny / nl) * o); tx.lineTo(puck.x + (nx / nl) * o, puck.y + (ny / nl) * o); tx.stroke(); });
    if (Math.hypot(puck.vx, puck.vy) > 9 && Math.random() < 0.5) burst(puck.x, puck.y, 2);
  }
  function burst(x, y, n) { for (let i = 0; i < n; i++) { const a = Math.random() * 6.28, v = Math.random() * 5 + 1; parts.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, r: Math.random() * 2.2 + 0.6, life: 1 }); } }
  function melt() {
    if (getComputedStyle(fr).display === "none") return;
    // námraza pomalu znovu narůstá
    if ((melt.n = (melt.n || 0) + 1) % 5 === 0) { fx.globalCompositeOperation = "source-over"; fx.globalAlpha = 0.05; fx.drawImage(frostTex, 0, 0, FW, FH); fx.globalAlpha = 1; }
    if (!mouse.active) return;
    const r = sec.getBoundingClientRect(), c = cv.getBoundingClientRect();
    const x = mouse.x + (c.left - r.left), y = mouse.y + (c.top - r.top);
    fx.globalCompositeOperation = "destination-out";
    const g = fx.createRadialGradient(x, y, 0, x, y, 110); g.addColorStop(0, "rgba(0,0,0,.35)"); g.addColorStop(1, "rgba(0,0,0,0)");
    fx.fillStyle = g; fx.beginPath(); fx.arc(x, y, 110, 0, 6.29); fx.fill();
    fx.globalCompositeOperation = "source-over";
  }
  function loop(t) {
    if (!run) return;
    if (visible && !document.hidden) { step(); drawRink(t); melt(); }
    raf = requestAnimationFrame(loop);
  }
  const toLocal = (e) => { const r = cv.getBoundingClientRect(); const p = e.touches ? e.touches[0] : e; mouse.x = p.clientX - r.left; mouse.y = p.clientY - r.top; mouse.active = true; };
  const onMove = (e) => toLocal(e);
  const onLeave = () => (mouse.active = false);
  const onClick = (e) => {
    if (e.target.closest("button, a, select, input")) return;
    toLocal(e);
    const a = Math.atan2(mouse.y - puck.y, mouse.x - puck.x) + Math.PI + (Math.random() - 0.5) * 0.6;
    puck.vx = Math.cos(a) * 34; puck.vy = Math.sin(a) * 34; puck.free = 45; burst(puck.x, puck.y, 26);
  };
  sec.addEventListener("mousemove", onMove); sec.addEventListener("touchmove", onMove, { passive: true });
  sec.addEventListener("mouseleave", onLeave); sec.addEventListener("touchend", onLeave);
  sec.addEventListener("click", onClick);
  const ro = new ResizeObserver(() => size()); ro.observe(sec);
  const io = new IntersectionObserver((e) => (visible = e[0].isIntersecting)); io.observe(sec);
  size();
  if (reduce) { step(); drawRink(0); } else raf = requestAnimationFrame(loop);
  HERO = { stop() { run = false; cancelAnimationFrame(raf); ro.disconnect(); io.disconnect(); } };
}
// vlastní kurzor + magnetická tlačítka
(function () {
  if (matchMedia("(hover: none)").matches) return;
  const c = $("cur"); let x = 0, y = 0, cx = 0, cy = 0;
  addEventListener("mousemove", (e) => { x = e.clientX; y = e.clientY; c.classList.add("on");
    const hot = e.target.closest && e.target.closest("button, a, .mcard, .pcard, .lg, select, .lrow");
    c.classList.toggle("big", !!hot);
    const mag = e.target.closest && e.target.closest(".btn, .nav a");
    document.querySelectorAll(".magnet").forEach((m) => { if (m !== mag) { m.style.transform = ""; m.classList.remove("magnet"); } });
    if (mag) { const r = mag.getBoundingClientRect(); mag.classList.add("magnet");
      mag.style.transform = `translate(${(e.clientX - r.left - r.width / 2) * 0.25}px, ${(e.clientY - r.top - r.height / 2) * 0.35}px)`; }
  });
  addEventListener("mouseout", (e) => { if (!e.relatedTarget) c.classList.remove("on"); });
  const z = () => parseFloat(getComputedStyle(document.body).zoom) || 1;
  (function f() { cx += (x - cx) * 0.22; cy += (y - cy) * 0.22; const k = z(); c.style.transform = `translate(${cx / k}px, ${cy / k}px)`; requestAnimationFrame(f); })();
})();

let revealObs = null;
function reveal(root) {
  const els = root.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) { els.forEach((e) => e.classList.add("in")); return; }
  revealObs = revealObs || new IntersectionObserver((ents) => ents.forEach((en) => {
    if (en.isIntersecting) { en.target.classList.add("in"); revealObs.unobserve(en.target); }
  }), { threshold: 0.08 });
  els.forEach((e, i) => { e.style.transitionDelay = `${Math.min(i % 8, 7) * 60}ms`; revealObs.observe(e); });
}

// ---------- ikony (SVG místo emoji) ----------
const ICP = {
  spark: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 3v4M17 5h4"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5"/>',
  bars: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
  bot: '<rect x="4" y="8" width="16" height="12" rx="3"/><path d="M12 4v4M9 13h.01M15 13h.01M9 17h6"/>',
  ticket: '<path d="M3 8a2 2 0 0 0 0 4v4h18v-4a2 2 0 0 1 0-4V4H3z" transform="translate(0 2)"/><path d="M14 6v12" stroke-dasharray="2 2"/>',
  stick: '<path d="M14 3l-6 15a2 2 0 0 0 2 2h6"/><ellipse cx="19" cy="19" rx="3" ry="1.5"/>',
  whistle: '<circle cx="12" cy="12" r="9"/><path d="M5.6 5.6l12.8 12.8"/>',
  goal: '<path d="M3 20V7h18v13"/><path d="M3 7l4 4h10l4-4M7 11v9M17 11v9M7 15h10"/>',
  glove: '<path d="M7 11V5a2 2 0 0 1 4 0v5M11 9V4a2 2 0 0 1 4 0v6M15 9V6a2 2 0 0 1 4 0v8a7 7 0 0 1-7 7h-1a7 7 0 0 1-6-3l-2-4a2 2 0 0 1 3-2l2 2"/>',
  coins: '<ellipse cx="9" cy="7" rx="6" ry="3"/><path d="M3 7v5c0 1.7 2.7 3 6 3s6-1.3 6-3V7"/><path d="M15 11c3.3 0 6 1.3 6 3v4c0 1.7-2.7 3-6 3s-6-1.3-6-3"/>',
  calc: '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M8 7h8M8 12h.01M12 12h.01M16 12h.01M8 16h.01M12 16h.01M16 16h.01"/>',
  brain: '<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 6 1V4a3 3 0 0 0-3 0zM15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-6 1"/>',
  swords: '<path d="M14.5 17.5L3 6V3h3l11.5 11.5M13 19l6-6M16 16l4 4M9.5 6.5L13 3h3v3l-3.5 3.5M5 14l-2 2 3 3 2-2"/>',
  chart: '<path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 6-6"/>',
  calendar: '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/>',
  trophy: '<path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0z"/><path d="M17 5h3v2a4 4 0 0 1-4 4M7 5H4v2a4 4 0 0 0 4 4"/>',
  book: '<path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2zM4 19V5"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5h.01"/>',
  flame: '<path d="M12 3c1 4 5 5 5 10a5 5 0 0 1-10 0c0-3 2-4 2-6 2 1 3 3 3 3s1-3 0-7z"/>',
  siren: '<path d="M6 18v-6a6 6 0 0 1 12 0v6M4 21h16M12 3v2M4.2 7.2l1.4 1.4M19.8 7.2l-1.4 1.4"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  hit: '<path d="M13 2L4 14h7l-1 8 9-12h-7z"/>',
  shield: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/>',
  arrow: '<path d="M7 17L17 7M9 7h8v8"/>',
  dot: '<circle cx="12" cy="12" r="4"/>',
  feed: '<path d="M4 6h16M4 12h16M4 18h10"/>',
  phone: '<rect x="7" y="2" width="10" height="20" rx="2"/><path d="M11 18h2"/>',
  star: '<path d="M12 3l2.8 5.7 6.2.9-4.5 4.4 1 6.2-5.5-2.9-5.5 2.9 1-6.2L3 9.6l6.2-.9z"/>',
  bolt: '<path d="M13 2L4 14h7l-1 8 9-12h-7z"/>',
  id: '<rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="9" cy="11" r="2.5"/><path d="M5.5 16c.6-1.6 2-2.5 3.5-2.5s2.9.9 3.5 2.5M14 9h4M14 12h4M14 15h2"/>',
  wave: '<path d="M7 11V6a1.5 1.5 0 0 1 3 0v5M10 10V4.5a1.5 1.5 0 0 1 3 0V10M13 10V5.5a1.5 1.5 0 0 1 3 0V12M16 11a1.5 1.5 0 0 1 3 0v3a7 7 0 0 1-12.6 4.2L4 14.8a1.5 1.5 0 0 1 2.4-1.8L7 14"/>',
  eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  play: '<path d="M7 4l13 8-13 8z"/>',
  medic: '<rect x="3" y="3" width="18" height="18" rx="4"/><path d="M12 7v10M7 12h10"/>',
  trash: '<path d="M4 7h16M9 7V4h6v3M6 7l1 14h10l1-14"/>',
  check: '<path d="M5 12l5 5 9-10"/>', cross: '<path d="M6 6l12 12M18 6L6 18"/>',
};
const ic = (n, cls = "") => `<span class="ic ${cls}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICP[n] || ICP.dot}</svg></span>`;
const EMOJI = { "🔮": "spark", "🎯": "target", "📊": "bars", "🤖": "bot", "🎫": "ticket", "🏒": "stick", "⛔": "whistle", "🥅": "goal", "🧤": "glove",
  "💰": "coins", "🧮": "calc", "🧠": "brain", "⚔️": "swords", "📈": "chart", "📅": "calendar", "🏆": "trophy", "📖": "book", "👋": "wave", "👀": "eye",
  "🔥": "flame", "🚨": "siren", "🕒": "clock", "💥": "hit", "🛡️": "shield", "↗️": "arrow", "⚪": "dot", "⚡": "bolt", "📰": "feed", "📱": "phone",
  "⭐": "star", "🪪": "id", "🗑": "trash", "🧊": "dot", "🔲": "dot", "🫳": "dot", "❌": "cross", "✅": "check", "⏹": "dot", "⏳": "clock", "🩹": "medic" };
const EMOJI_RE = new RegExp(Object.keys(EMOJI).map((e) => e.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"), "g");
function iconify(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, { acceptNode: (n) => (EMOJI_RE.test(n.nodeValue) ? (EMOJI_RE.lastIndex = 0, 1) : 2) });
  const nodes = []; let n; while ((n = walker.nextNode())) nodes.push(n);
  nodes.forEach((t) => {
    if (t.parentNode.closest && t.parentNode.closest("svg, option, select, title, textarea, input")) return;
    const span = document.createElement("span");
    span.innerHTML = esc(t.nodeValue).replace(EMOJI_RE, (m) => {
      const cls = m === "✅" ? "okc" : m === "❌" ? "noc" : m === "🔥" ? "fl" : "";
      return ic(EMOJI[m], cls);
    });
    t.replaceWith(...span.childNodes);
  });
}
new MutationObserver((ms) => { for (const m of ms) m.addedNodes.forEach((n) => { if (n.nodeType === 1) iconify(n); else if (n.nodeType === 3 && n.parentNode) iconify(n.parentNode); }); })
  .observe(document.body, { childList: true, subtree: true });
document.addEventListener("DOMContentLoaded", () => iconify(document.body));

const SKYLINE = `<svg class="skyline" viewBox="0 0 1200 120" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
  <defs><linearGradient id="glass" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#bfe3ff" stop-opacity=".10"/><stop offset="1" stop-color="#bfe3ff" stop-opacity=".02"/></linearGradient></defs>
  <rect x="0" y="0" width="1200" height="78" fill="url(#glass)"/>
  ${Array.from({ length: 13 }, (_, i) => `<line x1="${i * 100}" y1="0" x2="${i * 100}" y2="78" stroke="rgba(255,255,255,.18)" stroke-width="2"/>`).join("")}
  <rect x="0" y="78" width="1200" height="34" fill="#f4f7fb"/>
  <rect x="0" y="108" width="1200" height="12" fill="#ffd200"/>
  ${Array.from({ length: 8 }, (_, i) => `<rect x="${i * 150 + 20}" y="84" width="110" height="20" rx="2" fill="${["#0d47c9", "#05070b", "#d6001c", "#0d47c9", "#05070b", "#d6001c", "#0d47c9", "#05070b"][i]}" opacity=".85"/><text x="${i * 150 + 75}" y="99" font-family="Anton" font-size="13" fill="#fff" text-anchor="middle" letter-spacing="2">${["NHL", "PREDIKCE", "LIVE", "ICE", "NHL", "STATS", "LIVE", "ICE"][i]}</text>`).join("")}
</svg>`;

function sortTable(el, cols, rows, sort, onSort, rowClass) {
  const sc = cols.find((c) => c.k === sort.key) || cols[1];
  const val = (c, p) => (c.v ? c.v(p) : p[c.k]);
  rows = [...rows].sort((a, b) => {
    const x = val(sc, a), y = val(sc, b);
    if (typeof x === "string") return sort.dir * x.localeCompare(y, "cs");
    return sort.dir * ((x ?? -1e9) - (y ?? -1e9));
  });
  el.innerHTML = "<thead><tr>" + cols.map((c) =>
    `<th class="${c.cls || ""} ${c.k === sort.key ? "sorted" : ""}" data-k="${c.k}">${c.t}${c.k === sort.key ? (sort.dir < 0 ? " ▾" : " ▴") : ""}</th>`).join("") +
    "</tr></thead><tbody>" + (rows.length ? rows.map((p, i) => `<tr class="${rowClass ? rowClass(p) : ""}">` + cols.map((c) => {
      if (c.k === "_rank") return `<td class="rank">${i + 1}</td>`;
      const v = val(c, p);
      return `<td class="${c.cls || ""}">${c.f ? c.f(v, p) : (v ?? "–")}</td>`;
    }).join("") + "</tr>").join("") : `<tr><td colspan="${cols.length}" class="loader">Žádná data</td></tr>`) + "</tbody>";
  el.querySelectorAll("th").forEach((th) => {
    const c = cols.find((c) => c.k === th.dataset.k);
    if (!c || c.nosort) return;
    th.onclick = () => {
      if (sort.key === c.k) sort.dir *= -1;
      else { sort.key = c.k; sort.dir = (c.cls || "").includes("left") ? 1 : -1; }
      onSort();
    };
  });
}

// jednoduchý SVG graf (sloupce + čáry)
function chart({ labels, bars = [], lines = [], height = 190, avg = null, max = null }) {
  const W = 560, H = height, pl = 30, pr = 10, pt = 14, pb = 30;
  const all = [...bars.flatMap((b) => b.values), ...lines.flatMap((l) => l.values), avg ?? 0].filter((v) => v != null);
  const mx = max ?? Math.max(1, ...all) * 1.12;
  const n = labels.length || 1, cw = (W - pl - pr) / n;
  const y = (v) => pt + (H - pt - pb) * (1 - v / mx);
  let s = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" font-family="Inter,sans-serif">`;
  for (let i = 0; i <= 4; i++) {
    const v = (mx / 4) * i, yy = y(v);
    s += `<line x1="${pl}" x2="${W - pr}" y1="${yy}" y2="${yy}" stroke="rgba(148,163,184,.12)"/>` +
      `<text x="${pl - 6}" y="${yy + 4}" fill="#8796ae" font-size="10" text-anchor="end">${Math.round(v)}</text>`;
  }
  const bw = (cw * 0.72) / Math.max(1, bars.length);
  bars.forEach((b, bi) => b.values.forEach((v, i) => {
    if (v == null) return;
    const x = pl + i * cw + cw * 0.14 + bi * bw;
    s += `<rect x="${x}" y="${y(v)}" width="${bw - 2}" height="${Math.max(0, H - pb - y(v))}" rx="3" fill="${b.color}" opacity=".85"><title>${b.name}: ${v}</title></rect>`;
  }));
  if (avg != null) s += `<line x1="${pl}" x2="${W - pr}" y1="${y(avg)}" y2="${y(avg)}" stroke="#fbbf24" stroke-dasharray="5 4" opacity=".7"/>`;
  lines.forEach((l) => {
    const pts = l.values.map((v, i) => (v == null ? null : [pl + i * cw + cw / 2, y(v)])).filter(Boolean);
    s += `<polyline points="${pts.map((p) => p.join(",")).join(" ")}" fill="none" stroke="${l.color}" stroke-width="2.5" stroke-linejoin="round"/>`;
    pts.forEach((p, i) => (s += `<circle cx="${p[0]}" cy="${p[1]}" r="3.5" fill="${l.color}"><title>${l.name}: ${l.values[i]}</title></circle>`));
  });
  labels.forEach((lb, i) => (s += `<text x="${pl + i * cw + cw / 2}" y="${H - 10}" fill="#8796ae" font-size="10" text-anchor="middle">${esc(lb)}</text>`));
  return s + "</svg>";
}

// ===================================================================
//  Router
// ===================================================================
let OV = null;          // přehled
let GAME = null;        // detail zápasu
let pollTimer = null;
function route() {
  const h = location.hash;
  const game = h.startsWith("#zapas-") ? +h.slice(7) : null;
  const stats = h === "#statistiky";
  const live = h === "#live" || h.startsWith("#live-");
  const acc = h === "#presnost";
  const card = h.startsWith("#karta");
  const tk = h === "#tikety";
  const over = !game && !stats && !live && !acc && !card && !tk;
  $("view-tk").classList.toggle("hidden", !tk);
  $("nav-tk").classList.toggle("on", tk);
  $("view-card").classList.toggle("hidden", !card);
  $("nav-card").classList.toggle("on", card);
  $("view-over").classList.toggle("hidden", !over);
  $("view-game").classList.toggle("hidden", !game);
  $("view-stats").classList.toggle("hidden", !stats);
  $("view-live").classList.toggle("hidden", !live);
  $("view-acc").classList.toggle("hidden", !acc);
  $("nav-over").classList.toggle("on", over || !!game);
  $("nav-stats").classList.toggle("on", stats);
  $("nav-live").classList.toggle("on", live);
  $("nav-acc").classList.toggle("on", acc);
  CHAT.gameId = game;
  updateChatCtx();
  if (!live) stopLive();
  if (!over && HERO) { HERO.stop(); HERO = null; }
  if (stats) initStats();
  else if (game) openGame(game);
  else if (live) openLive(h.startsWith("#live-") ? +h.slice(6) : null);
  else if (acc) loadAcc();
  else if (card) openCard(h.slice(7));
  else if (tk) openTickets();
  else loadOverview();
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);

// ===================================================================
//  Přehled všech zápasů
// ===================================================================
let ovDay = null;
async function loadOverview() {
  const el = $("view-over");
  if (OV && Date.now() - OV._t < 120000) { renderOverview(); return; }
  if (!OV) el.innerHTML = `<div class="loader"><div class="spinner"></div>Načítám zápasy…</div>`;
  try {
    const d = await getJSON("/model/overview");
    if (d.loading) { renderLoading(el, d); clearTimeout(pollTimer); pollTimer = setTimeout(loadOverview, 2000); return; }
    OV = { ...d, _t: Date.now() };
    CHAT.ai = d.ai;
    renderOverview();
  } catch (e) {
    el.innerHTML = `<div class="loader err">${esc(e.message)}<br><br><button class="btn" onclick="loadOverview()">Zkusit znovu</button></div>`;
  }
}
function renderLoading(el, s) {
  const pct = s.total ? Math.round((s.done / s.total) * 100) : 5;
  el.innerHTML = `<div class="loader"><div class="spinner"></div><b>Model se připravuje</b><br>${esc(s.phase || "")}
    ${s.total ? ` (${s.done} / ${s.total})` : ""}<div class="progress"><i style="width:${pct}%"></i></div>
    <p class="sub" style="max-width:460px;margin:16px auto 0">Stahuji všechny odehrané zápasy sezóny, ze kterých model počítá.
    Trvá to zhruba minutu – jen po probuzení serveru, pak už jsou data hned.</p>
    ${s.error ? `<p class="err">${esc(s.error)}</p>` : ""}</div>`;
}
function renderOverview() {
  const el = $("view-over");
  const games = OV.games.filter((g) => g.win);
  const days = [...new Map(games.map((g) => [dayKey(g.start), g.start])).entries()];
  if (!ovDay || !days.find((d) => d[0] === ovDay)) ovDay = days.length ? days[0][0] : null;
  const dayGames = games.filter((g) => dayKey(g.start) === ovDay);
  const all = dayGames.flatMap((g) => g.top.map((p) => ({ ...p, g })));
  const byPoint = [...all].sort((a, b) => b.pPoint - a.pPoint).slice(0, 10);
  const byGoal = [...all].sort((a, b) => b.pGoal - a.pGoal).slice(0, 10);
  const byShots = [...all].sort((a, b) => b.expShots - a.expShots).slice(0, 10);
  const logo = (g, t) => (t === g.home ? g.homeLogo : g.awayLogo);
  const lrow = (p, i, val) => `<div class="lrow" data-go="${p.g.id}"><span class="rk">${i + 1}</span><img src="${esc(logo(p.g, p.team))}" alt="">
      <span class="nm"><span class="pname-link" data-player="${p.id}:${p.team}">${esc(p.name)}</span>${p.streak >= 3 ? ` <span class="pill hot">🔥${p.streak}</span>` : ""}<small>${esc(p.g.home)} – ${esc(p.g.away)} · ${fmtTime(p.g.start)}</small></span><span class="val">${val}</span></div>`;
  const valueBadge = (g) => {
    const v = g.value || {};
    const best = Object.entries(v).sort((a, b) => b[1] - a[1])[0];
    return best && best[1] > 0.03 ? `<span class="pill good">hodnota ${best[0] === "home" ? g.home : g.away} +${Math.round(best[1] * 100)} %</span>` : "";
  };
  const sum = (f) => dayGames.reduce((a, g) => a + f(g), 0);
  const dayIso = days.find((d) => d[0] === ovDay)?.[1];
  const best = byPoint[0];
  const mq = byPoint.map((p) => `<span>${esc(p.name)} <em>${pc(p.pPoint)}</em></span>`).join("");
  const seen = new Set();
  const cards = [...all].sort((a, b) => b.pPoint - a.pPoint).filter((p) => !seen.has(p.id) && seen.add(p.id)).slice(0, 8);
  el.innerHTML = `
    <section class="cine hero2" id="cine">
      <div class="sky"></div><div class="sun"></div>${SKYLINE}
      <div class="stage3d"><canvas id="icecv"></canvas></div><canvas id="frost"></canvas>
      <div class="backword" id="backword">NHL</div>
      <div><span class="badge-gta">NHL · Předpovědi · ${dayIso ? fmtDay(dayIso) : ""}</span></div>
      <h1 class="display"><span class="ln"><i>Hokejová noc</i></span><span class="ln"><i>v číslech</i></span></h1>
      <p class="lead">Model prošel ${(OV.status?.total || 0)} odehraných zápasů sezóny a spočítal, kdo dá bod, kolik padne střel, gólů a vyloučení
        ${best ? `– největší šanci na bod má dnes <b style="color:#fff">${esc(best.name)}</b> (${pc(best.pPoint)}).` : "."}</p>
      <div class="counters">
        <div><b data-count="${dayGames.length}">0</b><small>zápasů</small></div>
        <div><b data-count="${Math.round(sum((g) => g.goals.total))}">0</b><small>očekávaných gólů</small></div>
        <div><b data-count="${Math.round(sum((g) => g.shots.total))}">0</b><small>očekávaných střel</small></div>
        <div><b data-count="${Math.round(sum((g) => g.pens.total))}">0</b><small>očekávaných vyloučení</small></div>
      </div>
      <p class="help" style="position:relative">Čísla nahoře jsou součty za všechny zápasy vybraného dne. Den přepneš tlačítky níže.</p>
      <div class="hint2">🏒 Hýbej myší (na mobilu prstem) – <b>puk jede za tebou</b> a led pod ním taje. <b>Klikni</b> a puk vystřelí.</div>
      <div class="seg daytabs" id="days">${days.map(([k, iso]) => `<button data-v="${k}" class="${k === ovDay ? "on" : ""}">${fmtDay(iso)}</button>`).join("")}</div>
    </section>
    ${mq ? `<div class="marquee"><div class="track">${mq}${mq}</div></div>` : ""}
    <div class="helpbox"><h3>👋 Jak tuhle stránku číst</h3><div class="help" style="margin:0">
      <ul>
        <li><b>Počítač (model) prošel všechny odehrané zápasy této sezóny</b> a z nich odhaduje, co se stane v zápasech, které se teprve hrají.</li>
        <li><b>Procenta = šance.</b> Třeba „70 %“ znamená, že kdyby se ten samý zápas hrál 10×, stalo by se to zhruba v 7 z nich. Nikdy to není jistota.</li>
        <li><b>Čísla jako „29.8 střel“ jsou průměrné očekávání.</b> Přesně 29.8 střel samozřejmě nebude – znamená to, že nejspíš padne kolem 28–32.</li>
        <li><b>Nahoře vyber den</b>, pod tím uvidíš nejlepší tipy toho dne a pak všechny zápasy. <b>Na zápas klikni</b> a otevře se podrobný rozbor.</li>
        <li><b>🤖 Vpravo dole je AI</b> – můžeš se jí česky zeptat na cokoli k zápasům („kdo dnes dá bod?“).</li>
        <li><b>Běžící pás</b> pod nadpisem = hráči s největší šancí na bod dnes.</li>
        <li><b>🎫 Vlevo dole je tvůj tiket</b> – v detailu zápasu přidáváš sázky tlačítkem „+ tiket“ a stránka spočítá celkový kurz a šanci.</li>
        <li>V detailu zápasu najdeš i <b>simulaci zápasu</b>, <b>hřiště se střelami</b> a <b>porovnání hráčů</b>.</li>
        <li><b>🔴 Live</b> nahoře = zápasy, které se právě hrají, na 3D hřišti s pukem. <b>🎯 Přesnost</b> = jak se modelu dařilo v odehraných zápasech.</li>
        <li><b>🎫 Tikety</b> nahoře = nahraj screenshot tiketu z Tipsportu a AI s modelem ho vyhodnotí, opraví a okomentuje.</li>
        <li><b>🪪 NHL ID</b> nahoře = vytvoř si „občanku“ libovolného hráče ve stylu GTA. Na jméno hráče můžeš kdekoli kliknout a otevře se jeho <b>analýza</b>.</li>
        <li><b>📱 Na mobilu</b> si web můžeš dát na plochu jako aplikaci: v iPhonu Safari → tlačítko Sdílet → <b>Přidat na plochu</b>; v Androidu Chrome → menu ⋮ → <b>Přidat na plochu / Nainstalovat aplikaci</b>.</li>
      </ul></div></div>
    <div class="sechead"><span class="no">01</span><h2>Hráči dne</h2><span class="hint">největší šance na kanadský bod</span></div>
    <p class="help" style="margin-top:-6px">Každá karta = jeden hráč. <b>Velké žluté číslo</b> = šance, že dá gól nebo asistenci. Dole: šance na gól, kolik střel na branku čekáme
      a posledních 5 zápasů (žlutý čtvereček = v tom zápase bodoval, nejnovější vpravo). Velké průhledné číslo v rohu = číslo dresu. Kliknutím otevřeš jeho zápas.</p>
    <div class="pcards">${cards.map((p) => playerCard(p, logo(p.g, p.team), `data-go="${p.g.id}"`)).join("")}</div>
    <div class="sechead"><span class="no">02</span><h2>Žebříčky</h2><span class="hint">napříč všemi zápasy dne</span></div>
    <p class="help" style="margin-top:-6px">Tady jsou hráči ze <b>všech zápasů vybraného dne</b> seřazení podle toho, jak velkou šanci jim model dává. Kliknutím na hráče otevřeš jeho zápas. 🔥 a číslo = kolik zápasů v řadě už bodoval.</p>
    <div class="leaders">
      <div class="card leader reveal"><h3>🎯 Šance na kanadský bod</h3><p class="help" style="margin:-6px 0 8px">Šance, že hráč v zápase <b>dá gól nebo na gól přihraje</b> (asistence). Stačí jedno z toho.</p>${byPoint.map((p, i) => lrow(p, i, pc(p.pPoint))).join("")}</div>
      <div class="card leader"><h3>🥅 Šance na gól</h3><p class="help" style="margin:-6px 0 8px">Šance, že hráč <b>sám vstřelí aspoň jeden gól</b>. Je vždy menší než šance na bod.</p>${byGoal.map((p, i) => lrow(p, i, pc(p.pGoal))).join("")}</div>
      <div class="card leader"><h3>🏒 Nejvíc střel na branku</h3><p class="help" style="margin:-6px 0 8px">Kolik <b>střel na branku</b> hráč v průměru vystřelí (počítají se jen střely, které by šly do branky nebo je chytil brankář).</p>${byShots.map((p, i) => lrow(p, i, num(p.expShots, 1))).join("")}</div>
    </div>
    <div class="sechead"><span class="no">03</span><h2>Zápasy</h2><span class="hint">klikni na zápas pro detailní rozbor</span></div>
    <p class="help" style="margin-top:-6px">Každá karta = jeden zápas. <b>Vlevo je domácí tým, vpravo hosté.</b> Velká čísla uprostřed = kolik gólů model čeká.
      Modro-fialový pruh = šance na výhru (modrá domácí, fialová hosté). <b>Tipsport</b> = kurzy sázkovky na vítěze. Zelený štítek <b>„hodnota“</b> = kurz u sázkovky je vyšší,
      než by podle modelu měl být (sázka se podle modelu vyplatí). <b>Σ</b> = součet za oba týmy. 🎯 = 3 hráči s největší šancí na bod.</p>
    <div class="matches">${dayGames.map((g) => {
      const wh = Math.round(g.win.home * 100), wa = 100 - wh;
      return `<div class="card mcard reveal tilt" data-go="${g.id}"><span class="glare"></span><img class="wmL" src="${esc(g.homeLogo)}" alt=""><img class="wmR" src="${esc(g.awayLogo)}" alt="">
        <div class="mtop"><span>${["LIVE", "CRIT"].includes(g.state) ? '<span class="pill live">● ŽIVĚ</span>' : "🕒 " + fmtTime(g.start) + ` · <span class="countdown" data-start="${g.start}"></span>`}</span><span>${esc(g.venue || "")}</span></div>
        <div class="mteams">
          <div class="mteam"><img src="${esc(g.homeLogo)}" alt=""><b>${esc(g.home)}</b><small>${esc(g.homeName)}</small></div>
          <div class="mmid"><div class="xg">${g.goals.home.toFixed(1)} : ${g.goals.away.toFixed(1)}</div><small>očekávané góly</small></div>
          <div class="mteam"><img src="${esc(g.awayLogo)}" alt=""><b>${esc(g.away)}</b><small>${esc(g.awayName)}</small></div>
        </div>
        <div class="wbar"><i class="h" style="width:${wh}%"></i><i class="a" style="width:${wa}%"></i></div>
        <div class="wlab"><span class="h">${wh} %</span><span class="muted" style="font-weight:600">šance na výhru</span><span class="a">${wa} %</span></div>
        <div class="odds"><span>Tipsport ${g.odds?.home ? `<span class="odd">${g.odds.home.toFixed(2)}</span> <span class="odd">${g.odds.away.toFixed(2)}</span>` : "– kurzy zatím nejsou"}</span>${valueBadge(g)}</div>
        <div class="chips">
          <div class="chip" title="Očekávané střely na branku: domácí : hosté. Σ = oba týmy dohromady"><small>Střely na branku</small><b>${g.shots.home} : ${g.shots.away}</b><em>Σ ${g.shots.total}</em></div>
          <div class="chip" title="Očekávaný počet trestů (vyloučení): domácí : hosté"><small>Vyloučení</small><b>${g.pens.home.toFixed(1)} : ${g.pens.away.toFixed(1)}</b><em>Σ ${g.pens.total.toFixed(1)}</em></div>
          <div class="chip" title="Očekávané góly: domácí : hosté"><small>Góly</small><b>${g.goals.home.toFixed(1)} : ${g.goals.away.toFixed(1)}</b><em>Σ ${g.goals.total.toFixed(1)}</em></div>
        </div>
        <div class="tops">${g.top.slice(0, 3).map((p) => `<div><span>🎯 ${esc(p.name)} <span class="muted">${p.team}</span></span><span class="p">${pc(p.pPoint)}</span></div>`).join("")}</div>
      </div>`;
    }).join("")}</div>

    <details class="gloss card" open><summary>📖 Slovníček – co znamenají pojmy na stránce</summary><dl>
      <dt>Kanadský bod</dt><dd>Hráč dostane bod za gól i za asistenci (přihrávku na gól). „Šance na bod“ = šance, že udělá aspoň jedno z toho.</dd>
      <dt>Šance (%)</dt><dd>Jak moc je něco pravděpodobné. 50 % = jako hod mincí, 80 % = stane se to v 8 z 10 případů. Ani 90 % není jistota.</dd>
      <dt>Očekávané (např. 29.8 střel)</dt><dd>Průměr, který model čeká. Skutečné číslo bude kolem něj – někdy víc, někdy míň.</dd>
      <dt>Střely na branku</dt><dd>Střely, které by šly do branky (gól nebo zákrok brankáře). Střely vedle nebo zblokované se nepočítají.</dd>
      <dt>Vyloučení</dt><dd>Počet trestů (hráč jde na trestnou lavici). Nepočítají se osobní tresty na 10 minut.</dd>
      <dt>Přes / pod (např. přes 5.5)</dt><dd>Sázka na to, jestli padne víc nebo míň než dané číslo. „Přes 5.5 gólu“ = padne 6 a víc gólů.</dd>
      <dt>Kurz</dt><dd>Kolikrát se ti vrátí vsazená částka, když sázka vyjde. Kurz 2.00 = vsadíš 100 Kč, dostaneš 200 Kč.</dd>
      <dt>Férový kurz</dt><dd>Kurz, který by „spravedlivě“ odpovídal šanci podle modelu (1 ÷ šance). Šance 50 % → férový kurz 2.00.</dd>
      <dt>Hodnota (EV)</dt><dd>Jestli se sázka podle modelu vyplatí. Když je kurz sázkovky vyšší než férový kurz, vychází kladné číslo (zelené) = sázka má hodnotu. Záporné (červené) = nevyplatí se.</dd>
      <dt>Vč. prodloužení</dt><dd>Počítá se, kdo vyhraje celý zápas, i kdyby to bylo až v prodloužení nebo nájezdech.</dd>
      <dt>Základní hrací doba</dt><dd>Jen 60 minut bez prodloužení. Tam může být i remíza.</dd>
      <dt>Forma (V / P / PP)</dt><dd>Výsledky posledních zápasů: V = výhra, P = prohra, PP = prohra v prodloužení nebo nájezdech.</dd>
      <dt>Doma / venku</dt><dd>Týmy často hrají doma jinak než venku, proto to model počítá zvlášť.</dd>
      <dt>2. den po sobě</dt><dd>Tým hrál i včera – bývá unavenější a často chytá náhradní brankář. Model mu trochu ubere.</dd>
      <dt>%Z (úspěšnost zásahů)</dt><dd>Kolik procent střel brankář chytí. 0.910 = 91 %. Víc je líp.</dd>
      <dt>GAA</dt><dd>Kolik gólů brankář průměrně dostane za zápas. Míň je líp.</dd>
      <dt>TOI</dt><dd>Průměrný čas hráče na ledě za zápas (minuty:sekundy). Víc času = víc šancí bodovat.</dd>
      <dt>B/Z</dt><dd>Body na zápas – průměr kanadských bodů za jeden zápas.</dd>
      <dt>Σ</dt><dd>Součet za oba týmy dohromady.</dd>
    </dl></details>
    <div class="disc">Předpovědi počítá statistický model ze všech odehraných zápasů sezóny (novější mají větší váhu, zvlášť doma a venku, proti obraně soupeře).
      Kurzy na vítěze (včetně prodloužení) jsou z Tipsportu přes oficiální NHL API. Data se obnovují každých 15 minut.
      Je to odhad, ne jistota – model neví o čerstvých zraněních a sestavách. Sázej jen to, co si můžeš dovolit prohrát.</div>`;
  segment($("days"), (v) => { ovDay = v; renderOverview(); });
  reveal(el); interactive(el); initHero2();
  el.querySelectorAll("[data-go]").forEach((c) => (c.onclick = () => (location.hash = "zapas-" + c.dataset.go)));
}

// ===================================================================
//  Detail zápasu
// ===================================================================
const GS = { teamF: "all", showAll: false, sort: { key: "pPoint", dir: -1 }, mkt: "win" };
async function openGame(id) {
  const el = $("view-game");
  if (!GAME || GAME.id !== id) el.innerHTML = `<div class="loader"><div class="spinner"></div>Počítám předpověď…</div>`;
  try {
    const d = await getJSON("/model/predict?id=" + id);
    if (d.loading) { renderLoading(el, d); clearTimeout(pollTimer); pollTimer = setTimeout(() => openGame(id), 2000); return; }
    GAME = d;
    GS.teamF = "all"; GS.showAll = false;
    renderGame();
  } catch (e) {
    el.innerHTML = `<button class="btn back" onclick="location.hash=''">← Zpět na zápasy</button><div class="loader err">${esc(e.message)}</div>`;
  }
}
function oddKey(m) { return `odd:${GAME.id}:${m}`; }
function marketRows() {
  const d = GAME, H = d.home, A = d.away;
  const rows = [];
  const ou = (grp, label, lines) => lines.forEach((l) => {
    rows.push({ grp, id: `${grp}:${label}:o${l.line}`, name: `${label} – přes ${l.line}`, p: l.over });
    rows.push({ grp, id: `${grp}:${label}:u${l.line}`, name: `${label} – pod ${l.line}`, p: 1 - l.over });
  });
  rows.push({ grp: "win", id: "win:h", name: `Vítěz vč. prodloužení – ${d.homeName}`, p: d.win.home, auto: d.odds?.home });
  rows.push({ grp: "win", id: "win:a", name: `Vítěz vč. prodloužení – ${d.awayName}`, p: d.win.away, auto: d.odds?.away });
  rows.push({ grp: "win", id: "win:1", name: `Výhra v základní hrací době – ${H}`, p: d.win.homeReg });
  rows.push({ grp: "win", id: "win:x", name: `Remíza po 60 minutách (prodloužení)`, p: d.win.ot });
  rows.push({ grp: "win", id: "win:2", name: `Výhra v základní hrací době – ${A}`, p: d.win.awayReg });
  ou("goals", "Góly celkem", d.goals.totalLines);
  ou("goals", `Góly ${H}`, d.goals.homeLines);
  ou("goals", `Góly ${A}`, d.goals.awayLines);
  ou("shots", "Střely celkem", d.shots.total.lines);
  ou("shots", `Střely ${H}`, d.shots.home.lines);
  ou("shots", `Střely ${A}`, d.shots.away.lines);
  ou("pens", "Vyloučení celkem", d.pens.total.lines);
  ou("pens", `Vyloučení ${H}`, d.pens.home.lines);
  ou("pens", `Vyloučení ${A}`, d.pens.away.lines);
  const x = d.extra;
  if (x) {
    ou("hits", "Hity celkem", x.hits.total.lines); ou("hits", `Hity ${H}`, x.hits.home.lines); ou("hits", `Hity ${A}`, x.hits.away.lines);
    ou("blocks", "Zblokované střely celkem", x.blocks.total.lines); ou("blocks", `Bloky ${H}`, x.blocks.home.lines); ou("blocks", `Bloky ${A}`, x.blocks.away.lines);
    rows.push({ grp: "pp", id: "pp:h", name: `${H} dá gól v přesilovce`, p: x.ppGoals.home.p1 });
    rows.push({ grp: "pp", id: "pp:hn", name: `${H} nedá gól v přesilovce`, p: 1 - x.ppGoals.home.p1 });
    rows.push({ grp: "pp", id: "pp:a", name: `${A} dá gól v přesilovce`, p: x.ppGoals.away.p1 });
    rows.push({ grp: "pp", id: "pp:an", name: `${A} nedá gól v přesilovce`, p: 1 - x.ppGoals.away.p1 });
    ou("pp", "Góly v přesilovce celkem", x.ppGoals.total.lines);
    rows.push({ grp: "first", id: "fg:h", name: `První gól dá ${d.homeName}`, p: x.firstGoal.home });
    rows.push({ grp: "first", id: "fg:a", name: `První gól dá ${d.awayName}`, p: x.firstGoal.away });
    d.players.filter((p) => p.active).sort((a, b) => b.pFirstGoal - a.pFirstGoal).slice(0, 12)
      .forEach((p) => rows.push({ grp: "first", id: "fgp:" + p.id, name: `První gól zápasu – ${p.name} (${p.team})`, p: p.pFirstGoal }));
  }
  return rows;
}
function renderMarkets() {
  const rows = marketRows().filter((r) => r.grp === GS.mkt);
  $("mkttable").innerHTML = `<thead><tr><th class="left">Sázka</th><th title="Šance podle modelu">Šance (model)</th><th title="1 / šance">Férový kurz</th><th title="Sem napiš kurz z Tipsportu">Kurz sázkovky ✏️</th><th title="Zelená = vyplatí se">Vyplatí se?</th><th>Tiket</th></tr></thead><tbody>` +
    rows.map((r) => {
      const saved = store.get(oddKey(r.id));
      const v = saved ?? (r.auto ? r.auto.toFixed(2) : "");
      return `<tr><td class="left">${esc(r.name)}${r.auto ? ' <span class="pill">Tipsport</span>' : ""}</td><td class="gold">${pc(r.p)}</td>
        <td>${fair(r.p)}</td><td><input type="number" step="0.01" min="1" placeholder="např. 1.85" value="${esc(v)}" data-id="${r.id}" data-p="${r.p}"></td>
        <td class="evcell">${evHtml(r.p, parseFloat(v))}</td><td><button class="addbtn" data-add="${r.id}">+ tiket</button></td></tr>`;
    }).join("") + "</tbody>";
  $("mkttable").querySelectorAll("[data-add]").forEach((b) => (b.onclick = () => {
    const r = rows.find((x) => x.id === b.dataset.add);
    const odd = parseFloat(b.closest("tr").querySelector("input").value);
    addToSlip({ key: `${GAME.id}:${r.id}`, label: r.name, p: r.p, odd }, b);
  }));
  $("mkttable").querySelectorAll("input").forEach((inp) => (inp.oninput = () => {
    store.set(oddKey(inp.dataset.id), inp.value);
    inp.closest("tr").querySelector(".evcell").innerHTML = evHtml(+inp.dataset.p, parseFloat(inp.value));
  }));
}
const PL_COLS = [
  { k: "_rank", t: "#", cls: "rank", nosort: true },
  { k: "name", t: "Hráč", cls: "left", f: (v, p) => `<b class="pname-link" data-player="${p.id}:${p.team}">${esc(v)}</b>${p.streak >= 3 ? ` <span class="pill hot">🔥 ${p.streak} v řadě</span>` : ""}${p.injury ? ` <span class="injbadge ${p.injury.out ? "" : "dtd"}" title="${esc(p.injury.comment)}">🩹 ${esc(p.injury.status)}</span>` : !p.active ? ` <span class="pill bad">nehrál ${p.lastGameAgo} z.</span>` : ""}` },
  { k: "team", t: "Tým", cls: "left pos" },
  { k: "pos", t: "Poz", cls: "left pos", f: (v) => posName[v] || v },
  { k: "pPoint", t: "Šance na bod", cls: "gold", f: (v) => `<span class="pbar" style="width:${Math.round(v * 60)}px"></span>${pc(v)}` },
  { k: "fairPoint", t: "Fér. kurz", v: (p) => 1 / p.pPoint, f: (v) => num(v) },
  { k: "pGoal", t: "Šance na gól", f: pc },
  { k: "expShots", t: "Oček. střely", f: (v) => `<b>${num(v, 1)}</b>` },
  { k: "p2Shots", t: "Střely 1.5+", f: pc },
  { k: "p3Shots", t: "Střely 2.5+", f: pc },
  { k: "p2Points", t: "2+ body", f: pc },
  { k: "pFirstGoal", t: "První gól", f: (v) => (v ? (v * 100).toFixed(1) + " %" : "–") },
  { k: "last5", t: "Body posl. 5", v: (p) => p.form5, f: (v, p) => `<span class="dots">${p.last5.slice().reverse().map((x) => `<i class="${x > 0 ? "y" : ""}" title="${x} b."></i>`).join("")}</span> ${v}` },
  { k: "homePpg", t: "Doma B/Z", f: (v) => num(v) },
  { k: "awayPpg", t: "Venku B/Z", f: (v) => num(v) },
  { k: "gp", t: "Z" }, { k: "g", t: "G" }, { k: "a", t: "A" }, { k: "pts", t: "B" },
  { k: "toi", t: "TOI", f: toi },
];
function renderPlayers() {
  const rows = GAME.players.filter((p) => (GS.teamF === "all" || p.team === GS.teamF) && (GS.showAll || p.active));
  sortTable($("pltable"), PL_COLS, rows, GS.sort, renderPlayers, (p) => (p.active ? "" : "dim"));
}
const CALC_MK = {
  point: ["Kanadský bod (1+)", (p) => p.pPoint], goal: ["Gól", (p) => p.pGoal],
  s15: ["Střely na branku přes 1.5", (p) => p.p2Shots], s25: ["Střely na branku přes 2.5", (p) => p.p3Shots],
  s35: ["Střely na branku přes 3.5", (p) => p.p4Shots],
  p2: ["Aspoň 2 kanadské body", (p) => p.p2Points], fg: ["Vstřelí první gól zápasu", (p) => p.pFirstGoal],
};
function renderCalc() {
  const pid = +$("cpl").value, mk = $("cmk").value, odd = parseFloat($("codd").value);
  const p = GAME.players.find((x) => x.id === pid);
  if (!p) return;
  const prob = CALC_MK[mk][1](p);
  store.set(`odd:${GAME.id}:pl:${pid}:${mk}`, $("codd").value);
  $("cout").innerHTML = `<div><small>Šance podle modelu</small><b style="color:var(--gold)">${pc(prob)}</b></div>
    <div><small>Férový kurz</small><b>${fair(prob)}</b></div><div><small>Hodnota sázky</small><b>${evHtml(prob, odd)}</b></div>
    <div><small>Očekávané střely</small><b>${num(p.expShots, 1)}</b></div>`;
}
function goalieCard(team, t) {
  const rest = t.b2b ? `<span class="pill bad">2. den po sobě – může chytat náhradník</span>`
    : t.restDays != null ? `<span class="pill">odpočinek ${t.restDays} ${t.restDays === 1 ? "den" : t.restDays >= 2 && t.restDays <= 4 ? "dny" : "dní"}</span>` : "";
  return `<div style="margin-bottom:10px"><b>${esc(team)}</b> ${rest}</div>` + t.goalies.map((g) => `<div class="gk">
    <span>${g.likely ? "⭐ " : ""}<b>${esc(g.name)}</b><br><small class="muted">startů v posl. 5: ${g.starts5} · sezóna ${g.gs ?? "–"} startů</small></span>
    <span style="text-align:right"><b>${g.svp != null ? Number(g.svp).toFixed(3) : "–"}</b> %Z<br><small class="muted">GAA ${num(g.gaa)}</small></span></div>`).join("");
}
function renderGame() {
  const d = GAME, H = d.home, A = d.away, th = d.teams[H], ta = d.teams[A];
  const wh = Math.round(d.win.home * 100), wa = 100 - wh;
  const cmp = (h, a, fmt = (x) => x) => {
    const mx = Math.max(h, a) * 1.15;
    return `<div class="cmp"><div class="r"><span style="color:var(--home)">${H}</span><span class="bar"><i style="width:${(h / mx) * 100}%;background:var(--home)"></i></span><span class="num">${fmt(h)}</span></div>
      <div class="r"><span style="color:var(--away)">${A}</span><span class="bar"><i style="width:${(a / mx) * 100}%;background:var(--away)"></i></span><span class="num">${fmt(a)}</span></div></div>`;
  };
  const trendChart = (t, team) => chart({
    labels: t.trend.map((r) => (r.home ? "" : "@") + r.opp),
    bars: [{ name: "Střely " + team, color: team === H ? "#38bdf8" : "#a78bfa", values: t.trend.map((r) => r.sf) },
           { name: "Střely soupeře", color: "rgba(148,163,184,.45)", values: t.trend.map((r) => r.sa) }],
    avg: d.league.shots,
  });
  const penChart = (t, team) => chart({
    labels: t.trend.map((r) => (r.home ? "" : "@") + r.opp), height: 150,
    lines: [{ name: "Vyloučení " + team, color: team === H ? "#38bdf8" : "#a78bfa", values: t.trend.map((r) => r.pt) },
            { name: "Vyloučení soupeře", color: "#fb923c", values: t.trend.map((r) => r.pd) },
            { name: "Góly " + team, color: "#34d399", values: t.trend.map((r) => r.gf) }],
  });
  const h2h = d.h2h.length ? d.h2h.map((r) => {
    const [l, rr] = r.home ? [`${H} ${r.gf}`, `${r.ga} ${A}`] : [`${A} ${r.ga}`, `${r.gf} ${H}`];
    return `<div class="gk"><span>${fmtDate(r.date)}</span><b>${l}:${rr}${r.end !== "REG" ? " " + r.end : ""}</b>
      <small class="muted">střely ${r.home ? r.sf + ":" + r.sa : r.sa + ":" + r.sf} · vyl. ${r.home ? r.pt + ":" + r.pd : r.pd + ":" + r.pt}</small></div>`;
  }).join("") : `<div class="muted">Letos spolu ještě nehráli.</div>`;
  const plOpts = d.players.filter((p) => p.active).map((p) => `<option value="${p.id}">${esc(p.name)} (${p.team}) – bod ${pc(p.pPoint)}</option>`).join("");

  $("view-game").innerHTML = `
    <button class="btn back" onclick="location.hash=''">← Všechny zápasy</button>
    <div class="card hero" style="--hc:${tcol(H)};--ac:${tcol(A)}">${RINK}
      <div class="hero-grid">
        <div class="hteam"><img src="${esc(d.homeLogo)}" alt=""><h2>${esc(d.homeName)}</h2><div class="muted">doma ${th.form.home} · celkem ${th.form.all}</div>
          <div class="form">${th.form.results5.slice().reverse().map((r) => `<span class="res ${r}">${r}</span>`).join("")}</div></div>
        <div class="hmid"><div class="when">${fmtDay(d.start)} · ${fmtTime(d.start)}${d.venue ? " · " + esc(d.venue) : ""}<br><span class="countdown" data-start="${d.start}"></span></div>
          <div class="score">${d.goals.home.toFixed(1)} : ${d.goals.away.toFixed(1)}</div>
          <div class="lbl">očekávané góly · nejčastější výsledek <b>${d.goals.likelyScore}</b> · prodloužení ${pc(d.win.ot)}</div></div>
        <div class="hteam"><img src="${esc(d.awayLogo)}" alt=""><h2>${esc(d.awayName)}</h2><div class="muted">venku ${ta.form.away} · celkem ${ta.form.all}</div>
          <div class="form">${ta.form.results5.slice().reverse().map((r) => `<span class="res ${r}">${r}</span>`).join("")}</div></div>
      </div>
      <div class="bigwin"><div class="wbar"><i class="h" style="width:${wh}%"></i><i class="a" style="width:${wa}%"></i></div>
        <div class="wlab"><span class="h">${H} ${wh} %</span><span class="muted" style="font-weight:600;font-size:13px">šance na výhru vč. prodloužení</span><span class="a">${wa} % ${A}</span></div>
        <div class="odds" style="justify-content:center;gap:14px">${d.odds?.home ? `Tipsport: ${H} <span class="odd">${d.odds.home.toFixed(2)}</span> ${evHtml(d.win.home, d.odds.home)}
          &nbsp;·&nbsp; ${A} <span class="odd">${d.odds.away.toFixed(2)}</span> ${evHtml(d.win.away, d.odds.away)}` : "Kurzy Tipsportu zatím nejsou k dispozici"}
          <span>Férové kurzy: ${fair(d.win.home)} / ${fair(d.win.away)}</span></div></div>
      <p class="help" style="text-align:center;max-width:820px;margin:14px auto 0">
        <b>Velká čísla uprostřed</b> = kolik gólů model čeká (domácí : hosté). <b>Nejčastější výsledek</b> = skóre, které je ze všech možných nejpravděpodobnější (i tak je jeho šance jen kolem 10 %).
        <b>Prodloužení</b> = šance, že po 60 minutách bude remíza. <b>Pruh</b> = šance na výhru. <b>Tipsport</b> = kurz sázkovky, vedle je <b>hodnota</b>: zelené + znamená, že se sázka podle modelu vyplatí, červené − že ne.
        <b>Férové kurzy</b> = jaký kurz by odpovídal šanci podle modelu. Barevné čtverečky pod týmy = posledních 5 zápasů (V výhra, P prohra, PP prohra v prodloužení), nejnovější vpravo.</p>
      <div class="hero-actions"><button class="btn primary" id="aibtn">🤖 AI rozbor zápasu</button>
        <a class="btn" href="#trhy" style="text-decoration:none">💰 Kurzy a hodnota</a><a class="btn" href="#hraci" style="text-decoration:none">🎯 Hráči</a></div>
    </div>

    ${d.retro || ["LIVE", "CRIT", "FINAL", "OFF"].includes(d.state) ? `<div class="retro">${["LIVE", "CRIT"].includes(d.state) ? "🚨 <b>Zápas se právě hraje.</b>" : "<b>Zápas už začal nebo skončil.</b>"}
      Předpověď níže je spočítaná jen z dat <b>před zápasem</b>, takže ji můžeš porovnat se skutečností. <a href="#live-${d.id}">Otevřít Live hřiště / záznam →</a></div>` : ""}
    <div class="statrow">
      <div class="card stat"><h4>🏒 Střely na branku</h4><div class="total">${d.shots.total.exp} <small>celkem</small></div>${cmp(d.shots.home.exp, d.shots.away.exp, (x) => x.toFixed(1))}
        <p class="help">Kolik <b>střel na branku</b> čekáme za oba týmy dohromady (velké číslo) a za každý tým zvlášť (pruhy). Ligový průměr je ${d.league.shots.toFixed(1)} střel na tým.</p></div>
      <div class="card stat"><h4>⛔ Vyloučení</h4><div class="total">${d.pens.total.exp.toFixed(1)} <small>celkem</small></div>${cmp(d.pens.home.exp, d.pens.away.exp, (x) => x.toFixed(1))}
        <p class="help">Kolik <b>trestů (vyloučení)</b> čekáme. Číslo u týmu = kolikrát půjdou <b>jeho</b> hráči na trestnou lavici. Ligový průměr je ${d.league.pens.toFixed(1)} na tým.</p></div>
      <div class="card stat"><h4>🥅 Góly</h4><div class="total">${d.goals.total.toFixed(1)} <small>celkem</small></div>${cmp(d.goals.home, d.goals.away, (x) => x.toFixed(2))}
        <p class="help">Kolik <b>gólů</b> čekáme celkem a za každý tým. Ligový průměr je ${d.league.goals.toFixed(2)} gólu na tým.</p></div>
    </div>

    <div class="sechead"><span class="no">⚖</span><h2>Klíčové faktory</h2><span class="hint">přesilovky, oslabení, vhazování, marodka</span></div>
    <div class="grid g3">
      <div class="card"><b>Speciální týmy a vhazování</b>
        <p class="help" style="margin:4px 0 10px"><b>Přesilovka %</b> = jak často tým dá gól, když má o hráče víc. <b>Oslabení %</b> = jak často ubrání, když má o hráče míň. <b>Vhazování %</b> = kolik vhazování vyhraje. Vyšší = lepší.</p>
        ${(() => { const a = d.special?.[H] || {}, b = d.special?.[A] || {};
          const row = (lab, x, y) => { if (x == null || y == null) return ""; const mx = Math.max(x, y) || 1;
            return `<div class="cmprow"><div class="lab">${lab}</div>
              <div class="r"><span class="b"><i data-w="${(x / mx) * 100}" style="background:var(--home)"></i></span><span class="v" style="color:var(--home)">${H} ${(x * 100).toFixed(1)} %</span></div>
              <div class="r"><span class="b"><i data-w="${(y / mx) * 100}" style="background:var(--away)"></i></span><span class="v" style="color:var(--away)">${A} ${(y * 100).toFixed(1)} %</span></div></div>`; };
          const pp = a.pp != null && b.pk != null ? `<p class="help">⚡ Přesilovka ${H} (${(a.pp * 100).toFixed(0)} %) proti oslabení ${A} (${(b.pk * 100).toFixed(0)} %) · přesilovka ${A} (${(b.pp * 100).toFixed(0)} %) proti oslabení ${H} (${(a.pk * 100).toFixed(0)} %).</p>` : "";
          return (row("Přesilovka %", a.pp, b.pp) + row("Oslabení %", a.pk, b.pk) + row("Vhazování %", a.fo, b.fo) + pp) || `<p class="help">Data zatím nejsou.</p>`; })()}
      </div>
      <div class="card"><b>🩹 Marodka</b>
        <p class="help" style="margin:4px 0 10px">Zranění a vyřazení hráči (zdroj ESPN). Hráči označení jako <b>mimo hru</b> se do předpovědi nepočítají. <b>Day-To-Day</b> = nejisté, jestli nastoupí.</p>
        ${[H, A].map((t) => { const L = d.injuries?.[t] || []; return `<div style="margin-top:8px"><b style="color:${t === H ? "var(--home)" : "var(--away)"}">${t}</b>${L.length ? L.map((i) =>
          `<div class="gk"><span><b>${esc(i.name)}</b><br><small class="muted">${esc(i.comment || "")}</small></span><span style="text-align:right"><span class="injbadge ${i.out ? "" : "dtd"}">${esc(i.status)}</span>${i.pts != null ? `<br><small class="muted">${i.pts} b. v sezóně</small>` : ""}</span></div>`).join("")
          : `<p class="help">Nikdo zraněný ✅</p>`}</div>`; }).join("")}
      </div>
      <div class="card books"><b>💰 Kurzy sázkovek na vítěze</b>
        <p class="help" style="margin:4px 0 10px">Kurzy z oficiálních dat NHL. <b>Zelená</b> = nejlepší kurz. Sloupec <b>hodnota</b> ukazuje, jestli se sázka podle modelu vyplatí (kladné = ano). Sázkovky označené „1X2“ nabízejí sázku jen na základní hrací dobu (bez prodloužení).</p>
        ${(d.odds?.books || []).length ? `<div class="tablewrap" style="background:none"><table class="t"><thead><tr><th class="left">Sázkovka</th><th>${H}</th><th>${A}</th><th>Hodnota ${H}</th><th>Hodnota ${A}</th></tr></thead><tbody>
          ${(() => { const B = d.odds.books, bh = Math.max(...B.filter((x) => !x.reg).map((x) => x.home), 0), ba = Math.max(...B.filter((x) => !x.reg).map((x) => x.away), 0);
            return B.map((b) => `<tr><td class="left"><b>${esc(b.name)}</b> <small class="muted">${esc(b.country || "")}${b.reg ? " · 1X2" : ""}</small></td>
              <td class="${!b.reg && b.home === bh ? "best" : ""}">${b.home.toFixed(2)}</td><td class="${!b.reg && b.away === ba ? "best" : ""}">${b.away.toFixed(2)}</td>
              <td>${evHtml(b.pHome, b.home)}</td><td>${evHtml(b.pAway, b.away)}</td></tr>`).join(""); })()}
          </tbody></table></div>` : `<p class="help">Kurzy k tomuto zápasu zatím nejsou.</p>`}
      </div>
    </div>

    <div class="sechead"><span class="no">▶</span><h2>Simulace zápasu</h2><span class="hint">počítač odehraje zápas 1000×</span></div>
    <div class="card">
      <p class="help" style="margin:0 0 14px">Klikni na <b>„Odehraj zápas 1000×“</b>. Počítač zápas tisíckrát „odehraje“ podle čísel modelu – pokaždé trochu jinak, jako ve skutečnosti.
        Na světelné tabuli uvidíš, jak běží jednotlivé zápasy, a pod ní, <b>jak často co vyšlo</b>: kolikrát vyhrál který tým, která skóre padala nejčastěji, kolik bylo střel a vyloučení
        a kteří hráči bodovali nejčastěji. Je to jiný pohled na stejná čísla – čím víc simulací něco vyhraje, tím je to pravděpodobnější.</p>
      <div class="simtop">
        <div class="scoreboard"><div class="tm"><span>${H}</span><span>${A}</span></div><div class="digits" id="simscore">0 : 0</div><small id="simnote">zatím neodehráno</small></div>
        <div><button class="btn primary" id="simbtn" style="font-size:17px;padding:14px 22px">▶ Odehraj zápas 1000×</button>
          <div class="simstats" style="margin-top:14px">
            <div><small>Odehráno</small><b id="simn">0</b></div>
            <div><small>Výhry ${H}</small><b id="simwh" style="color:var(--home)">–</b></div>
            <div><small>Výhry ${A}</small><b id="simwa" style="color:var(--away)">–</b></div>
            <div><small>Prodloužení</small><b id="simot">–</b></div>
          </div></div>
      </div>
      <div class="simgrid">
        <div><b>Nejčastější výsledky</b><div id="simscores" class="help"></div></div>
        <div><b>Střely celkem</b><div id="simshots"></div><p class="help">Výška sloupce = jak často padl daný počet střel.</p></div>
        <div><b>Vyloučení celkem</b><div id="simpens"></div></div>
        <div><b>Kdo nejčastěji bodoval</b><div id="simpl" class="help"></div></div>
      </div>
    </div>

    <div class="sechead"><span class="no">◎</span><h2>Hřiště</h2><span class="hint">odkud týmy střílí – posledních 10 zápasů</span></div>
    <div class="card">
      <p class="help" style="margin:0 0 12px">Na hřišti je vidět <b>každá střela na branku</b> z posledních 10 zápasů týmu – útočí se vždy <b>doprava na branku</b>.
        <b>Modré tečky</b> = střely, <b>zlaté zářící</b> = góly. Barevné skvrny = místa, odkud se střílí nejčastěji. Vpravo je, kolik střel jde z blízka a kolik z dálky.
        Přepni si tým a jestli chceš vidět <b>střely týmu</b>, nebo <b>střely soupeřů proti němu</b> (= jak dobře brání). Tlačítko <b>„Přehrát“</b> střely znovu animuje.</p>
      <div class="controls" style="margin-top:0">
        <div class="seg" id="rkteam"><button data-v="${H}" class="on">${H}</button><button data-v="${A}">${A}</button></div>
        <div class="seg" id="rkmode"><button data-v="for" class="on">Střely týmu</button><button data-v="against">Střely soupeřů</button></div>
        <button class="btn" id="rkplay">▶ Přehrát</button>
      </div>
      <div class="rinkwrap"><div id="rinkbox"><div class="loader"><div class="spinner"></div>Načítám střely…</div></div><div id="rinkzones" class="zones"></div></div>
    </div>

    <div class="sechead"><span class="no">⇆</span><h2>Přes nebo pod?</h2><span class="hint">posuň hranici a sleduj šanci</span></div>
    <div class="card slider">
      <p class="help" style="margin:0 0 12px">Vyber, co tě zajímá, a <b>táhni posuvníkem</b>. Velké číslo = hranice (např. 58.5 střel). Zelený box = šance, že padne <b>víc</b>,
        červený = šance, že padne <b>míň</b>. Pod tím je férový kurz. Graf ukazuje, jak pravděpodobné je každé číslo – žlutá čára je tvoje hranice.</p>
      <div class="seg" id="slkind" style="margin-bottom:14px">
        <button data-v="shots.total" class="on">Střely celkem</button><button data-v="shots.home">Střely ${H}</button><button data-v="shots.away">Střely ${A}</button>
        <button data-v="goals.total">Góly celkem</button><button data-v="hits.total">Hity celkem</button><button data-v="blocks.total">Bloky celkem</button><button data-v="pens.total">Vyloučení celkem</button><button data-v="pens.home">Vyloučení ${H}</button><button data-v="pens.away">Vyloučení ${A}</button></div>
      <div class="srow"><input type="range" id="slrange"><div class="lineval" id="slval"></div></div>
      <div class="split">
        <div class="side over"><div class="fill" id="slfo"></div><small>Přes (víc než hranice)</small><b id="slpo"></b><span id="slfa"></span></div>
        <div class="side under"><div class="fill" id="slfu"></div><small>Pod (míň než hranice)</small><b id="slpu"></b><span id="slfb"></span></div>
      </div>
      <div class="dist" id="sldist"></div>
      <div style="display:flex;gap:10px;margin-top:12px;flex-wrap:wrap"><button class="addbtn" id="sladdo">+ Na tiket: přes</button><button class="addbtn" id="sladdu">+ Na tiket: pod</button></div>
    </div>

    <h2 class="sec" id="trhy">💰 Kurzy a hodnota sázek <span class="hint">zadej kurz z Tipsportu – zelená = model vidí hodnotu</span></h2>
    <div class="card mkt">
      <div class="help" style="margin:0 0 14px"><b>Jak na to:</b> 1) Nahoře vyber typ sázky (výsledek, góly, střely, vyloučení). 2) Na Tipsportu najdi stejnou sázku.
        3) Její kurz napiš do políčka <b>„Kurz sázkovky“</b>. 4) Vpravo uvidíš <b>hodnotu</b>: <span class="ev good">zelená +</span> = kurz je lepší, než by podle modelu měl být (vyplatí se),
        <span class="ev bad">červená −</span> = nevyplatí se.<br><b>Příklad:</b> model dává 60 % → férový kurz 1.67. Když Tipsport nabízí 1.90, je to +14 % hodnota. Když nabízí 1.50, je to −10 %.
        <br>U vítěze zápasu se kurzy z Tipsportu vyplní samy. <b>Hity</b> = tvrdé bodyčeky, <b>Bloky</b> = střely zastavené tělem bránícího hráče,
        <b>Přesilovky</b> = góly, když má tým víc hráčů na ledě (soupeř je vyloučený), <b>První gól</b> = kdo skóruje jako první v zápase.</div>
      <div class="seg" id="mkseg" style="margin-bottom:12px">
        <button data-v="win" class="${GS.mkt === "win" ? "on" : ""}">Výsledek</button><button data-v="goals" class="${GS.mkt === "goals" ? "on" : ""}">Góly</button>
        <button data-v="shots" class="${GS.mkt === "shots" ? "on" : ""}">Střely</button><button data-v="pens" class="${GS.mkt === "pens" ? "on" : ""}">Vyloučení</button>
        <button data-v="hits" class="${GS.mkt === "hits" ? "on" : ""}">Hity</button><button data-v="blocks" class="${GS.mkt === "blocks" ? "on" : ""}">Bloky</button>
        <button data-v="pp" class="${GS.mkt === "pp" ? "on" : ""}">Přesilovky</button><button data-v="first" class="${GS.mkt === "first" ? "on" : ""}">První gól</button></div>
      <div class="tablewrap" style="background:none"><table class="t" id="mkttable"></table></div>
      <div class="legendtxt">Férový kurz = 1 / šance podle modelu. Hodnota (EV) = šance × kurz − 1: kladné číslo znamená, že sázkovka dává vyšší kurz,
        než by podle modelu odpovídalo. Kurzy, které zadáš, si stránka pamatuje v tvém prohlížeči.</div>
    </div>

    <h2 class="sec">🧮 Kalkulačka hráče <span class="hint">bod, gól nebo střely konkrétního hráče</span></h2>
    <div class="card"><p class="help" style="margin:0 0 12px">Vyber hráče, druh sázky a napiš kurz z Tipsportu. Uvidíš šanci podle modelu, férový kurz a jestli se sázka vyplatí.
      <b>„Střely přes 2.5“</b> = hráč vystřelí na branku aspoň 3×. <b>„Kanadský bod“</b> = dá gól nebo asistenci.</p><div class="calc">
      <select id="cpl">${plOpts}</select>
      <select id="cmk">${Object.entries(CALC_MK).map(([k, v]) => `<option value="${k}">${v[0]}</option>`).join("")}</select>
      <input id="codd" type="number" step="0.01" min="1" placeholder="Kurz Tipsport" style="width:150px"></div>
      <div class="out" id="cout"></div>
      <button class="addbtn" id="caddbtn" style="margin-top:12px;font-size:14px;padding:8px 14px">+ Přidat na tiket</button></div>

    <div class="sechead" id="hraci"><span class="no">★</span><h2>Hráči zápasu</h2><span class="hint">6 největších šancí na bod</span></div>
    <div class="pcards">${d.players.filter((p) => p.active).slice(0, 6).map((p) => playerCard(p, p.team === H ? d.homeLogo : d.awayLogo)).join("")}</div>
    <div class="sechead"><span class="no">VS</span><h2>Porovnání hráčů</h2><span class="hint">vyber 2–3 hráče</span></div>
    <div class="card">
      <p class="help" style="margin:0 0 12px">Vyber až 3 hráče (i z obou týmů). <b>Pavučinový graf</b> vlevo ukazuje, v čem je kdo silný – čím větší barevná plocha, tím lepší.
        Každý cíp = jedna vlastnost, okraj grafu = nejlepší hráč v tomto zápase. Vpravo jsou stejná čísla jako pruhy s přesnou hodnotou.</p>
      <div class="cmpsel" id="cmpsel"></div>
      <div class="cmpwrap"><div id="radar"></div><div id="cmpbars"></div></div>
    </div>

    <h2 class="sec">🎯 Všichni hráči <span class="hint">klikni na záhlaví pro řazení</span></h2>
    <div class="controls" style="margin-top:0">
      <div class="seg" id="plteam"><button data-v="all" class="on">Oba týmy</button><button data-v="${H}">${H}</button><button data-v="${A}">${A}</button></div>
      <label class="muted" style="display:flex;gap:6px;align-items:center;font-size:13px"><input type="checkbox" id="plall"> i hráči, kteří poslední zápasy nehráli</label>
    </div>
    <p class="help" style="margin:-4px 0 10px"><b>Co znamenají sloupce:</b> <b>Šance na bod</b> = gól nebo asistence ·
      <b>Fér. kurz</b> = jaký kurz by odpovídal té šanci · <b>Šance na gól</b> = sám dá gól · <b>Oček. střely</b> = kolik střel na branku čekáme ·
      <b>Střely 1.5+ / 2.5+</b> = šance na aspoň 2 / aspoň 3 střely · <b>2+ body</b> = šance na aspoň 2 kanadské body · <b>První gól</b> = šance, že vstřelí první gól celého zápasu · <b>Body posl. 5</b> = žluté čtverečky = zápasy, kdy bodoval (nejnovější vpravo), číslo = body celkem ·
      <b>Doma / Venku B/Z</b> = průměr bodů na zápas doma a venku · <b>Z, G, A, B</b> = zápasy, góly, asistence, body za sezónu · <b>TOI</b> = čas na ledě v posledních 3 zápasech.
      Šedí hráči v posledních zápasech nehráli (zranění, mimo sestavu).</p>
    <div class="tablewrap"><table class="t" id="pltable"></table></div>

    <h2 class="sec">📈 Forma – posledních 10 zápasů <span class="hint">@ = venku · žlutá čára = ligový průměr</span></h2>
    <p class="help" style="margin:-4px 0 12px">Grafy ukazují posledních 10 zápasů obou týmů zleva (nejstarší) doprava (nejnovější). Pod sloupci je soupeř, <b>@</b> znamená, že tým hrál venku.
      <b>Barevný sloupec</b> = kolik střel tým vystřelil, <b>šedý</b> = kolik jich vystřelil soupeř. <b>Žlutá přerušovaná čára</b> = průměr ligy. Na čárových grafech vidíš vyloučení obou týmů a góly.</p>
    <div class="grid g2 charts">
      <div class="card"><b>${H} – střely na branku</b>${trendChart(th, H)}<div class="legend"><span><i style="background:#38bdf8"></i>${H}</span><span><i style="background:rgba(148,163,184,.45)"></i>soupeř</span></div></div>
      <div class="card"><b>${A} – střely na branku</b>${trendChart(ta, A)}<div class="legend"><span><i style="background:#a78bfa"></i>${A}</span><span><i style="background:rgba(148,163,184,.45)"></i>soupeř</span></div></div>
      <div class="card"><b>${H} – vyloučení a góly</b>${penChart(th, H)}<div class="legend"><span><i style="background:#38bdf8"></i>vyloučení ${H}</span><span><i style="background:#fb923c"></i>vyloučení soupeře</span><span><i style="background:#34d399"></i>góly ${H}</span></div></div>
      <div class="card"><b>${A} – vyloučení a góly</b>${penChart(ta, A)}<div class="legend"><span><i style="background:#a78bfa"></i>vyloučení ${A}</span><span><i style="background:#fb923c"></i>vyloučení soupeře</span><span><i style="background:#34d399"></i>góly ${A}</span></div></div>
    </div>

    <h2 class="sec">🧠 Proč model čeká tohle</h2>
    <p class="help" style="margin:-6px 0 12px">Tady je česky vysvětleno, z jakých čísel model vychází. Bere útok jednoho týmu a obranu druhého a porovná je s průměrem ligy.
      <b>⭐ u brankáře</b> = nejspíš bude chytat (chytal nejvíc z posledních 5 zápasů). <b>%Z</b> = kolik procent střel chytí, <b>GAA</b> = kolik gólů průměrně dostane.</p>
    <div class="grid g3">
      <div class="card"><b style="color:var(--home)">${H}</b><ul class="why">${th.why.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>
      <div class="card"><b style="color:var(--away)">${A}</b><ul class="why">${ta.why.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>
      <div class="card"><b>🧤 Pravděpodobní brankáři</b><div style="height:10px"></div>${goalieCard(H, th)}<div style="height:10px"></div>${goalieCard(A, ta)}</div>
      <div class="card"><b>⚔️ Vzájemné zápasy v sezóně</b><div style="height:6px"></div>${h2h}</div>
    </div>


    <details class="gloss card" open><summary>📖 Slovníček – co znamenají pojmy na stránce</summary><dl>
      <dt>Kanadský bod</dt><dd>Hráč dostane bod za gól i za asistenci (přihrávku na gól). „Šance na bod“ = šance, že udělá aspoň jedno z toho.</dd>
      <dt>Šance (%)</dt><dd>Jak moc je něco pravděpodobné. 50 % = jako hod mincí, 80 % = stane se to v 8 z 10 případů. Ani 90 % není jistota.</dd>
      <dt>Očekávané (např. 29.8 střel)</dt><dd>Průměr, který model čeká. Skutečné číslo bude kolem něj – někdy víc, někdy míň.</dd>
      <dt>Střely na branku</dt><dd>Střely, které by šly do branky (gól nebo zákrok brankáře). Střely vedle nebo zblokované se nepočítají.</dd>
      <dt>Vyloučení</dt><dd>Počet trestů (hráč jde na trestnou lavici). Nepočítají se osobní tresty na 10 minut.</dd>
      <dt>Přes / pod (např. přes 5.5)</dt><dd>Sázka na to, jestli padne víc nebo míň než dané číslo. „Přes 5.5 gólu“ = padne 6 a víc gólů.</dd>
      <dt>Kurz</dt><dd>Kolikrát se ti vrátí vsazená částka, když sázka vyjde. Kurz 2.00 = vsadíš 100 Kč, dostaneš 200 Kč.</dd>
      <dt>Férový kurz</dt><dd>Kurz, který by „spravedlivě“ odpovídal šanci podle modelu (1 ÷ šance). Šance 50 % → férový kurz 2.00.</dd>
      <dt>Hodnota (EV)</dt><dd>Jestli se sázka podle modelu vyplatí. Když je kurz sázkovky vyšší než férový kurz, vychází kladné číslo (zelené) = sázka má hodnotu. Záporné (červené) = nevyplatí se.</dd>
      <dt>Vč. prodloužení</dt><dd>Počítá se, kdo vyhraje celý zápas, i kdyby to bylo až v prodloužení nebo nájezdech.</dd>
      <dt>Základní hrací doba</dt><dd>Jen 60 minut bez prodloužení. Tam může být i remíza.</dd>
      <dt>Forma (V / P / PP)</dt><dd>Výsledky posledních zápasů: V = výhra, P = prohra, PP = prohra v prodloužení nebo nájezdech.</dd>
      <dt>Doma / venku</dt><dd>Týmy často hrají doma jinak než venku, proto to model počítá zvlášť.</dd>
      <dt>2. den po sobě</dt><dd>Tým hrál i včera – bývá unavenější a často chytá náhradní brankář. Model mu trochu ubere.</dd>
      <dt>%Z (úspěšnost zásahů)</dt><dd>Kolik procent střel brankář chytí. 0.910 = 91 %. Víc je líp.</dd>
      <dt>GAA</dt><dd>Kolik gólů brankář průměrně dostane za zápas. Míň je líp.</dd>
      <dt>TOI</dt><dd>Průměrný čas hráče na ledě za zápas (minuty:sekundy). Víc času = víc šancí bodovat.</dd>
      <dt>B/Z</dt><dd>Body na zápas – průměr kanadských bodů za jeden zápas.</dd>
      <dt>Σ</dt><dd>Součet za oba týmy dohromady.</dd>
    </dl></details>
    <div class="disc"><b>Jak to model počítá:</b> bere ${d.games.home} + ${d.games.away} odehraných zápasů obou týmů, novější mají větší váhu.
      Útok týmu (střely, góly, vynucená vyloučení) kombinuje s obranou soupeře zvlášť doma a venku a porovnává s ligovým průměrem.
      Výsledek a góly počítá Poissonovým rozdělením, u hráčů bere bodový průměr, podíl zápasů s bodem, formu, doma/venku, čas na ledě
      a kolik gólů má jeho tým v tomto zápase čekat. Únava (2 zápasy za sebou) snižuje očekávání.<br>
      Model neví o čerstvých zraněních ani o tom, kdo skutečně chytá. Je to odhad, ne jistota – sázej zodpovědně.</div>`;

  segment($("mkseg"), (v) => { GS.mkt = v; renderMarkets(); });
  segment($("plteam"), (v) => { GS.teamF = v; renderPlayers(); });
  $("plall").onchange = (e) => { GS.showAll = e.target.checked; renderPlayers(); };
  $("aibtn").onclick = () => { openChat(); sendChat(`Udělej mi podrobný rozbor zápasu ${H} – ${A}: kdo je favorit, kolik čekat střel, vyloučení a gólů, kteří hráči mají největší šanci na bod a kde vidíš největší hodnotu v kurzech.`); };
  const loadCalcOdd = () => { $("codd").value = store.get(`odd:${GAME.id}:pl:${$("cpl").value}:${$("cmk").value}`) || ""; renderCalc(); };
  $("cpl").onchange = loadCalcOdd; $("cmk").onchange = loadCalcOdd; $("codd").oninput = renderCalc;
  renderMarkets(); renderPlayers(); loadCalcOdd();
  initSlider(); initSim(); initRink(); initCompare();
  requestAnimationFrame(() => setTimeout(() => $("view-game").querySelectorAll(".cmprow i[data-w]").forEach((i) => (i.style.width = i.dataset.w + "%")), 60));
  $("sladdo").onclick = (e) => slAdd(true, e.target); $("sladdu").onclick = (e) => slAdd(false, e.target);
  $("caddbtn").onclick = (e) => {
    const p = GAME.players.find((x) => x.id === +$("cpl").value), mk = $("cmk").value;
    if (!p) return;
    addToSlip({ key: `${GAME.id}:pl:${p.id}:${mk}`, label: `${p.name} – ${CALC_MK[mk][0]}`, p: CALC_MK[mk][1](p), odd: parseFloat($("codd").value) }, e.target);
  };
  reveal($("view-game")); interactive($("view-game"));
  document.querySelectorAll('#view-game a[href^="#"]:not([href="#"])').forEach((a) => (a.onclick = (e) => {
    e.preventDefault(); document.getElementById(a.getAttribute("href").slice(1)).scrollIntoView({ behavior: "smooth" });
  }));
}

// ===================================================================
//  Simulace zápasu
// ===================================================================
function rpois(l) { const L = Math.exp(-l); let k = 0, p = 1; do { k++; p *= Math.random(); } while (p > L); return k - 1; }
function rnorm() { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
let simRun = 0;
function initSim() { $("simbtn").onclick = runSim; }
function histSvg(map, color) {
  const ks = Object.keys(map).map(Number).sort((a, b) => a - b);
  if (!ks.length) return "";
  const lo = ks[0], hi = ks[ks.length - 1], arr = [];
  for (let k = lo; k <= hi; k++) arr.push([k, map[k] || 0]);
  const W = 420, H = 120, mx = Math.max(...arr.map((a) => a[1])), bw = W / arr.length;
  let s = `<svg viewBox="0 0 ${W} ${H + 18}" xmlns="http://www.w3.org/2000/svg" font-family="Inter,sans-serif">`;
  arr.forEach(([k, v], i) => {
    const h = (v / mx) * H;
    s += `<rect x="${i * bw + 1}" y="${H - h}" width="${Math.max(1, bw - 2)}" height="${h}" rx="2" fill="${color}" opacity=".8"><title>${k}: ${v}×</title></rect>`;
    if (arr.length < 22 || k % 5 === 0) s += `<text x="${i * bw + bw / 2}" y="${H + 14}" font-size="10" fill="#8796ae" text-anchor="middle">${k}</text>`;
  });
  return s + "</svg>";
}
function runSim() {
  const d = GAME, H = d.home, A = d.away, my = ++simRun, N = 1000;
  const gH = d.goals.home, gA = d.goals.away;
  const sh = d.shots.home, sa = d.shots.away;
  const pl = d.players.filter((p) => p.active).slice(0, 14);
  const st = { n: 0, wh: 0, wa: 0, ot: 0, scores: {}, shots: {}, pens: {}, pts: {} };
  $("simbtn").disabled = true; $("simbtn").textContent = "⏳ Simuluji…";
  const one = () => {
    let h = rpois(gH), a = rpois(gA), ot = false;
    const key = `${h}:${a}`; st.scores[key] = (st.scores[key] || 0) + 1;
    if (h === a) { ot = true; st.ot++; if (Math.random() < gH / (gH + gA)) h++; else a++; }
    if (h > a) st.wh++; else st.wa++;
    const shots = Math.max(0, Math.round((sh.mean ?? sh.exp) + rnorm() * (sh.sd ?? 6))) + Math.max(0, Math.round((sa.mean ?? sa.exp) + rnorm() * (sa.sd ?? 6)));
    st.shots[shots] = (st.shots[shots] || 0) + 1;
    const pens = rpois(d.pens.home.exp) + rpois(d.pens.away.exp);
    st.pens[pens] = (st.pens[pens] || 0) + 1;
    const goalsReg = { [H]: +key.split(":")[0], [A]: +key.split(":")[1] };
    pl.forEach((p) => {
      const expG = p.team === H ? gH : gA, k = goalsReg[p.team] / expG;
      const prob = 1 - Math.pow(1 - p.pPoint, k);
      if (Math.random() < prob) st.pts[p.id] = (st.pts[p.id] || 0) + 1;
    });
    st.n++;
    return { h, a, ot, shots, pens };
  };
  const frame = () => {
    if (my !== simRun || !$("simscore")) return;
    let last;
    const batch = st.n < 60 ? 2 : st.n < 300 ? 12 : 30;
    for (let i = 0; i < batch && st.n < N; i++) last = one();
    $("simscore").textContent = `${last.h} : ${last.a}`;
    $("simnote").textContent = `simulace č. ${st.n}${last.ot ? " · po prodloužení" : ""} · ${last.shots} střel · ${last.pens} vyloučení`;
    $("simn").textContent = st.n;
    $("simwh").textContent = Math.round((st.wh / st.n) * 100) + " %";
    $("simwa").textContent = Math.round((st.wa / st.n) * 100) + " %";
    $("simot").textContent = Math.round((st.ot / st.n) * 100) + " %";
    const top = Object.entries(st.scores).sort((a, b) => b[1] - a[1]).slice(0, 8), mx = top[0][1];
    $("simscores").innerHTML = top.map(([k, v]) => `<div class="hbar"><span>${H} ${k} ${A}</span><span class="b"><i style="width:${(v / mx) * 100}%"></i></span><span class="n">${Math.round((v / st.n) * 1000) / 10} %</span></div>`).join("");
    $("simshots").innerHTML = histSvg(st.shots, "#38bdf8");
    $("simpens").innerHTML = histSvg(st.pens, "#fb923c");
    const tp = pl.map((p) => [p, st.pts[p.id] || 0]).sort((a, b) => b[1] - a[1]).slice(0, 8), mp = Math.max(1, tp[0][1]);
    $("simpl").innerHTML = tp.map(([p, v]) => `<div class="hbar"><span>${esc(splitName(p.name)[1])} <small class="muted">${p.team}</small></span><span class="b"><i style="width:${(v / mp) * 100}%;background:linear-gradient(90deg,var(--orange),var(--gold))"></i></span><span class="n">${Math.round((v / st.n) * 100)} %</span></div>`).join("");
    if (st.n < N) requestAnimationFrame(frame);
    else { $("simbtn").disabled = false; $("simbtn").textContent = "↻ Odehraj znovu 1000×"; $("simnote").textContent = `hotovo – odehráno ${N} zápasů (poslední výsledek nahoře)`; }
  };
  requestAnimationFrame(frame);
}

// ===================================================================
//  Hřiště – mapa střel
// ===================================================================
const RK = { team: null, mode: "for", data: {} };
function initRink() {
  RK.team = GAME.home; RK.mode = "for";
  segment($("rkteam"), (v) => { RK.team = v; drawRink(); });
  segment($("rkmode"), (v) => { RK.mode = v; drawRink(); });
  $("rkplay").onclick = () => drawRink(true);
  drawRink();
}
async function drawRink() {
  const team = RK.team, box = $("rinkbox");
  if (!RK.data[team]) {
    box.innerHTML = `<div class="loader"><div class="spinner"></div>Načítám střely z posledních zápasů…</div>`;
    try { RK.data[team] = await getJSON("/model/shotmap?team=" + team); }
    catch (e) { box.innerHTML = `<div class="loader err">${esc(e.message)}</div>`; return; }
    if (RK.team !== team) return;
  }
  const d = RK.data[team], shots = (RK.mode === "for" ? d.for : d.against).filter((s) => s[0] >= 0);
  const S = 8, W = 104 * S, Hh = 85 * S, X = (x) => (x + 2) * S, Y = (y) => (42.5 - y) * S;
  const col = RK.mode === "for" ? tcol(team) : "#f87171";
  // heat
  const cell = 6, heat = {};
  shots.forEach(([x, y]) => { const k = `${Math.floor(x / cell)},${Math.floor((y + 42.5) / cell)}`; heat[k] = (heat[k] || 0) + 1; });
  const hmax = Math.max(1, ...Object.values(heat));
  let svg = `<svg viewBox="0 0 ${W} ${Hh}" xmlns="http://www.w3.org/2000/svg" font-family="Inter,sans-serif">
    <defs><clipPath id="rc"><path d="M${X(0)} ${Y(42.5)} H${X(72)} A${28 * S} ${28 * S} 0 0 1 ${X(100)} ${Y(14.5)} V${Y(-14.5)} A${28 * S} ${28 * S} 0 0 1 ${X(72)} ${Y(-42.5)} H${X(0)} Z"/></clipPath>
    <radialGradient id="ice" cx="60%" cy="50%" r="80%"><stop offset="0" stop-color="#eef6ff"/><stop offset="1" stop-color="#cfe0f2"/></radialGradient></defs>
    <g clip-path="url(#rc)"><rect x="0" y="0" width="${W}" height="${Hh}" fill="url(#ice)"/>`;
  Object.entries(heat).forEach(([k, v]) => {
    const [cx, cy] = k.split(",").map(Number);
    svg += `<rect x="${X(cx * cell)}" y="${Y(cy * cell - 42.5 + cell)}" width="${cell * S}" height="${cell * S}" fill="${col}" opacity="${(0.08 + 0.5 * (v / hmax)).toFixed(2)}"/>`;
  });
  svg += `<line x1="${X(0)}" x2="${X(0)}" y1="0" y2="${Hh}" stroke="#e11d48" stroke-width="${S}"/>
    <line x1="${X(25)}" x2="${X(25)}" y1="0" y2="${Hh}" stroke="#2563eb" stroke-width="${S}"/>
    <line x1="${X(89)}" x2="${X(89)}" y1="0" y2="${Hh}" stroke="#e11d48" stroke-width="3"/>
    <path d="M${X(89)} ${Y(4)} A${6 * S} ${6 * S} 0 0 0 ${X(89)} ${Y(-4)}" fill="#93c5fd" opacity=".7" stroke="#e11d48" stroke-width="3"/>
    <rect x="${X(89)}" y="${Y(3)}" width="${3.3 * S}" height="${6 * S}" fill="none" stroke="#111" stroke-width="3"/>
    <circle cx="${X(69)}" cy="${Y(22)}" r="${15 * S}" fill="none" stroke="#e11d48" stroke-width="3"/><circle cx="${X(69)}" cy="${Y(-22)}" r="${15 * S}" fill="none" stroke="#e11d48" stroke-width="3"/>
    <circle cx="${X(69)}" cy="${Y(22)}" r="${S}" fill="#e11d48"/><circle cx="${X(69)}" cy="${Y(-22)}" r="${S}" fill="#e11d48"/>
    <circle cx="${X(20)}" cy="${Y(22)}" r="${S}" fill="#e11d48"/><circle cx="${X(20)}" cy="${Y(-22)}" r="${S}" fill="#e11d48"/>
    <path d="M${X(0)} ${Y(15)} A${15 * S} ${15 * S} 0 0 1 ${X(0)} ${Y(-15)}" fill="none" stroke="#2563eb" stroke-width="3"/>`;
  const dur = Math.min(3200, shots.length * 9);
  shots.forEach(([x, y, g], i) => {
    const delay = Math.round((i / Math.max(1, shots.length)) * dur);
    svg += g ? `<circle class="shot goal" style="animation-delay:${delay}ms,${delay}ms" cx="${X(x)}" cy="${Y(y)}" r="${1.5 * S}" fill="#fbbf24" stroke="#7c4a03" stroke-width="2"><title>Gól</title></circle>`
             : `<circle class="shot" style="animation-delay:${delay}ms" cx="${X(x)}" cy="${Y(y)}" r="${0.75 * S}" fill="#1e3a8a" opacity=".75"/>`;
  });
  svg += `</g><path d="M${X(0)} ${Y(42.5)} H${X(72)} A${28 * S} ${28 * S} 0 0 1 ${X(100)} ${Y(14.5)} V${Y(-14.5)} A${28 * S} ${28 * S} 0 0 1 ${X(72)} ${Y(-42.5)} H${X(0)}" fill="none" stroke="#9fb0c8" stroke-width="5"/>
    <text x="${X(93)}" y="${Y(-38)}" font-size="22" fill="#475569" text-anchor="middle" font-weight="700">BRANKA →</text></svg>`;
  box.innerHTML = svg;
  // zóny
  const dist = (x, y) => Math.hypot(89 - x, y);
  const zones = [["Blízko branky (do 6 m)", (s) => dist(s[0], s[1]) < 20], ["Střední vzdálenost (6–12 m)", (s) => dist(s[0], s[1]) >= 20 && dist(s[0], s[1]) < 40], ["Z dálky (přes 12 m)", (s) => dist(s[0], s[1]) >= 40]];
  const n = shots.length || 1, goals = shots.filter((s) => s[2]).length;
  $("rinkzones").innerHTML = `<div><span><b style="color:${col}">${shots.length}</b><br><small class="muted">${RK.mode === "for" ? "střel na branku" : "střel soupeřů"} za ${d.games} zápasů (${(shots.length / Math.max(1, d.games)).toFixed(1)} na zápas)</small></span></div>
    <div><span><b style="color:var(--gold)">${goals}</b><br><small class="muted">gólů · úspěšnost ${((goals / n) * 100).toFixed(1)} %</small></span></div>` +
    zones.map(([lab, f]) => { const z = shots.filter(f), zg = z.filter((s) => s[2]).length;
      return `<div><span>${lab}<br><small class="muted">${zg} gólů z ${z.length} střel (${z.length ? ((zg / z.length) * 100).toFixed(0) : 0} % úspěšnost)</small></span><b>${Math.round((z.length / n) * 100)} %</b></div>`; }).join("") +
    `<p class="help">Procenta vpravo = jaká část střel šla z dané vzdálenosti. Zblízka padá víc gólů, proto týmy, které se dostanou k brance, bývají nebezpečnější.</p>`;
}

// ===================================================================
//  Porovnání hráčů
// ===================================================================
const CMP = { ids: [] };
const CMP_COLORS = ["#38bdf8", "#a78bfa", "#fbbf24"];
const CMP_MET = [
  ["Šance na bod", (p) => p.pPoint, pc], ["Šance na gól", (p) => p.pGoal, pc], ["Oček. střely", (p) => p.expShots, (v) => num(v, 1)],
  ["Body na zápas (sezóna)", (p) => (p.gp ? p.pts / p.gp : 0), (v) => num(v)], ["Body v posl. 5", (p) => p.form5, (v) => v], ["Čas na ledě", (p) => p.toi, toi],
];
function initCompare() {
  const act = GAME.players.filter((p) => p.active);
  const tH = act.find((p) => p.team === GAME.home), tA = act.find((p) => p.team === GAME.away);
  CMP.ids = [tH?.id, tA?.id, ""].map((x) => x ?? "");
  const opts = (sel) => `<option value="">– nikdo –</option>` + act.map((p) => `<option value="${p.id}" ${p.id === sel ? "selected" : ""}>${esc(p.name)} (${p.team})</option>`).join("");
  $("cmpsel").innerHTML = CMP.ids.map((id, i) => `<select data-i="${i}" style="border-color:${CMP_COLORS[i]}">${opts(id)}</select>`).join("");
  $("cmpsel").querySelectorAll("select").forEach((s) => (s.onchange = () => { CMP.ids[+s.dataset.i] = s.value ? +s.value : ""; drawCompare(); }));
  drawCompare();
}
function drawCompare() {
  const act = GAME.players.filter((p) => p.active);
  const sel = CMP.ids.map((id, i) => ({ p: GAME.players.find((x) => x.id === id), c: CMP_COLORS[i] })).filter((x) => x.p);
  const maxes = CMP_MET.map(([, f]) => Math.max(0.0001, ...act.map(f)));
  const R = 150, C = 200, n = CMP_MET.length;
  const pt = (i, r) => [C + r * Math.sin((2 * Math.PI * i) / n), C - r * Math.cos((2 * Math.PI * i) / n)];
  let svg = `<svg viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg" font-family="Inter,sans-serif">`;
  [0.25, 0.5, 0.75, 1].forEach((k) => (svg += `<polygon points="${CMP_MET.map((_, i) => pt(i, R * k).join(",")).join(" ")}" fill="none" stroke="rgba(148,163,184,.2)"/>`));
  CMP_MET.forEach(([lab], i) => {
    const [x, y] = pt(i, R), [lx, ly] = pt(i, R + 26);
    svg += `<line x1="${C}" y1="${C}" x2="${x}" y2="${y}" stroke="rgba(148,163,184,.2)"/><text x="${lx}" y="${ly + 4}" font-size="12" fill="#b8c6dc" text-anchor="middle" font-weight="700">${lab.replace(" (sezóna)", "")}</text>`;
  });
  sel.forEach(({ p, c }, si) => {
    const pts = CMP_MET.map(([, f], i) => pt(i, R * Math.min(1, f(p) / maxes[i])).join(",")).join(" ");
    svg += `<polygon class="radar-poly" style="animation-delay:${si * 150}ms" points="${pts}" fill="${c}" fill-opacity=".22" stroke="${c}" stroke-width="3"/>`;
  });
  $("radar").innerHTML = svg + "</svg>";
  $("cmpbars").innerHTML = sel.length ? CMP_MET.map(([lab, f, fmt], mi) => `<div class="cmprow"><div class="lab">${lab}</div>` +
    sel.map(({ p, c }) => `<div class="r"><span class="b" title="${esc(p.name)}"><i data-w="${Math.min(100, (f(p) / maxes[mi]) * 100)}" style="background:${c}"></i></span><span class="v" style="color:${c}">${fmt(f(p))}</span></div>`).join("") + `</div>`).join("") +
    `<div class="legend">${sel.map(({ p, c }) => `<span><i style="background:${c}"></i>${esc(p.name)}</span>`).join("")}</div>`
    : `<p class="help">Vyber nahoře aspoň jednoho hráče.</p>`;
  requestAnimationFrame(() => setTimeout(() => $("cmpbars").querySelectorAll("i[data-w]").forEach((i) => (i.style.width = i.dataset.w + "%")), 40));
}

// ===================================================================
//  Můj tiket
// ===================================================================
const SLIP = { items: [], stake: 100 };
try { SLIP.items = JSON.parse(store.get("slip") || "[]"); SLIP.stake = +(store.get("slipstake") || 100); } catch (e) {}
function saveSlip() { store.set("slip", JSON.stringify(SLIP.items)); store.set("slipstake", String(SLIP.stake)); }
function addToSlip(it, btn) {
  const g = GAME;
  it.gid = g.id; it.match = `${g.home} – ${g.away}`; it.start = g.start;
  if (!(it.odd > 1)) it.odd = null;
  const i = SLIP.items.findIndex((x) => x.key === it.key);
  if (i >= 0) SLIP.items[i] = it; else SLIP.items.push(it);
  saveSlip(); renderSlip();
  $("slipbtn").classList.remove("bump"); void $("slipbtn").offsetWidth; $("slipbtn").classList.add("bump");
  if (btn) { btn.classList.add("done"); btn.textContent = "✓ na tiketu"; }
}
function slAdd(over, btn) {
  const c = slCfg(), line = +$("slrange").value;
  const p = c.normal ? 1 - ncdf(line, c.mean, c.sd) : 1 - poisCdf(Math.floor(line), c.mean);
  const label = $("slkind").querySelector("button.on").textContent;
  addToSlip({ key: `${GAME.id}:sl:${SL.kind}:${over ? "o" : "u"}${line}`, label: `${label} – ${over ? "přes" : "pod"} ${line}`, p: over ? p : 1 - p, odd: null }, btn);
}
function renderSlip() {
  const it = SLIP.items;
  $("slipcnt").textContent = it.length;
  if (!it.length) {
    $("slipitems").innerHTML = `<p class="help">Tiket je prázdný. V detailu zápasu klikni u sázky na <b class="addbtn" style="cursor:default">+ tiket</b> a přidá se sem.
      Pak sem napiš kurzy z Tipsportu a uvidíš celkový kurz, šanci podle modelu a možnou výhru.</p>`;
    $("slipsum").innerHTML = ""; return;
  }
  $("slipitems").innerHTML = `<p class="help" style="margin:0 0 6px">Ke každé sázce napiš kurz z Tipsportu. ✕ sázku odebere.</p>` + it.map((x, i) => `<div class="sitem">
    <div><b>${esc(x.label)}</b><small>${esc(x.match)} · model ${pc(x.p)} · férový kurz ${fair(x.p)}</small></div>
    <input type="number" step="0.01" min="1" placeholder="kurz" value="${x.odd ?? ""}" data-i="${i}">
    <button data-del="${i}" title="Odebrat">✕</button></div>`).join("");
  const pAll = it.reduce((a, x) => a * x.p, 1);
  const haveOdds = it.every((x) => x.odd > 1);
  const oAll = haveOdds ? it.reduce((a, x) => a * x.odd, 1) : null;
  const games = new Set(it.map((x) => x.gid));
  const ev = haveOdds ? pAll * oAll - 1 : null;
  $("slipsum").innerHTML = `
    <div class="row"><span>Šance, že vyjde všechno (model)</span><b style="color:var(--gold)">${(pAll * 100).toFixed(pAll < 0.1 ? 1 : 0)} %</b></div>
    <div class="row"><span>Férový celkový kurz</span><b>${fair(pAll)}</b></div>
    <div class="row"><span>Celkový kurz Tipsport</span><b>${oAll ? oAll.toFixed(2) : "doplň kurzy"}</b></div>
    <div class="row"><span>Vyplatí se? (hodnota)</span><b>${ev != null ? evHtml(pAll, oAll) : "–"}</b></div>
    <div class="row"><span>Vklad (Kč)</span><input type="number" id="slipstake" min="1" value="${SLIP.stake}" style="width:110px;text-align:right"></div>
    <div class="row"><span>Možná výhra</span><span class="big">${oAll ? Math.round(SLIP.stake * oAll).toLocaleString("cs-CZ") + " Kč" : "–"}</span></div>
    ${it.length > games.size ? `<p class="help" style="color:var(--orange)">⚠️ Máš víc sázek ze stejného zápasu. Ty spolu souvisí (např. víc gólů = víc bodů hráčů), takže šance podle modelu je jen přibližná.</p>` : ""}
    <p class="help">Šance, že vyjde celý tiket, je součin šancí jednotlivých sázek – proto s každou další sázkou rychle klesá. Sázej jen to, co si můžeš dovolit prohrát.</p>
    <button class="btn" id="slipclear" style="margin-top:6px">🗑 Vyprázdnit tiket</button>`;
  $("slipitems").querySelectorAll("input").forEach((inp) => (inp.onchange = () => { SLIP.items[+inp.dataset.i].odd = parseFloat(inp.value) > 1 ? parseFloat(inp.value) : null; saveSlip(); renderSlip(); }));
  $("slipitems").querySelectorAll("[data-del]").forEach((b) => (b.onclick = () => { SLIP.items.splice(+b.dataset.del, 1); saveSlip(); renderSlip(); }));
  $("slipstake").onchange = (e) => { SLIP.stake = Math.max(1, +e.target.value || 100); saveSlip(); renderSlip(); };
  $("slipclear").onclick = () => { SLIP.items = []; saveSlip(); renderSlip(); };
}
$("slipbtn").onclick = () => { $("slip").classList.remove("hidden"); $("slipbtn").classList.add("hidden"); renderSlip(); };
$("slipclose").onclick = () => { $("slip").classList.add("hidden"); $("slipbtn").classList.remove("hidden"); };
renderSlip();

// ===================================================================
//  LIVE – 3D hřiště
// ===================================================================
const LV = { id: null, data: null, timer: null, shown: 0, replay: null, filt: { shots: true, goals: true, missed: false, blocked: false, hits: false, pens: true, faceoff: false }, per: "all", d3: true };
const EVCZ = { "faceoff": "Vhazování", "shot-on-goal": "Střela na branku", "goal": "GÓL", "missed-shot": "Střela mimo", "blocked-shot": "Zblokovaná střela",
  "hit": "Hit (bodyček)", "penalty": "Vyloučení", "takeaway": "Získaný puk", "giveaway": "Ztracený puk" };
const EVICO = { "faceoff": "⚪", "shot-on-goal": "🎯", "goal": "🚨", "missed-shot": "↗️", "blocked-shot": "🛡️", "hit": "💥", "penalty": "⛔", "takeaway": "🫳", "giveaway": "❌" };
const PENCZ = { tripping: "podražení", hooking: "hákování", slashing: "sekání", "high-sticking": "vysoká hokejka", "high-sticking-double-minor": "vysoká hokejka (2+2)",
  interference: "bránění ve hře", roughing: "hrubost", holding: "držení", "holding-the-stick": "držení hokejky", "cross-checking": "krosček", fighting: "bitka",
  "delay-of-game": "zdržování hry", "delay-of-game-puck-over-glass": "puk mimo hřiště", "too-many-men-on-the-ice": "příliš mnoho hráčů na ledě",
  "unsportsmanlike-conduct": "nesportovní chování", boarding: "napadení na hrazení", elbowing: "loktem", "embellishment": "simulování", charging: "napadení",
  "goalie-interference": "bránění brankáři", misconduct: "osobní trest", "game-misconduct": "osobní trest do konce utkání", "instigator": "vyprovokování bitky", kneeing: "kolenem" };
const PERCZ = (n, t) => (t === "OT" ? "Prodloužení" : t === "SO" ? "Nájezdy" : `${n}. třetina`);
function stopLive() { clearInterval(LV.timer); LV.timer = null; if (LV.replay) { cancelAnimationFrame(LV.replay.raf); clearTimeout(LV.replay.t); LV.replay = null; } }
async function openLive(id) {
  const el = $("view-live");
  stopLive();
  el.innerHTML = `<div class="sechead" style="margin-top:30px"><span class="no"><span class="livedot"></span></span><h2>Live hřiště</h2><span class="hint">zápasy dnes a včera</span></div>
    <p class="help" style="margin-top:-6px">Vyber zápas. <b>Červená tečka = právě se hraje</b> (stránka se sama obnovuje každých 8 vteřin). U odehraných zápasů si můžeš pustit <b>záznam</b> – puk projede celý zápas akci po akci.
      U zápasů, které ještě nezačaly, uvidíš odpočet.</p>
    <div class="lgames" id="lgames"><div class="loader" style="padding:20px">Načítám zápasy…</div></div><div id="lmain"></div>`;
  let list;
  try { list = (await getJSON("/model/livegames")).games; } catch (e) { $("lgames").innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }
  const stLabel = (g) => (["LIVE", "CRIT"].includes(g.state) ? `<span class="livedot"></span> ${g.period ? PERCZ(g.period, g.periodType) : "ŽIVĚ"} · ${g.clock?.inIntermission ? "přestávka" : g.clock?.timeRemaining || ""}`
    : ["FINAL", "OFF"].includes(g.state) ? `konec${g.periodType && g.periodType !== "REG" ? " (" + g.periodType + ")" : ""} · záznam` : `🕒 ${fmtDay(g.start)} ${fmtTime(g.start)}`);
  $("lgames").innerHTML = list.length ? list.map((g) => `<div class="lg ${g.id === id ? "on" : ""}" data-id="${g.id}">
    <div class="r"><img src="${esc(g.homeLogo)}" alt=""><span>${esc(g.home)}</span><b>${g.score[0] ?? ""}</b></div>
    <div class="r"><img src="${esc(g.awayLogo)}" alt=""><span>${esc(g.away)}</span><b>${g.score[1] ?? ""}</b></div><small>${stLabel(g)}</small></div>`).join("")
    : `<p class="help">Dnes ani včera nejsou žádné zápasy.</p>`;
  $("lgames").querySelectorAll(".lg").forEach((c) => (c.onclick = () => (location.hash = "live-" + c.dataset.id)));
  if (!id) {
    const first = list.find((g) => ["LIVE", "CRIT"].includes(g.state)) || list.find((g) => ["FINAL", "OFF"].includes(g.state)) || list[0];
    if (first) { history.replaceState(null, "", "#live-" + first.id); id = first.id; $("lgames").querySelector(`[data-id="${id}"]`)?.classList.add("on"); }
    else return;
  }
  LV.id = id; LV.data = null; LV.shown = 0;
  $("lmain").innerHTML = `<div class="loader"><div class="spinner"></div>Načítám zápas…</div>`;
  await liveFetch(true);
  if (LV.data && ["LIVE", "CRIT", "PRE"].includes(LV.data.state)) LV.timer = setInterval(() => liveFetch(false), 8000);
}
async function liveFetch(first) {
  const id = LV.id;
  let d;
  try { d = await getJSON("/model/live?id=" + id); } catch (e) { if (first) $("lmain").innerHTML = `<div class="loader err">${esc(e.message)}</div>`; return; }
  if (LV.id !== id) return;
  const prevGoals = LV.data ? (LV.data.home.score ?? 0) + (LV.data.away.score ?? 0) : null;
  LV.data = d;
  if (first) renderLive();
  else { updateLive(prevGoals); }
}
function sitInfo(code, d) {
  if (!code || code.length !== 4) return "";
  const ag = +code[0], as = +code[1], hs = +code[2], hg = +code[3];
  let t = "";
  if (hs > as) t = `<span class="badge pp">⚡ Přesilovka ${d.home.abbr} ${hs} na ${as}</span>`;
  else if (as > hs) t = `<span class="badge pp">⚡ Přesilovka ${d.away.abbr} ${as} na ${hs}</span>`;
  if (!hg && ["LIVE", "CRIT"].includes(d.state)) t += ` <span class="badge en">🥅 ${d.home.abbr} bez brankáře</span>`;
  if (!ag && ["LIVE", "CRIT"].includes(d.state)) t += ` <span class="badge en">🥅 ${d.away.abbr} bez brankáře</span>`;
  return t;
}
function elapsedMin(d) {
  if (["FINAL", "OFF"].includes(d.state)) return 60;
  if (!d.period) return 0;
  const rem = d.clock?.secondsRemaining ?? 1200;
  return Math.min(65, (Math.min(d.period, 3) - 1) * 20 + (d.period > 3 ? 20 + (300 - rem) / 60 : (1200 - rem) / 60));
}
function rinkSvg(d) {
  const S = 5, W = 200 * S, H = 85 * S, X = (x) => (x + 100) * S, Y = (y) => (42.5 - y) * S;
  const homeSide = d.homeSide || "left";
  const lbl = (side, ab) => `<text x="${side === "left" ? X(-80) : X(80)}" y="${Y(-37)}" font-size="22" font-weight="800" fill="#334155" text-anchor="middle">${ab}</text>`;
  return `<svg viewBox="-10 -10 ${W + 20} ${H + 20}" xmlns="http://www.w3.org/2000/svg" font-family="Inter,sans-serif">
    <defs><radialGradient id="ice2" cx="50%" cy="50%" r="70%"><stop offset="0" stop-color="#f4f9ff"/><stop offset="1" stop-color="#cddcec"/></radialGradient>
      <clipPath id="rclip"><rect x="0" y="0" width="${W}" height="${H}" rx="${28 * S}"/></clipPath>
      <radialGradient id="puckg" cx="40%" cy="35%" r="70%"><stop offset="0" stop-color="#4b5563"/><stop offset="1" stop-color="#030712"/></radialGradient></defs>
    <rect x="-8" y="-8" width="${W + 16}" height="${H + 16}" rx="${30 * S}" fill="#0f172a" stroke="#64748b" stroke-width="6"/>
    <g clip-path="url(#rclip)"><rect x="0" y="0" width="${W}" height="${H}" fill="url(#ice2)"/>
      <image href="${esc(d.home.logo)}" x="${X(0) - 70}" y="${Y(0) - 70}" width="140" height="140" opacity=".12"/>
      <line x1="${X(0)}" x2="${X(0)}" y1="0" y2="${H}" stroke="#e11d48" stroke-width="${S * 1.2}"/>
      <line x1="${X(-25)}" x2="${X(-25)}" y1="0" y2="${H}" stroke="#2563eb" stroke-width="${S * 1.2}"/><line x1="${X(25)}" x2="${X(25)}" y1="0" y2="${H}" stroke="#2563eb" stroke-width="${S * 1.2}"/>
      <line x1="${X(-89)}" x2="${X(-89)}" y1="0" y2="${H}" stroke="#e11d48" stroke-width="3"/><line x1="${X(89)}" x2="${X(89)}" y1="0" y2="${H}" stroke="#e11d48" stroke-width="3"/>
      <circle cx="${X(0)}" cy="${Y(0)}" r="${15 * S}" fill="none" stroke="#2563eb" stroke-width="3"/><circle cx="${X(0)}" cy="${Y(0)}" r="${S}" fill="#2563eb"/>
      ${[-69, 69].map((fx) => [22, -22].map((fy) => `<circle cx="${X(fx)}" cy="${Y(fy)}" r="${15 * S}" fill="none" stroke="#e11d48" stroke-width="3"/><circle cx="${X(fx)}" cy="${Y(fy)}" r="${S}" fill="#e11d48"/>`).join("")).join("")}
      ${[-20, 20].map((fx) => [22, -22].map((fy) => `<circle cx="${X(fx)}" cy="${Y(fy)}" r="${S}" fill="#e11d48"/>`).join("")).join("")}
      <path d="M${X(-89)} ${Y(4)} A${6 * S} ${6 * S} 0 0 1 ${X(-89)} ${Y(-4)}" fill="#93c5fd" opacity=".8" stroke="#e11d48" stroke-width="3"/>
      <path d="M${X(89)} ${Y(4)} A${6 * S} ${6 * S} 0 0 0 ${X(89)} ${Y(-4)}" fill="#93c5fd" opacity=".8" stroke="#e11d48" stroke-width="3"/>
      <rect x="${X(-92.3)}" y="${Y(3)}" width="${3.3 * S}" height="${6 * S}" fill="none" stroke="#111" stroke-width="3"/>
      <rect x="${X(89)}" y="${Y(3)}" width="${3.3 * S}" height="${6 * S}" fill="none" stroke="#111" stroke-width="3"/>
      ${lbl(homeSide, "branka " + d.home.abbr)}${lbl(homeSide === "left" ? "right" : "left", "branka " + d.away.abbr)}
      <g id="evlayer"></g>
      <g id="puck" style="transform:translate(${X(0)}px,${Y(0)}px)"><ellipse cx="0" cy="4" rx="13" ry="6" fill="rgba(0,0,0,.25)"/><circle r="10" fill="url(#puckg)" stroke="#000" stroke-width="2"/></g>
    </g></svg>`;
}
function evVisible(e) {
  const f = LV.filt;
  if (LV.per !== "all" && String(e.per) !== LV.per) return false;
  return (e.type === "shot-on-goal" && f.shots) || (e.type === "goal" && f.goals) || (e.type === "missed-shot" && f.missed) ||
    (e.type === "blocked-shot" && f.blocked) || (e.type === "hit" && f.hits) || (e.type === "penalty" && f.pens) || (e.type === "faceoff" && f.faceoff);
}
function evMark(e, d, ripple) {
  if (e.x == null || e.y == null) return "";
  const S = 5, x = (e.x + 100) * S, y = (42.5 - e.y) * S, c = e.team === d.home.abbr ? "#0ea5e9" : "#8b5cf6";
  const tip = `<title>${PERCZ(e.per)} ${e.time} · ${EVCZ[e.type]} · ${e.who ? e.who.name : ""} (${e.team || ""})</title>`;
  let m = "";
  if (e.type === "goal") m = `<g class="ev goal"><circle cx="${x}" cy="${y}" r="14" fill="#fbbf24" stroke="#78350f" stroke-width="3"/><text x="${x}" y="${y + 5}" font-size="15" font-weight="900" text-anchor="middle" fill="#78350f">G</text>${tip}</g>`;
  else if (e.type === "shot-on-goal") m = `<circle class="ev" cx="${x}" cy="${y}" r="7" fill="${c}" stroke="#fff" stroke-width="2">${tip}</circle>`;
  else if (e.type === "missed-shot") m = `<circle class="ev" cx="${x}" cy="${y}" r="6" fill="none" stroke="${c}" stroke-width="3">${tip}</circle>`;
  else if (e.type === "blocked-shot") m = `<g class="ev"><path d="M${x - 6} ${y - 6} L${x + 6} ${y + 6} M${x + 6} ${y - 6} L${x - 6} ${y + 6}" stroke="${c}" stroke-width="3.5"/>${tip}</g>`;
  else if (e.type === "hit") m = `<rect class="ev" x="${x - 6}" y="${y - 6}" width="12" height="12" transform="rotate(45 ${x} ${y})" fill="#64748b" opacity=".8">${tip}</rect>`;
  else if (e.type === "penalty") m = `<g class="ev"><rect x="${x - 10}" y="${y - 10}" width="20" height="20" rx="4" fill="#ef4444"/><text x="${x}" y="${y + 6}" font-size="16" font-weight="900" text-anchor="middle" fill="#fff">!</text>${tip}</g>`;
  else if (e.type === "faceoff") m = `<circle class="ev" cx="${x}" cy="${y}" r="4" fill="#94a3b8">${tip}</circle>`;
  if (ripple) m += `<circle class="ripple" cx="${x}" cy="${y}" r="12" fill="none" stroke="${e.type === "goal" ? "#fbbf24" : c}" stroke-width="3"/>`;
  return m;
}
function movePuck(e) {
  if (!e || e.x == null) return;
  const p = document.getElementById("puck");
  if (p) p.style.transform = `translate(${(e.x + 100) * 5}px,${(42.5 - e.y) * 5}px)`;
}
function evText(e) {
  const w = e.who ? `<b>${esc(e.who.name)}</b>${e.who.num != null ? " #" + e.who.num : ""}` : "";
  let t = `${EVCZ[e.type]} ${e.team ? `<span class="muted">${e.team}</span>` : ""} ${w}`;
  if (e.type === "goal") t = `<b style="color:var(--gold)">🚨 GÓL ${e.team}!</b> ${w}${e.extra?.length ? `<small>asistence: ${e.extra.map((a) => esc(a.name)).join(", ")}</small>` : "<small>bez asistence</small>"}${e.score ? `<small>stav ${e.score[0]}:${e.score[1]}${e.clip ? ` · <a href="${esc(e.clip)}" target="_blank" rel="noopener">▶ video</a>` : ""}</small>` : ""}`;
  if (e.type === "penalty") t += `<small>${PENCZ[e.desc] || esc(e.desc || "")} · ${e.dur || 2} min</small>`;
  if (e.type === "hit" && e.extra?.length) t += `<small>do hráče ${esc(e.extra[0].name)}</small>`;
  return t;
}
function liveCounts(d, evs) {
  const c = { [d.home.abbr]: { g: 0, sog: 0, pen: 0, hits: 0 }, [d.away.abbr]: { g: 0, sog: 0, pen: 0, hits: 0 } };
  evs.forEach((e) => { const t = c[e.team]; if (!t) return;
    if (e.type === "goal") { t.g++; t.sog++; } else if (e.type === "shot-on-goal") t.sog++; else if (e.type === "penalty") t.pen++; else if (e.type === "hit") t.hits++; });
  return c;
}
function renderLive() {
  const d = LV.data, H = d.home.abbr, A = d.away.abbr;
  const fut = ["FUT", "PRE"].includes(d.state), fin = ["FINAL", "OFF"].includes(d.state);
  $("lmain").innerHTML = `
    <div class="lscore" style="--hc:${tcol(H)};--ac:${tcol(A)}">
      <div class="t"><img src="${esc(d.home.logo)}" alt=""><div><b>${esc(d.home.name)}</b><small>doma · střely <span id="lsogh">${d.home.sog ?? 0}</span></small></div></div>
      <div class="mid"><div class="clk" id="lclk"></div><div class="sc" id="lsc">${d.home.score ?? 0} : ${d.away.score ?? 0}</div><div id="lsit"></div></div>
      <div class="t r"><img src="${esc(d.away.logo)}" alt=""><div><b>${esc(d.away.name)}</b><small>venku · střely <span id="lsoga">${d.away.sog ?? 0}</span></small></div></div>
    </div>
    ${fut ? `<div class="card" style="margin-top:14px;text-align:center"><h3 class="display" style="font-size:40px;margin:6px 0">Zápas ještě nezačal</h3>
      <div class="countdown" data-start="${d.start}" style="font-size:20px"></div><p class="help">Jakmile začne, uvidíš tady živě puk, střely a vyloučení. Mezitím se podívej na <a href="#zapas-${d.id}">předpověď zápasu</a>.</p></div>` : ""}
    <div class="lfilters">
      <button class="btn" id="l3d">${LV.d3 ? "🔲 Přepnout na 2D" : "🧊 Přepnout na 3D"}</button>
      ${fin ? `<button class="btn primary" id="lreplay">▶ Přehrát záznam zápasu</button><select id="lspeed"><option value="1">rychlost 1×</option><option value="4" selected>rychlost 4×</option><option value="15">rychlost 15×</option></select>` : ""}
      <select id="lper"><option value="all">všechny třetiny</option><option value="1">1. třetina</option><option value="2">2. třetina</option><option value="3">3. třetina</option><option value="4">prodloužení</option></select>
      ${[["shots", "🎯 střely"], ["goals", "🚨 góly"], ["missed", "↗️ mimo"], ["blocked", "🛡️ zblokované"], ["hits", "💥 hity"], ["pens", "⛔ vyloučení"], ["faceoff", "⚪ vhazování"]]
        .map(([k, l]) => `<label><input type="checkbox" data-f="${k}" ${LV.filt[k] ? "checked" : ""}> ${l}</label>`).join("")}
    </div>
    <p class="help">Hřiště ukazuje akce zápasu přesně tam, kde se staly. <b>Černý puk</b> se přesouvá na místo poslední akce. Barevná tečka = střela na branku (barva týmu), <b>zlaté G</b> = gól,
      prázdný kroužek = střela mimo, křížek = zblokovaná střela, šedý kosočtverec = hit, <b>červené !</b> = vyloučení. Najeď na značku myší a uvidíš, kdo to byl.
      <b style="color:#0ea5e9">Modrá = ${H} (domácí)</b>, <b style="color:#8b5cf6">fialová = ${A} (hosté)</b>. U spodního okraje je napsané, čí je která branka (v každé třetině se strany mění). Zaškrtávátky výše si vybereš, co chceš vidět.</p>
    <div class="lgrid">
      <div><div class="rinkstage ${LV.d3 ? "d3" : ""}" id="rstage"><div class="tilt3d">${rinkSvg(d)}</div></div>
        <div class="sechead" style="margin-top:20px"><h2 style="font-size:34px">Tempo zápasu</h2><span class="hint">skutečnost vs. předpověď</span></div><div class="pace" id="lpace"></div>
        <p class="help"><b>Tempo</b> = kolik by toho padlo za celý zápas, kdyby se hrálo stejně jako doteď. Porovnej s tím, co čekal model před zápasem.</p>
        <div class="sechead" style="margin-top:20px"><h2 style="font-size:34px">Po třetinách</h2></div><div class="tablewrap"><table class="t" id="lper-t"></table></div>
        <p class="help">G = góly, Střely = na branku, Mimo = střely vedle, Zblok. = střely, které soupeř zblokoval, Hity = bodyčeky, Vyl. = počet vyloučení (v závorce trestné minuty).</p>
      </div>
      <div><div class="card"><b>🚨 Góly zápasu</b><p class="help" style="margin:4px 0 8px"><b>▶ Video</b> otevře záznam gólu na NHL.com. <b>AI rozbor</b> vysvětlí, jak gól padl (odkud se střílelo, jestli to byla dorážka, protiútok nebo přesilovka) – AI video nevidí, popisuje gól podle dat zápasu.</p><div id="lgoals"></div></div>
        <div class="card" style="margin-top:14px"><b>📰 Průběh zápasu</b><p class="help" style="margin:4px 0 8px">Nejnovější akce nahoře.</p><div class="feed" id="lfeed"></div></div></div>
    </div>
    <div class="sechead"><h2 style="font-size:40px">Hráči</h2><span class="hint">seřazeno podle střel na branku</span></div>
    <p class="help" style="margin-top:-6px"><b>Střely</b> = střely na branku v tomto zápase, za lomítkem kolik jich čekal model (✅ = už splnil). <b>TM</b> = trestné minuty, <b>+/-</b> = plus/minus, <b>TOI</b> = čas na ledě.</p>
    <div class="grid g2" id="lplayers"></div>`;
  $("l3d").onclick = () => { LV.d3 = !LV.d3; $("rstage").classList.toggle("d3", LV.d3); $("l3d").textContent = LV.d3 ? "🔲 Přepnout na 2D" : "🧊 Přepnout na 3D"; };
  $("lper").onchange = (e) => { LV.per = e.target.value; drawEvents(LV.replay ? LV.replay.i : d.events.length, false); };
  document.querySelectorAll("#lmain [data-f]").forEach((c) => (c.onchange = () => { LV.filt[c.dataset.f] = c.checked; drawEvents(LV.replay ? LV.replay.i : d.events.length, false); }));
  if (fin) $("lreplay").onclick = startReplay;
  LV.shown = d.events.length;
  drawEvents(d.events.length, false);
  movePuck([...d.events].reverse().find((e) => e.x != null));
  updatePanels(d.events.length);
  countdowns();
}
function drawEvents(upto, animateLast) {
  const d = LV.data, layer = document.getElementById("evlayer");
  if (!layer) return;
  const evs = d.events.slice(0, upto).filter(evVisible);
  layer.innerHTML = evs.map((e, i) => evMark(e, d, animateLast && i === evs.length - 1)).join("");
}
function updatePanels(upto) {
  const d = LV.data, H = d.home.abbr, A = d.away.abbr, evs = d.events.slice(0, upto);
  const replay = !!LV.replay;
  const c = liveCounts(d, evs);
  const last = evs[evs.length - 1];
  $("lsc").textContent = replay ? `${c[H].g} : ${c[A].g}` : `${d.home.score ?? 0} : ${d.away.score ?? 0}`;
  $("lsogh").textContent = replay ? c[H].sog : d.home.sog ?? c[H].sog;
  $("lsoga").textContent = replay ? c[A].sog : d.away.sog ?? c[A].sog;
  const fin = ["FINAL", "OFF"].includes(d.state);
  $("lclk").innerHTML = replay ? `▶ záznam · ${last ? PERCZ(last.per) + " · " + last.time : ""}`
    : fin ? `Konec zápasu${d.end && d.end !== "REG" ? " (" + (d.end === "SO" ? "po nájezdech" : "po prodloužení") + ")" : ""}` : d.period ? `<span class="livedot"></span> ${PERCZ(d.period, d.periodType)} · ${d.clock?.inIntermission ? "přestávka" : (d.clock?.timeRemaining || "") + " do konce"}` : "";
  $("lsit").innerHTML = replay ? sitInfo(last?.sit, { ...d, state: "LIVE" }) : sitInfo(d.situation, d);
  const goalsEv = evs.filter((e) => e.type === "goal");
  const gEl = $("lgoals");
  if (gEl && gEl.dataset.n !== String(goalsEv.length)) {
    gEl.dataset.n = goalsEv.length;
    gEl.innerHTML = goalsEv.length ? goalsEv.map((e) => `<div class="goalrow" data-eid="${e.id}"><span class="t">${e.score ? e.score[0] + ":" + e.score[1] : ""}</span>
      <span><b>${e.who ? `<span class="pname-link" data-player="${e.who.id}:${e.team}">${esc(e.who.name)}</span>` : ""}</b> <span class="muted">${e.team} · ${PERCZ(e.per)} ${e.time}</span>
        <small class="muted" style="display:block">${e.extra?.length ? "asistence: " + e.extra.map((a) => esc(a.name)).join(", ") : "bez asistence"}${e.shotType ? " · střela: " + esc(SHOTCZ[e.shotType] || e.shotType) : ""}</small></span>
      <span class="acts">${e.clip ? `<a class="addbtn" href="${esc(e.clip)}" target="_blank" rel="noopener" style="text-decoration:none">▶ Video</a>` : ""}<button class="addbtn" data-ai="${e.id}">🤖 AI rozbor</button></span></div>`).join("")
      : `<p class="help">Zatím žádný gól.</p>`;
    gEl.querySelectorAll("[data-ai]").forEach((b) => (b.onclick = () => goalAI(+b.dataset.ai, b)));
  }
  $("lfeed").innerHTML = evs.filter((e) => e.type !== "faceoff").slice(-40).reverse().map((e) =>
    `<div class="fe ${e.type === "goal" ? "goal" : ""}"><span class="tm">${e.per}. ${e.time}</span><span>${EVICO[e.type]}</span><span>${evText(e)}</span></div>`).join("") || `<p class="help">Zatím žádné akce.</p>`;
  // třetiny
  const per = {};
  evs.forEach((e) => { if (!e.team) return; const s = ((per[e.per] = per[e.per] || {})[e.team] = per[e.per][e.team] || { g: 0, sog: 0, miss: 0, blk: 0, hits: 0, pen: 0, pim: 0 });
    if (e.type === "goal") { s.g++; s.sog++; } else if (e.type === "shot-on-goal") s.sog++; else if (e.type === "missed-shot") s.miss++; else if (e.type === "blocked-shot") s.blk++;
    else if (e.type === "hit") s.hits++; else if (e.type === "penalty") { s.pen++; s.pim += e.dur || 2; } });
  const pers = Object.keys(per).map(Number).sort();
  const z = { g: 0, sog: 0, miss: 0, blk: 0, hits: 0, pen: 0, pim: 0 };
  const cell = (s) => `<td><b>${s.g}</b></td><td>${s.sog}</td><td>${s.miss}</td><td>${s.blk}</td><td>${s.hits}</td><td>${s.pen} (${s.pim})</td>`;
  const tot = (t) => pers.reduce((a, p) => { const s = per[p][t] || z; Object.keys(a).forEach((k) => (a[k] += s[k])); return a; }, { ...z });
  $("lper-t").innerHTML = `<thead><tr><th class="left">Třetina</th><th class="left">Tým</th><th>G</th><th>Střely</th><th>Mimo</th><th>Zblok.</th><th>Hity</th><th>Vyl. (TM)</th></tr></thead><tbody>` +
    pers.map((p) => [H, A].map((t, i) => `<tr><td class="left">${i ? "" : PERCZ(p, p > 3 ? "OT" : "REG")}</td><td class="left" style="color:${i ? "var(--away)" : "var(--home)"};font-weight:800">${t}</td>${cell(per[p][t] || z)}</tr>`).join("")).join("") +
    [H, A].map((t, i) => `<tr style="background:rgba(255,255,255,.04)"><td class="left">${i ? "" : "<b>Celkem</b>"}</td><td class="left" style="font-weight:800">${t}</td>${cell(tot(t))}</tr>`).join("") + "</tbody>";
  // tempo
  const pre = d.pre, em = replay ? (last ? Math.min(60, (last.per - 1) * 20 + (+last.time.split(":")[0]) + (+last.time.split(":")[1]) / 60) : 0) : elapsedMin(d);
  const pace = (v) => (em > 2 ? (v * 60) / em : null);
  const shotsNow = c[H].sog + c[A].sog, pensNow = c[H].pen + c[A].pen, goalsNow = c[H].g + c[A].g;
  const box = (lab, now, exp) => `<div><small>${lab}</small><b>${now}</b><small>${pace(now) != null && em < 60 ? `tempo na <b style="font-size:15px;display:inline;font-family:Inter">${pace(now).toFixed(0)}</b> za zápas · ` : ""}${exp != null ? `model čekal ${exp.toFixed(1)}` : ""}</small></div>`;
  $("lpace").innerHTML = box("Střely na branku celkem", shotsNow, pre?.shots.total) + box("Vyloučení celkem", pensNow, pre?.pens.total) + box("Góly celkem", goalsNow, pre?.goals.total) +
    `<div><small>Odehráno</small><b>${Math.round(em)}′</b><small>z 60 minut</small></div>`;
  // hráči
  if (!replay) $("lplayers").innerHTML = [H, A].map((t) => {
    const p = d.players[t] || { skaters: [], goalies: [] };
    const rows = [...p.skaters].sort((a, b) => b.sog - a.sog || b.pts - a.pts);
    return `<div class="card"><b style="color:${t === H ? "var(--home)" : "var(--away)"};font-size:18px">${t}</b>
      <div class="tablewrap" style="margin-top:8px;background:none"><table class="t"><thead><tr><th class="left">#</th><th class="left">Hráč</th><th>G</th><th>A</th><th>Střely</th><th>TM</th><th>Hity</th><th>Bloky</th><th>+/-</th><th>TOI</th></tr></thead><tbody>
      ${rows.map((r) => { const ex = pre?.players?.[r.id]?.expShots; return `<tr><td class="left pos">${r.num ?? ""}</td><td class="left"><b class="pname-link" data-player="${r.id}:${t}">${esc(r.name)}</b>${r.pts ? ` <span class="pill hot">${r.pts} b.</span>` : ""}</td>
        <td>${r.g}</td><td>${r.a}</td><td class="gold">${r.sog}${ex != null ? `<small class="muted">/${ex.toFixed(1)}</small>${r.sog >= ex ? " ✅" : ""}` : ""}</td><td>${r.pim}</td><td>${r.hits}</td><td>${r.blk}</td>
        <td>${r.pm > 0 ? "+" + r.pm : r.pm}</td><td>${toi(r.toi)}</td></tr>`; }).join("")}</tbody></table></div>
      ${p.goalies.map((g) => `<p class="help">🧤 <b>${esc(g.name)}</b>: ${g.sv} zákroků z ${g.sa} střel (${g.sa ? ((g.sv / g.sa) * 100).toFixed(1) : "–"} %), ${g.ga} gólů</p>`).join("")}</div>`;
  }).join("");
}
const SHOTCZ = { wrist: "zápěstím", snap: "švihem", slap: "golfák", backhand: "bekhend", "tip-in": "teč", deflected: "tečovaná", "wrap-around": "zpoza branky", poke: "šťouchnutí", bat: "z voleje", cradle: "lakros", "between-legs": "mezi nohama" };
async function goalAI(eid, btn) {
  const row = btn.closest(".goalrow");
  let out = row.querySelector(".airesp");
  if (!out) { out = document.createElement("div"); out.className = "airesp"; row.appendChild(out); }
  out.innerHTML = `<span class="typing"><span></span><span></span><span></span></span> AI rozebírá gól…`;
  btn.disabled = true;
  try {
    const r = await fetch("/model/goalai", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ gameId: LV.id, eventId: eid }) });
    const j = await r.json();
    out.innerHTML = j.reply ? md(j.reply) : `<span class="err">${esc(j.error || "AI neodpověděla.")}</span>`;
  } catch (e) { out.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  btn.disabled = false;
}
function updateLive(prevGoals) {
  const d = LV.data;
  const newEvs = d.events.slice(LV.shown);
  LV.shown = d.events.length;
  const goals = (d.home.score ?? 0) + (d.away.score ?? 0);
  if (prevGoals != null && goals > prevGoals) { const sc = $("lsc"); sc.classList.remove("flash"); void sc.offsetWidth; sc.classList.add("flash"); }
  let i = 0;
  const step = () => {
    if (!$("evlayer")) return;
    const upto = d.events.length - newEvs.length + i + 1;
    if (i < newEvs.length) { drawEvents(upto, evVisible(newEvs[i])); movePuck(newEvs[i]); i++; setTimeout(step, 700); }
  };
  if (newEvs.length) step(); else drawEvents(d.events.length, false);
  updatePanels(d.events.length);
  if (!["LIVE", "CRIT", "PRE"].includes(d.state)) stopLive();
}
function startReplay() {
  const d = LV.data;
  if (LV.replay) { clearTimeout(LV.replay.t); LV.replay = null; $("lreplay").textContent = "▶ Přehrát záznam zápasu"; drawEvents(d.events.length, false); updatePanels(d.events.length); return; }
  LV.replay = { i: 0, t: null };
  $("lreplay").textContent = "⏹ Zastavit záznam";
  const tick = () => {
    if (!LV.replay || !$("evlayer")) return;
    const sp = +($("lspeed")?.value || 4);
    const r = LV.replay;
    r.i++;
    const e = d.events[r.i - 1];
    drawEvents(r.i, e && evVisible(e));
    movePuck(e);
    if (e && e.type === "goal") { const sc = $("lsc"); sc.classList.remove("flash"); void sc.offsetWidth; sc.classList.add("flash"); }
    updatePanels(r.i);
    if (r.i >= d.events.length) { LV.replay = null; $("lreplay").textContent = "↻ Přehrát znovu"; updatePanels(d.events.length); return; }
    const pause = e && e.type === "goal" ? 2200 : 650;
    r.t = setTimeout(tick, pause / sp);
  };
  tick();
}

// ===================================================================
//  Přesnost modelu
// ===================================================================
async function loadAcc() {
  const el = $("view-acc");
  el.innerHTML = `<div class="loader"><div class="spinner"></div>Počítám, jak by model tipoval odehrané zápasy…</div>`;
  let b;
  try { b = await getJSON("/model/backtest"); } catch (e) { el.innerHTML = `<div class="loader err">${esc(e.message)}</div>`; return; }
  if (b.loading) { renderLoading(el, b); clearTimeout(pollTimer); pollTimer = setTimeout(loadAcc, 3000); return; }
  const pct = (x) => (x == null ? "–" : Math.round(x * 100) + " %");
  const cal = b.calib;
  const W = 560, Hh = 220, bw = W / Math.max(1, cal.length);
  let svg = `<svg viewBox="0 0 ${W} ${Hh + 40}" xmlns="http://www.w3.org/2000/svg" font-family="Inter,sans-serif">`;
  for (let i = 0; i <= 4; i++) svg += `<line x1="0" x2="${W}" y1="${Hh - (Hh * i) / 4}" y2="${Hh - (Hh * i) / 4}" stroke="rgba(148,163,184,.12)"/><text x="2" y="${Hh - (Hh * i) / 4 - 3}" font-size="10" fill="#8796ae">${i * 25} %</text>`;
  cal.forEach((c, i) => {
    const x = i * bw + bw * 0.18, w = bw * 0.3;
    svg += `<rect x="${x}" y="${Hh - c.pred * Hh}" width="${w}" height="${c.pred * Hh}" rx="4" fill="#38bdf8"><title>Model čekal ${pct(c.pred)}</title></rect>
      <rect x="${x + w + 4}" y="${Hh - c.actual * Hh}" width="${w}" height="${c.actual * Hh}" rx="4" fill="#fbbf24"><title>Ve skutečnosti ${pct(c.actual)}</title></rect>
      <text x="${i * bw + bw / 2}" y="${Hh + 16}" font-size="11" fill="#c7d3e6" text-anchor="middle" font-weight="700">${c.bin * 20}–${c.bin * 20 + 20} %</text>
      <text x="${i * bw + bw / 2}" y="${Hh + 32}" font-size="10" fill="#8796ae" text-anchor="middle">${c.n} tipů</text>`;
  });
  svg += "</svg>";
  el.innerHTML = `
    <section class="cine" id="cine" style="padding-bottom:30px">${RINK}<div class="spot"></div>
      <div class="kicker">zpětný test · ${fmtDate(b.from)} – ${fmtDate(b.to)}</div>
      <h1 class="display">Jak moc<span>model trefuje</span></h1>
      <p class="lead">Model jsme nechali „tipovat“ ${b.games} už odehraných zápasů – vždy jen z dat, která byla známá <b>před</b> zápasem. Pak jsme to porovnali se skutečností.</p>
      <div class="counters">
        <div><b data-count="${Math.round(b.winner.pct * 100)}">0</b><small>% trefený vítěz</small></div>
        <div><b data-count="${b.top1.hit}">0</b><small>z ${b.top1.n} top tipů na bod vyšlo</small></div>
        <div><b>±${b.shotsMae}</b><small>odchylka střel</small></div>
        <div><b>±${b.pensMae}</b><small>odchylka vyloučení</small></div>
      </div>
    </section>
    <div class="helpbox"><h3>👀 Jak to číst</h3><div class="help" style="margin:0"><ul>
      <li><b>Trefený vítěz</b>: v kolika zápasech vyhrál tým, kterému model dával víc než 50 %. Hokej je hodně náhodný – i nejlepší modely na světě trefí vítěze jen asi v 58–62 % zápasů.</li>
      <li><b>Odchylka</b>: o kolik se model v průměru spletl. „±${b.shotsMae} střel“ = skutečný počet střel byl v průměru o ${b.shotsMae} jinde, než model čekal (průměrný zápas má ${b.avgShots} střel).</li>
      <li><b>Top tipy na bod</b>: v každém zápase hráč s nejvyšší šancí na bod. Model čekal, že jich vyjde zhruba ${b.top1.expected} z ${b.top1.n} – ve skutečnosti vyšlo <b>${b.top1.hit}</b>.</li>
      <li><b>Graf níže</b> je nejdůležitější: když model říká „60 %“, mělo by to opravdu vyjít asi v 60 % případů. Čím víc se modrý a žlutý sloupec shodují, tím víc se dá procentům věřit.</li>
    </ul></div></div>
    <div class="grid g3">
      <div class="card bigstat"><small>Vítěz zápasu</small><b style="color:var(--cyan)">${pct(b.winner.pct)}</b><p class="help">${b.winner.ok} z ${b.games} zápasů.
        Když si byl model jistý (60 % a víc): <b>${pct(b.winner.confPct)}</b> z ${b.winner.confN} zápasů.</p></div>
      <div class="card bigstat"><small>Top 3 tipy na bod v zápase</small><b style="color:var(--gold)">${pct(b.top3.hit / Math.max(1, b.top3.n))}</b><p class="help">${b.top3.hit} z ${b.top3.n} tipů vyšlo (model čekal ${b.top3.expected}).</p></div>
      <div class="card bigstat"><small>Góly celkem – odchylka</small><b>±${b.goalsMae}</b><p class="help">Góly jsou nejvíc náhodné – i dobrý model se tu splete o 1,5–2 góly.</p></div>
    </div>
    <div class="sechead"><span class="no">01</span><h2>Dá se procentům věřit?</h2><span class="hint">šance na kanadský bod – všechny tipy</span></div>
    <div class="card calib">${svg}<div class="legend"><span><i style="background:#38bdf8"></i>co model čekal</span><span><i style="background:#fbbf24"></i>jak to dopadlo ve skutečnosti</span></div>
      <p class="help">Hráče jsme rozdělili podle toho, jakou jim model dával šanci na bod (např. 40–60 %). <b>Modrý sloupec</b> = průměrná šance od modelu, <b>žlutý</b> = kolik z nich opravdu bodovalo.
        Když jsou sloupce stejně vysoké, model je „poctivý“ – jeho procenta odpovídají realitě.</p></div>
    <div class="sechead"><span class="no">02</span><h2>Zápas po zápasu</h2><span class="hint">nejnovější nahoře</span></div>
    <p class="help" style="margin-top:-6px">✅ = model tipoval správně, ❌ = netrefil. U střel a vyloučení: <b>čekal / skutečnost</b>. U hráčů: ✅ = bodoval, ❌ = nebodoval.</p>
    <div class="tablewrap"><table class="t"><thead><tr><th class="left">Datum</th><th class="left">Zápas</th><th>Šance domácích</th><th>Výsledek</th><th>Střely</th><th>Vyloučení</th><th>Góly</th><th class="left">Top 3 tipy na bod</th></tr></thead><tbody>
      ${b.rows.map((r) => { const ok = (r.winP >= 0.5) === r.homeWon; return `<tr><td class="left">${fmtDate(r.date)}</td><td class="left"><b>${r.home} – ${r.away}</b></td>
        <td>${pc(r.winP)}</td><td>${r.score}${r.end !== "REG" ? " " + r.end : ""} ${ok ? '<span class="okc">✅</span>' : '<span class="noc">❌</span>'}</td>
        <td>${r.expShots} / <b>${r.shots}</b></td><td>${r.expPens} / <b>${r.pens ?? "–"}</b></td><td>${r.expGoals} / <b>${r.goals}</b></td>
        <td class="left">${r.top3.map((t) => `${t.hit ? "✅" : "❌"} ${esc(splitName(t.name)[1])} <small class="muted">${pc(t.p)}</small>`).join(" · ")}</td></tr>`; }).join("")}
    </tbody></table></div>
    <div class="disc">Zpětný test používá stejný model jako předpovědi, ale pro každý zápas jen data z dřívějších zápasů. Soupisky jsou aktuální, takže hráči, kteří mezitím přestoupili, chybí.
      Výsledky za 2 týdny jsou malý vzorek – čísla se mohou dost měnit. Ani dobrý model neznamená jistou výhru.</div>`;
  interactive(el);
}

// PWA
if ("serviceWorker" in navigator) { try { navigator.serviceWorker.register("/sw.js").catch(() => {}); } catch (e) {} }

// ===================================================================
//  NHL ID karta
// ===================================================================
const IDS = { teams: null, team: null, players: [], pid: null, theme: "neon", nick: "" };
const ID_THEMES = {
  sunset: "linear-gradient(160deg, #ffffff 0%, #eef2f7 55%, #dfe6ef 100%)",
  neon: "linear-gradient(160deg, color-mix(in srgb, var(--tc) 22%, #ffffff) 0%, #ffffff 60%, color-mix(in srgb, var(--tc) 12%, #ffffff) 100%)",
  ice: "linear-gradient(160deg, #f4f8fc 0%, #c9d4e2 45%, #f4f8fc 70%, #aeb9c8 100%)",
};
const proxImg = (u) => (u ? "/img?u=" + encodeURIComponent(u) : "");
function barcode(seed) {
  let s = "", x = 0, r = seed || 12345;
  for (let i = 0; i < 46; i++) { r = (r * 9301 + 49297) % 233280; const w = 1 + (r % 3); if (i % 2 === 0) s += `<rect x="${x}" y="0" width="${w}" height="40" fill="#1d0b2a"/>`; x += w; }
  return `<svg class="bar" viewBox="0 0 ${x} 40" preserveAspectRatio="none">${s}</svg>`;
}
async function openCard(arg) {
  const el = $("view-card");
  if (!IDS.teams) {
    el.innerHTML = `<div class="loader"><div class="spinner"></div>Načítám týmy…</div>`;
    try {
      const d = await getJSON("/api/standings/now");
      IDS.teams = (d.standings || []).map((t) => ({ ab: t.teamAbbrev?.default, name: t.teamName?.default, logo: t.teamLogo })).filter((t) => t.ab).sort((a, b) => a.name.localeCompare(b.name, "cs"));
    } catch (e) { el.innerHTML = `<div class="loader err">${esc(e.message)}</div>`; return; }
  }
  const [qt, qp] = (arg || "").split("-");
  if (qt && IDS.teams.find((t) => t.ab === qt)) IDS.team = qt;
  if (!IDS.team) IDS.team = store.get("nhl_team") && IDS.teams.find((t) => t.ab === store.get("nhl_team")) ? store.get("nhl_team") : IDS.teams[0].ab;
  if (qp) IDS.pid = +qp;
  el.innerHTML = `
    <section class="cine hero2" id="cine" style="min-height:44vh"><div class="sky"></div><div class="sun"></div>${SKYLINE}
      <div><span class="badge-gta">NHL · Player ID</span></div>
      <h1 class="display"><span class="ln"><i>NHL ID</i></span><span class="ln"><i>karta hráče</i></span></h1>
      <p class="lead">Vyber tým a hráče a vznikne jeho oficiální karta v barvách týmu – s čísly ze sezóny a dnešní předpovědí. Klikni na kartu a otočí se.</p></section>
    <div class="idwrap">
      <div class="card idctl">
        <div class="row"><label>Tým</label><select id="idteam">${IDS.teams.map((t) => `<option value="${t.ab}" ${t.ab === IDS.team ? "selected" : ""}>${esc(t.name)}</option>`).join("")}</select></div>
        <div class="row"><label>Hráč</label><select id="idplayer"><option>Načítám…</option></select></div>
        <div class="row"><label>Přezdívka (nepovinné)</label><input id="idnick" type="text" maxlength="22" placeholder="např. Ledová bestie" value="${esc(IDS.nick)}"></div>
        <div class="row"><label>Styl karty (venkovní · domácí · stříbrná)</label><div class="themes" id="idthemes">
          ${Object.entries(ID_THEMES).map(([k, v]) => `<button data-v="${k}" class="${k === IDS.theme ? "on" : ""}" style="background:${v}" title="${k}"></button>`).join("")}</div></div>
        <div class="row" style="display:flex;gap:8px;flex-wrap:wrap"><button class="btn" id="idflip">↻ Otočit kartu</button><button class="btn primary" id="iddl">Stáhnout jako obrázek</button>
          <button class="btn" id="idmore">Analýza hráče</button></div>
        <p class="help">Na počítači najeď myší na kartu – holografický odlesk se hýbe jako u pravé karty. Přední strana = údaje hráče, zadní = statistiky sezóny, forma a dnešní šance.</p>
      </div>
      <div><div class="idstage"><div class="idcard" id="idcard"><div class="loader">Vyber hráče…</div></div></div><p class="idnote" id="idnote"></p></div>
    </div>`;
  $("idteam").onchange = (e) => { IDS.team = e.target.value; IDS.pid = null; loadIdPlayers(); };
  $("idplayer").onchange = (e) => { IDS.pid = +e.target.value; renderId(); };
  $("idnick").oninput = (e) => { IDS.nick = e.target.value; renderId(); };
  segment($("idthemes"), (v) => { IDS.theme = v; renderId(); });
  $("idflip").onclick = () => $("idcard").classList.toggle("flip");
  $("iddl").onclick = downloadId;
  $("idmore").onclick = () => { const p = IDS.players.find((x) => x.playerId === IDS.pid); if (p) openPlayer(p.playerId, IDS.team, `${p.firstName.default} ${p.lastName.default}`); };
  interactive(el);
  loadIdPlayers();
}
async function loadIdPlayers() {
  try {
    const d = await getJSON(`/api/club-stats/${IDS.team}/now`);
    IDS.players = (d.skaters || []).sort((a, b) => b.points - a.points);
    IDS.goalies = d.goalies || [];
  } catch (e) { $("idcard").innerHTML = `<div class="loader err">${esc(e.message)}</div>`; return; }
  if (!IDS.pid || !IDS.players.find((p) => p.playerId === IDS.pid)) IDS.pid = IDS.players[0]?.playerId;
  $("idplayer").innerHTML = IDS.players.map((p) => `<option value="${p.playerId}" ${p.playerId === IDS.pid ? "selected" : ""}>${esc(name(p))} – ${p.points} b.</option>`).join("");
  renderId();
}
async function renderId() {
  const t = IDS.teams.find((x) => x.ab === IDS.team), p = IDS.players.find((x) => x.playerId === IDS.pid);
  if (!p) return;
  history.replaceState(null, "", `#karta-${IDS.team}-${p.playerId}`);
  let log = null;
  try { log = await getJSON(`/model/player?id=${p.playerId}&team=${IDS.team}`); } catch (e) {}
  if (IDS.pid !== p.playerId) return;
  const num = log?.games?.length ? "" : "";
  const nm = name(p), [first, last] = splitName(nm);
  const jersey = log?.num ?? "";
  const card = $("idcard");
  const flipped = card.classList.contains("flip");
  const games = log?.games?.slice(-10) || [];
  const mx = Math.max(1, ...games.map((g) => g.pts));
  const spark = games.map((g, i) => `<rect x="${i * 10 + 1}" y="${30 - (g.pts / mx) * 28}" width="7" height="${Math.max(2, (g.pts / mx) * 28)}" rx="1.5" fill="${g.pts ? "#ffc24b" : "#6b4a7a"}"/>`).join("");
  const inj = log?.injury;
  card.style.setProperty("--cardbg", ID_THEMES[IDS.theme]);
  card.style.setProperty("--tc", tcol(IDS.team));
  card.innerHTML = `
    <div class="idface front"><div class="bg"></div>
      <div class="hdr"><b>NHL · PLAYER ID</b><span>${esc(t?.name || IDS.team)}</span></div>
      <div class="photo"><div class="num">${jersey || posName[p.positionCode] || "#"}</div><img src="${proxImg(t?.logo)}" alt="" crossorigin="anonymous"></div>
      <svg class="stamp" viewBox="0 0 100 100"><path d="M50 4l40 16v26c0 25-18 42-40 50C28 88 10 71 10 46V20z" fill="#05070b" stroke="#c9d4e2" stroke-width="4"/>
        <text x="50" y="44" text-anchor="middle" font-family="Anton" font-size="18" fill="#fff">SEZÓNA</text><text x="50" y="70" text-anchor="middle" font-family="Anton" font-size="24" fill="${tcol(IDS.team)}">26/27</text></svg>
      <div class="fields">
        <div class="wide"><small>JMÉNO / NAME</small><b>${esc(first)} ${esc(last)}</b></div>
        <div><small>POZICE</small><b>${{ C: "Center", L: "Levé křídlo", R: "Pravé křídlo", D: "Obránce" }[p.positionCode] || p.positionCode}</b></div>
        <div><small>TÝM</small><b>${esc(IDS.team)}</b></div>
        <div><small>ZÁPASY · BODY</small><b>${p.gamesPlayed} · ${p.points}</b></div>
        <div><small>STAV</small><b style="color:${inj ? (inj.out ? "#c81e4a" : "#b7791f") : "#127a52"}">${inj ? esc(inj.status) : "Fit"}</b></div>
        ${IDS.nick ? `<div class="wide"><small>PŘEZDÍVKA</small><b style="font-size:clamp(16px,2.6vw,24px)">„${esc(IDS.nick)}“</b></div>` : ""}
        <div><small>ID Č.</small><b>${p.playerId}</b></div>
      </div>
      <div class="sig">${esc(last)}</div>${barcode(p.playerId)}
      <div class="holo"></div></div>
    <div class="idface back"><div class="bg"></div>
      <div class="hdr"><b>STATISTIKY</b><span>${esc(nm)}</span></div>
      <div class="bstats">
        <div><small>GÓLY</small><b>${p.goals}</b></div><div><small>ASISTENCE</small><b>${p.assists}</b></div>
        <div><small>BODY</small><b>${p.points}</b></div><div><small>B / ZÁPAS</small><b>${(p.points / Math.max(1, p.gamesPlayed)).toFixed(2)}</b></div>
        <div><small>STŘELY</small><b>${p.shots}</b></div><div><small>% STŘELBY</small><b>${p.shootingPctg != null ? (p.shootingPctg * 100).toFixed(1) : "–"}</b></div>
        <div><small>+/-</small><b>${p.plusMinus > 0 ? "+" + p.plusMinus : p.plusMinus}</b></div><div><small>ČAS NA LEDĚ</small><b>${toi(p.avgTimeOnIcePerGame)}</b></div>
        <div class="spark"><small>BODY V POSLEDNÍCH ${games.length} ZÁPASECH</small><svg viewBox="0 0 100 32" preserveAspectRatio="none" style="width:100%;height:40px">${spark}</svg></div>
        ${log?.next ? `<div class="spark" style="background:linear-gradient(90deg,#0d47c9,#2f7bff)"><small style="color:#fff">DNES ${esc(log.next.game)}</small><b style="color:#fff">${pc(log.next.pPoint)} na bod</b></div>` : ""}
      </div><div class="holo"></div></div>`;
  if (flipped) card.classList.add("flip");
  card.onclick = () => card.classList.toggle("flip");
  card.onmousemove = (e) => {
    const r = card.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
    card.querySelectorAll(".holo").forEach((h) => { h.style.setProperty("--hx", x * 100 + "%"); h.style.setProperty("--hy", y * 100 + "%"); });
    card.style.transform = `${card.classList.contains("flip") ? "rotateY(180deg) " : ""}rotateX(${(0.5 - y) * 10}deg) rotateY(${(x - 0.5) * 14}deg)`;
  };
  card.onmouseleave = () => (card.style.transform = "");
  $("idnote").textContent = inj ? `Marodka: ${inj.status}${inj.comment ? " – " + inj.comment : ""}` : "Klikni na kartu a otočí se.";
}
const LAST_NUMS = {};
async function downloadId() {
  const card = $("idcard");
  try {
    if (!window.htmlToImage) await new Promise((res, rej) => { const s = document.createElement("script"); s.src = "https://cdn.jsdelivr.net/npm/html-to-image@1.11.11/dist/html-to-image.js"; s.onload = res; s.onerror = rej; document.head.appendChild(s); });
    const face = card.classList.contains("flip") ? card.querySelector(".back") : card.querySelector(".front");
    const prev = card.style.transform; card.style.transform = card.classList.contains("flip") ? "rotateY(180deg)" : "";
    const url = await window.htmlToImage.toPng(face, { pixelRatio: 2, style: { transform: "none" } });
    card.style.transform = prev;
    const a = document.createElement("a"); a.href = url; a.download = `nhl-id-${IDS.team}-${IDS.pid}.png`; a.click();
  } catch (e) { alert("Obrázek se nepodařilo vytvořit. Zkus to v jiném prohlížeči."); }
}

// ===================================================================
//  Tikety z Tipsportu
// ===================================================================
const TK = { img: null, busy: false, last: null };
const KINDCZ = { winner_ot: "vítěz vč. prodloužení", "1x2": "výsledek (60 min)", total_goals: "góly celkem", team_goals: "góly týmu", total_shots: "střely celkem",
  team_shots: "střely týmu", total_pens: "vyloučení", total_pim: "trestné minuty", total_hits: "hity", player_point: "bod hráče", player_goal: "gól hráče",
  player_assist: "asistence hráče", player_2points: "2+ body hráče", player_shots: "střely hráče", first_goal_team: "první gól (tým)", first_goal_player: "první gól (hráč)",
  pp_goal_team: "gól v přesilovce", other: "jiná sázka" };
function tkHistory() { try { return JSON.parse(store.get("tk_hist") || "[]"); } catch (e) { return []; } }
function openTickets() {
  const el = $("view-tk");
  const hist = tkHistory();
  el.innerHTML = `
    <div class="sechead" style="margin-top:30px"><span class="no">TK</span><h2>Moje tikety z Tipsportu</h2></div>
    <div class="helpbox"><div class="help" style="margin:0"><ul>
      <li><b>Jak na to:</b> v aplikaci Tipsport otevři tiket (v Moje tikety) a udělej <b>screenshot</b>. Ten sem nahraj – na mobilu klepni na rámeček a vyber fotku, na počítači ho sem přetáhni nebo vlož <b>Ctrl+V</b>.</li>
      <li>AI tiket přečte, najde zápasy a hráče a <b>model každou sázku spočítá</b>: šance, férový kurz, jestli se vyplatí, zranění. U odehraných zápasů řekne, <b>co vyšlo a co ne</b>.</li>
      <li>Pak dostaneš <b>komentář a návrh úpravy</b> – co vyhodit a čím nahradit. Funguje pro NHL zápasy z posledních 7 dní a nadcházející.</li>
      <li>Obrázek se nikam neukládá, jen se pošle AI k přečtení. V prohlížeči si pamatuju jen výsledky posledních vyhodnocení.</li>
    </ul></div></div>
    <div class="tkgrid">
      <div class="card">
        <div class="drop" id="tkdrop"><b style="font-size:17px">Nahrát screenshot tiketu</b><p class="help">klepni / přetáhni sem / Ctrl+V</p><div id="tkprev"></div></div>
        <input type="file" id="tkfile" accept="image/*" style="display:none">
        <p class="help" style="margin-top:14px">…nebo vlož text tiketu (zkopírovaný z webu Tipsportu):</p>
        <textarea id="tktext" rows="4" style="width:100%;background:rgba(255,255,255,.05);color:var(--text);border:1px solid var(--line2);border-radius:10px;padding:10px;font:inherit" placeholder="např. Edmonton - Vegas, Vítěz zápasu: Edmonton 1.99&#10;McDavid Connor - kanadský bod: ano 1.45"></textarea>
        <button class="btn primary" id="tkgo" style="width:100%;margin-top:12px;font-size:16px;padding:13px">Vyhodnotit tiket</button>
        ${hist.length ? `<h3 style="margin:22px 0 8px;font-size:15px">Poslední vyhodnocení</h3><div class="tkhist">${hist.map((h, i) => `<div class="card h" data-i="${i}" style="padding:12px">
          <b>${esc(h.title)}</b><br><small class="muted">${esc(h.when)} · ${h.legs} sázek · kurz ${h.odds ?? "–"}</small><br><span class="${h.status === "won" ? "okc" : h.status === "lost" ? "noc" : "muted"}">${h.status === "won" ? "✅ vyhrál" : h.status === "lost" ? "❌ prohrál" : "⏳ otevřený"}</span></div>`).join("")}</div>` : ""}
      </div>
      <div id="tkres"><div class="card" style="min-height:260px;display:grid;place-items:center;text-align:center"><div><div class="display" style="font-size:42px;color:#fff">Čekám na tiket</div><p class="help">Výsledek vyhodnocení se ukáže tady.</p></div></div></div>
    </div>`;
  const drop = $("tkdrop"), file = $("tkfile");
  drop.onclick = () => file.click();
  file.onchange = () => file.files[0] && loadTkImage(file.files[0]);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) loadTkImage(f); };
  $("tkgo").onclick = sendTicket;
  el.querySelectorAll(".tkhist .h").forEach((c) => (c.onclick = () => { const h = tkHistory()[+c.dataset.i]; if (h?.data) renderTicket(h.data); }));
  if (TK.img) $("tkprev").innerHTML = `<img src="${TK.img}" alt="">`;
}
document.addEventListener("paste", (e) => {
  if (location.hash !== "#tikety") return;
  const it = [...(e.clipboardData?.items || [])].find((x) => x.type.startsWith("image/"));
  if (it) { e.preventDefault(); loadTkImage(it.getAsFile()); }
});
function loadTkImage(f) {
  const img = new Image(), url = URL.createObjectURL(f);
  img.onload = () => {
    const k = Math.min(1, 1800 / Math.max(img.width, img.height));
    const c = document.createElement("canvas"); c.width = Math.round(img.width * k); c.height = Math.round(img.height * k);
    c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
    TK.img = c.toDataURL("image/jpeg", 0.88);
    $("tkprev").innerHTML = `<img src="${TK.img}" alt="">`;
    URL.revokeObjectURL(url);
  };
  img.src = url;
}
async function sendTicket() {
  const text = $("tktext").value.trim();
  if (!TK.img && !text) { $("tkres").innerHTML = `<div class="card"><p class="err">Nahraj screenshot tiketu nebo vlož jeho text.</p></div>`; return; }
  if (TK.busy) return;
  TK.busy = true; $("tkgo").disabled = true; $("tkgo").textContent = "Vyhodnocuji…";
  $("tkres").innerHTML = `<div class="card loader"><div class="spinner"></div><b>AI čte tiket a model počítá sázky…</b><p class="help">Trvá to 10–30 vteřin.</p></div>`;
  try {
    const r = await fetch("/model/ticket", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: TK.img, text }) });
    const j = await r.json();
    if (!j.legs) throw new Error(j.error || "Vyhodnocení se nepovedlo.");
    renderTicket(j);
    const hist = tkHistory();
    hist.unshift({ when: new Date().toLocaleString("cs-CZ", { day: "numeric", month: "numeric", hour: "2-digit", minute: "2-digit" }),
      title: j.legs.map((x) => x.home && x.away ? `${x.home}–${x.away}` : "?").filter((v, i, a) => a.indexOf(v) === i).slice(0, 2).join(", "),
      legs: j.legs.length, odds: j.summary.totalOdds ? (+j.summary.totalOdds).toFixed(2) : null, status: j.summary.status, data: j });
    store.set("tk_hist", JSON.stringify(hist.slice(0, 8)));
  } catch (e) {
    $("tkres").innerHTML = `<div class="card"><p class="err">${esc(e.message)}</p></div>`;
  }
  TK.busy = false; $("tkgo").disabled = false; $("tkgo").textContent = "Vyhodnotit tiket";
}
function renderTicket(j) {
  TK.last = j;
  const S = j.summary;
  const evTxt = S.ev == null ? null : S.ev;
  const verdict = S.status === "won" ? ["good", "✅ Tiket vyšel"] : S.status === "lost" ? ["bad", "❌ Tiket nevyšel"]
    : evTxt == null ? ["mid", "Nelze plně spočítat"] : evTxt > 0.03 ? ["good", "Model vidí hodnotu"] : evTxt > -0.1 ? ["mid", "Zhruba férový tiket"] : ["bad", "Podle modelu nevýhodný"];
  $("tkres").innerHTML = `
    <div class="card">
      <span class="verdict ${verdict[0]}">${verdict[1]}</span>
      <div class="simstats" style="margin-top:14px">
        <div><small>Celkový kurz</small><b>${S.totalOdds ? (+S.totalOdds).toFixed(2) : "–"}</b></div>
        <div><small>Šance podle modelu</small><b style="color:var(--cyan)">${S.pAll != null ? (S.pAll * 100).toFixed(S.pAll < 0.1 ? 1 : 0) + " %" : "–"}</b></div>
        <div><small>Férový kurz</small><b>${S.pAll ? (1 / S.pAll).toFixed(2) : "–"}</b></div>
        <div><small>Hodnota tiketu</small><b>${S.ev != null ? evHtml(S.pAll, S.totalOdds) : "–"}</b></div>
        ${S.stake ? `<div><small>Vklad</small><b>${S.stake} Kč</b></div>` : ""}${S.possibleWin ? `<div><small>Možná výhra</small><b>${S.possibleWin} Kč</b></div>` : ""}
      </div>
      <p class="help"><b>Šance podle modelu</b> = jak pravděpodobné je, že vyjdou všechny sázky najednou. <b>Hodnota</b> kladná = kurz je vyšší, než by podle modelu měl být.</p>
    </div>
    <div class="tablewrap" style="margin-top:14px"><table class="t tkleg"><thead><tr><th class="left">Zápas</th><th class="left">Sázka</th><th>Kurz</th><th>Model</th><th>Férový</th><th>Hodnota</th><th>Výsledek</th></tr></thead><tbody>
      ${j.legs.map((x) => { const L = x.leg; return `<tr>
        <td class="left"><b>${x.gameId ? `<a href="#zapas-${x.gameId}">${esc(x.home)} – ${esc(x.away)}</a>` : esc(L.match || "?")}</b><small>${esc(L.match || "")}${x.date ? " · " + fmtDate(x.date) : ""}</small></td>
        <td class="left"><b>${esc(L.selection || "")}</b><small>${esc(L.market || "")}${x.note && x.note !== KINDCZ[L.kind] && x.note !== "vítěz vč. prodloužení" ? " · " + esc(x.note) : ""}</small>
          ${x.warn ? `<small style="color:var(--red)">🩹 ${esc(x.warn)}</small>` : ""}</td>
        <td>${L.odds ? (+L.odds).toFixed(2) : "–"}</td><td class="gold">${x.prob != null ? pc(x.prob) : "–"}</td><td>${x.fair ?? "–"}</td>
        <td>${x.prob != null && L.odds ? evHtml(x.prob, +L.odds) : "–"}</td>
        <td>${x.result === true ? "✅" : x.result === false ? "❌" : "⏳"}${x.finalScore ? `<small>${esc(x.finalScore)}</small>` : ""}</td></tr>`; }).join("")}
    </tbody></table></div>
    <div class="card" style="margin-top:14px"><b>🤖 Komentář AI</b><div class="help" style="margin-top:8px;font-size:15px;color:var(--text)">${j.comment ? md(j.comment) : esc(j.commentError || "Komentář se nepodařilo vytvořit.")}</div></div>
    ${j.suggestions?.length ? `<div class="card" style="margin-top:14px"><b>💡 Silné sázky z modelu ve stejných zápasech</b>
      <p class="help" style="margin:4px 0 8px">Pokud chceš tiket upravit, tohle jsou sázky, kterým model v těchto zápasech věří nejvíc. Porovnej jejich férový kurz s Tipsportem – vyplatí se jen tehdy, když je kurz u Tipsportu <b>vyšší</b> než férový.</p>
      ${j.suggestions.map((g) => `<div class="gk"><span><b>${esc(g.text)}</b><br><small class="muted">${esc(g.game)}</small></span><span style="text-align:right"><b style="color:var(--cyan)">${pc(g.prob)}</b><br><small class="muted">férový kurz ${fair(g.prob)}${g.odds ? " · Tipsport " + g.odds : ""}</small></span></div>`).join("")}</div>` : ""}
    <div style="display:flex;gap:8px;margin-top:14px;flex-wrap:wrap"><button class="btn" id="tkslip">Přenést do „Můj tiket“</button><button class="btn" id="tkai">Zeptat se AI na tento tiket</button></div>
    <p class="disc">Model je odhad, ne jistota. Sázky ze stejného zápasu spolu souvisí, takže šance celého tiketu je jen přibližná. Sázej jen to, co si můžeš dovolit prohrát.</p>`;
  $("tkslip").onclick = () => {
    j.legs.forEach((x, i) => { if (x.prob == null) return; SLIP.items = SLIP.items.filter((s) => s.key !== "tk" + i);
      SLIP.items.push({ key: "tk" + i + ":" + (x.gameId || ""), label: `${x.leg.selection} (${x.leg.market || ""})`, p: x.prob, odd: x.leg.odds ? +x.leg.odds : null, gid: x.gameId, match: `${x.home || "?"} – ${x.away || "?"}` }); });
    saveSlip(); renderSlip(); $("slipbtn").click();
  };
  $("tkai").onclick = () => { openChat(); sendChat("Mám tento tiket z Tipsportu: " + j.legs.map((x) => `${x.home}–${x.away}: ${x.leg.market} ${x.leg.selection} @ ${x.leg.odds}`).join("; ") + ". Co si o něm myslíš a jak bys ho vylepšil?"); };
}

// ===================================================================
//  Analýza hráče (okno)
// ===================================================================
async function openPlayer(pid, team, nm) {
  const m = $("pmodal"), box = $("pmbox");
  m.classList.remove("hidden");
  box.innerHTML = `<button class="x" onclick="closePlayer()">✕</button><div class="loader"><div class="spinner"></div>Načítám ${esc(nm || "hráče")}…</div>`;
  let d;
  try { d = await getJSON(`/model/player?id=${pid}&team=${team}`); } catch (e) { box.innerHTML += `<p class="err">${esc(e.message)}</p>`; return; }
  const g = d.games;
  const labels = g.map((x) => (x.home ? "" : "@") + x.opp);
  box.innerHTML = `<button class="x" onclick="closePlayer()">✕</button>
    <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap"><div class="display" style="font-size:44px;background:var(--sunset);-webkit-background-clip:text;background-clip:text;color:transparent">${esc(d.name || nm)}</div>
      <span class="pill">${esc(team)}</span>${d.injury ? `<span class="injbadge ${d.injury.out ? "" : "dtd"}">🩹 ${esc(d.injury.status)}</span>` : `<span class="pill good">fit</span>`}</div>
    ${d.injury?.comment ? `<p class="help">${esc(d.injury.comment)} (${esc(d.injury.date)})</p>` : ""}
    <div class="simstats" style="margin-top:14px">
      <div><small>Body na zápas</small><b>${num(d.avg.pts)}</b></div><div><small>Zápasy s bodem</small><b>${pc(d.avg.pointRate)}</b></div>
      <div><small>Střely na zápas</small><b>${num(d.avg.sog, 1)}</b></div><div><small>Čas na ledě</small><b>${toi(d.avg.toi)}</b></div>
      <div><small>Doma</small><b>${d.home.pts} b.</b><small>v ${d.home.gp} záp.</small></div><div><small>Venku</small><b>${d.away.pts} b.</b><small>v ${d.away.gp} záp.</small></div>
    </div>
    ${d.next ? `<div class="retro"><b>Další zápas: ${esc(d.next.game)}</b> · šance na bod <b style="color:var(--gold)">${pc(d.next.pPoint)}</b> · na gól ${pc(d.next.pGoal)} · 2+ body ${pc(d.next.p2Points)} · oček. střely ${num(d.next.expShots, 1)}
      <a href="#zapas-${d.next.gid}" onclick="closePlayer()" style="margin-left:8px">rozbor zápasu →</a></div>` : ""}
    <h3 style="margin:18px 0 4px">Posledních ${g.length} zápasů</h3>
    <p class="help" style="margin:0 0 6px">Žluté sloupce = kanadské body v zápase, modrá čára = střely na branku. @ = venku.</p>
    ${chart({ labels, bars: [{ name: "Body", color: "#ffc24b", values: g.map((x) => x.pts) }], lines: [{ name: "Střely", color: "#2de2e6", values: g.map((x) => x.sog) }], height: 170 })}
    <div class="tablewrap" style="margin-top:12px"><table class="t"><thead><tr><th class="left">Datum</th><th class="left">Soupeř</th><th>Výsledek</th><th>G</th><th>A</th><th>B</th><th>Střely</th><th>Hity</th><th>TM</th><th>TOI</th></tr></thead><tbody>
      ${g.slice().reverse().map((x) => `<tr><td class="left">${fmtDate(x.date)}</td><td class="left">${x.home ? "" : "@"}${x.opp}</td><td>${x.res === "V" ? "✅" : "❌"} ${x.score}</td><td>${x.g}</td><td>${x.a}</td><td class="gold">${x.pts}</td><td>${x.sog}</td><td>${x.hits}</td><td>${x.pim}</td><td>${toi(x.toi)}</td></tr>`).join("")}
    </tbody></table></div>
    <div style="margin-top:14px;display:flex;gap:8px;flex-wrap:wrap"><a class="btn" href="#karta-${team}-${pid}" onclick="closePlayer()" style="text-decoration:none">🪪 Vytvořit NHL ID kartu</a>
      <button class="btn primary" onclick="closePlayer();openChat();sendChat('Rozeber mi formu hráče ${esc((d.name || nm || "").replace(/'/g, ""))} (${team}) a jeho šance v dalším zápase.')">🤖 AI rozbor hráče</button></div>`;
}
function closePlayer() { $("pmodal").classList.add("hidden"); }
$("pmodal").onclick = (e) => { if (e.target.id === "pmodal") closePlayer(); };
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closePlayer(); });
document.addEventListener("click", (e) => {
  const a = e.target.closest && e.target.closest("[data-player]");
  if (a) { e.preventDefault(); e.stopPropagation(); const [pid, team] = a.dataset.player.split(":"); openPlayer(+pid, team, a.textContent); }
}, true);

// ===================================================================
//  AI chat
// ===================================================================
const CHAT = { msgs: [], gameId: null, busy: false, ai: true };
function md(t) {
  let h = esc(t).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|[^*])\*(?!\s)(.+?)\*/g, "$1<i>$2</i>");
  const out = []; let list = false;
  for (const line of h.split("\n")) {
    const m = line.match(/^\s*[-*•]\s+(.*)/);
    if (m) { if (!list) { out.push("<ul>"); list = true; } out.push(`<li>${m[1]}</li>`); continue; }
    if (list) { out.push("</ul>"); list = false; }
    if (line.trim()) out.push(`<p>${line.replace(/^#+\s*/, "")}</p>`);
  }
  if (list) out.push("</ul>");
  return out.join("");
}
function renderChat() {
  const el = $("msgs");
  if (!CHAT.msgs.length) {
    el.innerHTML = `<div class="msg ai"><p>Ahoj! 👋 Jsem AI analytik. Znám předpovědi všech nadcházejících zápasů NHL – šance na body, střely, vyloučení, góly i kurzy Tipsportu.</p><p>Ptej se normálně česky, třeba: <i>„Kdo dnes dá bod?“</i>, <i>„Kolik bude střel v zápase Bostonu?“</i> nebo <i>„Kde je hodnota v kurzech?“</i>. Když máš otevřený konkrétní zápas, odpovídám hlavně k němu.</p><p>Můžeš kliknout i na jednu z otázek dole.</p></div>`;
  } else {
    el.innerHTML = CHAT.msgs.map((m) => `<div class="msg ${m.role === "user" ? "user" : "ai"} ${m.err ? "err" : ""}">${m.role === "user" ? esc(m.content) : md(m.content)}</div>`).join("") +
      (CHAT.busy ? `<div class="msg ai typing"><span></span><span></span><span></span></div>` : "");
  }
  el.scrollTop = el.scrollHeight;
  const sugg = CHAT.gameId
    ? ["Udělej rozbor tohoto zápasu", "Kdo dá v tomto zápase bod?", "Bude víc než 60 střel?", "Kde je hodnota v kurzech?"]
    : ["Kdo má dnes největší šanci na bod?", "Který zápas bude mít nejvíc střel?", "Kde je dnes hodnota v kurzech?", "Kde čekat nejvíc vyloučení?"];
  $("sugg").innerHTML = CHAT.msgs.length > 2 ? "" : sugg.map((s) => `<button>${esc(s)}</button>`).join("");
  $("sugg").querySelectorAll("button").forEach((b) => (b.onclick = () => sendChat(b.textContent)));
}
function updateChatCtx() {
  const g = CHAT.gameId && OV?.games?.find((x) => x.id === CHAT.gameId);
  $("chatctx").textContent = CHAT.gameId ? `Zaměřeno na zápas ${g ? g.home + " – " + g.away : ""}` : "Všechny nadcházející zápasy";
  if (!$("chat").classList.contains("hidden")) renderChat();
}
function openChat() { $("chat").classList.remove("hidden"); $("chatbtn").classList.add("hidden"); renderChat(); setTimeout(() => $("chatin").focus(), 50); }
async function sendChat(text) {
  text = (text || "").trim();
  if (!text || CHAT.busy) return;
  CHAT.msgs.push({ role: "user", content: text });
  CHAT.busy = true; renderChat();
  try {
    const r = await fetch("/model/chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: CHAT.msgs.filter((m) => !m.err).map(({ role, content }) => ({ role, content })), gameId: CHAT.gameId }) });
    const j = await r.json().catch(() => ({}));
    if (j.reply) CHAT.msgs.push({ role: "assistant", content: j.reply });
    else CHAT.msgs.push({ role: "assistant", content: j.error || "AI teď neodpovídá.", err: true });
  } catch (e) {
    CHAT.msgs.push({ role: "assistant", content: "Spojení selhalo: " + e.message, err: true });
  }
  CHAT.busy = false; renderChat();
}
$("chatbtn").onclick = openChat;
$("chatclose").onclick = () => { $("chat").classList.add("hidden"); $("chatbtn").classList.remove("hidden"); };
$("chatform").onsubmit = (e) => { e.preventDefault(); const v = $("chatin").value; $("chatin").value = ""; sendChat(v); };

// ===================================================================
//  Statistiky týmu (sezóna)
// ===================================================================
const ST = { init: false, team: null, season: "now", gameType: "2", pos: "all", q: "", skaters: [], goalies: [], teams: {},
  sSort: { key: "points", dir: -1 }, gSort: { key: "wins", dir: -1 } };
const SK_COLS = [
  { k: "_rank", t: "#", cls: "rank", nosort: true },
  { k: "_name", t: "Hráč", cls: "left", v: (p) => name(p), f: (v) => `<b>${esc(v)}</b>` },
  { k: "positionCode", t: "Poz", cls: "left pos", f: (v) => posName[v] || v },
  { k: "gamesPlayed", t: "Z" }, { k: "goals", t: "G" }, { k: "assists", t: "A" },
  { k: "points", t: "B", cls: "gold" },
  { k: "_ppg", t: "B/Z", v: (p) => (p.gamesPlayed ? p.points / p.gamesPlayed : 0), f: (v) => num(v) },
  { k: "plusMinus", t: "+/-", f: (v) => (v > 0 ? `<span class="plus">+${v}</span>` : v < 0 ? `<span class="minus">${v}</span>` : "0") },
  { k: "penaltyMinutes", t: "TM" }, { k: "shots", t: "S" },
  { k: "shootingPctg", t: "%S", f: (v) => (v == null ? "–" : (v * 100).toFixed(1)) },
  { k: "powerPlayGoals", t: "PPG" }, { k: "shorthandedGoals", t: "SHG" }, { k: "gameWinningGoals", t: "GWG" },
  { k: "avgTimeOnIcePerGame", t: "TOI", f: toi },
  { k: "faceoffWinPctg", t: "%VB", f: (v) => (v ? (v * 100).toFixed(1) : "–") },
];
const G_COLS = [
  { k: "_rank", t: "#", cls: "rank", nosort: true },
  { k: "_name", t: "Brankář", cls: "left", v: (p) => name(p), f: (v) => `<b>${esc(v)}</b>` },
  { k: "gamesPlayed", t: "Z" }, { k: "gamesStarted", t: "Start" }, { k: "wins", t: "V" }, { k: "losses", t: "P" },
  { k: "overtimeLosses", t: "PP" }, { k: "goalsAgainstAverage", t: "GAA", f: (v) => num(v) },
  { k: "savePercentage", t: "%Z", cls: "gold", f: (v) => (v == null ? "–" : Number(v).toFixed(3)) },
  { k: "shotsAgainst", t: "Střely" }, { k: "saves", t: "Zákroky" }, { k: "goalsAgainst", t: "OG" }, { k: "shutouts", t: "ČK" },
];
function renderStats() {
  const q = ST.q.toLowerCase();
  const rows = ST.skaters.filter((p) => {
    if (ST.pos === "D" && p.positionCode !== "D") return false;
    if (ST.pos === "F" && p.positionCode === "D") return false;
    return !q || name(p).toLowerCase().includes(q);
  });
  const s = ST.skaters;
  if (s.length) {
    const top = (k) => s.reduce((a, b) => ((b[k] ?? 0) > (a[k] ?? 0) ? b : a));
    const tot = (k) => s.reduce((a, b) => a + (b[k] ?? 0), 0);
    $("scards").innerHTML = [["Nejvíc bodů", "points"], ["Nejlepší střelec", "goals"], ["Nejvíc asistencí", "assists"], ["Nejvíc střel", "shots"], ["Nejvíc trestných minut", "penaltyMinutes"]]
      .map(([l, k]) => { const p = top(k); return `<div class="card scard"><div class="lbl">${l}</div><div class="val">${p[k] ?? 0}</div><div class="who">${esc(name(p))}</div></div>`; }).join("") +
      `<div class="card scard"><div class="lbl">Tým celkem</div><div class="val">${tot("goals")} G</div><div class="who">${tot("shots")} střel · ${tot("penaltyMinutes")} TM</div></div>`;
  } else $("scards").innerHTML = "";
  sortTable($("skaters"), SK_COLS, rows, ST.sSort, renderStats);
  sortTable($("goalies"), G_COLS, ST.goalies, ST.gSort, renderStats);
}
async function loadStats() {
  $("skaters").innerHTML = `<tbody><tr><td class="loader"><div class="spinner"></div>Načítám…</td></tr></tbody>`;
  const t = ST.teams[ST.team] || {};
  $("slogo").src = t.logo || ""; $("stitle").textContent = t.name || ST.team;
  try {
    const path = ST.season === "now" ? `club-stats/${ST.team}/now` : `club-stats/${ST.team}/${ST.season}/${ST.gameType}`;
    const d = await getJSON("/api/" + path);
    ST.skaters = d.skaters || []; ST.goalies = d.goalies || [];
    $("ssub").textContent = d.season ? `Sezóna ${fmtSeason(d.season)} · ${d.gameType == 3 ? "play-off" : "základní část"}` : "";
    renderStats();
  } catch (e) {
    $("skaters").innerHTML = `<tbody><tr><td class="loader err">${esc(e.message)}</td></tr></tbody>`;
    $("goalies").innerHTML = ""; $("scards").innerHTML = "";
  }
}
async function loadSeasons() {
  $("season").innerHTML = `<option value="now">Aktuální sezóna</option>`;
  ST.season = "now";
  try {
    const list = await getJSON(`/api/club-stats-season/${ST.team}`);
    list.sort((a, b) => b.season - a.season).forEach((s) => $("season").insertAdjacentHTML("beforeend", `<option value="${s.season}">${fmtSeason(s.season)}</option>`));
  } catch (e) {}
}
async function initStats() {
  if (ST.init) return;
  ST.init = true;
  try {
    const d = await getJSON("/api/standings/now");
    (d.standings || []).forEach((t) => { const ab = t.teamAbbrev?.default; if (ab) ST.teams[ab] = { name: t.teamName?.default || ab, logo: t.teamLogo }; });
  } catch (e) {}
  const sorted = Object.entries(ST.teams).sort((a, b) => a[1].name.localeCompare(b[1].name, "cs"));
  $("team").innerHTML = sorted.map(([ab, t]) => `<option value="${ab}">${esc(t.name)}</option>`).join("");
  const saved = store.get("nhl_team");
  ST.team = saved && ST.teams[saved] ? saved : sorted[0]?.[0];
  if (!ST.team) return;
  $("team").value = ST.team;
  segment($("gtype"), (v) => {
    ST.gameType = v;
    if (ST.season === "now" && v === "3" && $("season").options[1]) { ST.season = $("season").options[1].value; $("season").value = ST.season; }
    loadStats();
  });
  segment($("posf"), (v) => { ST.pos = v; renderStats(); });
  $("search").oninput = (e) => { ST.q = e.target.value; renderStats(); };
  $("season").onchange = (e) => { ST.season = e.target.value; loadStats(); };
  $("team").onchange = async (e) => { ST.team = e.target.value; store.set("nhl_team", ST.team); await loadSeasons(); loadStats(); };
  await loadSeasons();
  loadStats();
}

route();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
