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
        self.lg = {"shots": 28.5, "goals": 3.05, "pens": 3.4}
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

        st.update(phase="Počítám předpovědi…")
        teams = build_team_records(self.games, pens)
        lg = league_averages(self.games, pens)
        upcoming = load_upcoming(info)

        preds = {}
        for g in upcoming:
            if g["home"] in teams and g["away"] in teams:
                try:
                    preds[g["id"]] = predict_game(g, teams, lg, roster, info)
                except Exception as e:
                    print("Předpověď selhala", g["id"], e, flush=True)

        with self.lock:
            self.info, self.roster, self.teams, self.lg = info, roster, teams, lg
            self.upcoming, self.preds = upcoming, preds
            self.ready = True
            self.updated = time.time()
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
                "players": me["players"], "starter": me["starter"],
            })
    return teams


def league_averages(games, pens):
    n = len(games) * 2
    if not n:
        return {"shots": 28.5, "goals": 3.05, "pens": 3.4}
    shots = sum(g["home"]["sog"] + g["away"]["sog"] for g in games.values()) / n
    goals = sum(g["home"]["score"] + g["away"]["score"] for g in games.values()) / n
    pv = list(pens.values())
    p = sum(pv) / len(pv) if pv else 3.4
    return {"shots": round(shots, 2), "goals": round(goals, 3), "pens": round(p, 3)}


def load_upcoming(info):
    d = get_json("schedule/now")
    tip_id = 3
    for p in d.get("oddsPartners") or []:
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

            out.append({
                "id": g["id"], "start": g.get("startTimeUTC"), "state": g.get("gameState"),
                "date": day.get("date"), "venue": _abbr(g.get("venue")),
                "home": ht["abbrev"], "away": at["abbrev"],
                "homeName": (info.get(ht["abbrev"]) or {}).get("name") or ht["abbrev"],
                "awayName": (info.get(at["abbrev"]) or {}).get("name") or at["abbrev"],
                "homeLogo": ht.get("darkLogo") or ht.get("logo"),
                "awayLogo": at.get("darkLogo") or at.get("logo"),
                "score": [ht.get("score"), at.get("score")],
                "odds": {"home": odd(ht), "away": odd(at), "provider": "Tipsport"},
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
           "pt": lg["pens"], "pd": lg["pens"]}
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


def predict_game(g, teams, lg, roster, info):
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
    players.sort(key=lambda p: (-p["active"], -p["pPoint"]))

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
        ]

    odds = g.get("odds") or {}
    value = {}
    for side, prob in (("home", win_home), ("away", win_away)):
        o = odds.get(side)
        if o:
            value[side] = round(prob * o - 1, 4)

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
    lines = [f"Ligový průměr na tým a zápas: střely {L.lg['shots']:.1f}, góly {L.lg['goals']:.2f}, vyloučení {L.lg['pens']:.2f}.",
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
                         f"gól {_p(pl['pGoal'])}, střely {pl['expShots']} (2+ {_p(pl['p2Shots'])}, 3+ {_p(pl['p3Shots'])}), "
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


def chat(messages, gid=None):
    key = gemini_key()
    if not key:
        return {"error": "AI zatím není zapnutá – chybí klíč GEMINI_API_KEY v nastavení Renderu."}
    if not LEAGUE.ready:
        return {"error": "Data se ještě načítají, zkus to za chvilku."}
    now = datetime.datetime.now(PRAGUE).strftime("%d.%m.%Y %H:%M")
    system = SYSTEM_PROMPT.format(now=now, context=chat_context(gid))
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
            if p:
                self._json(p)
            elif not LEAGUE.ready:
                self._json({"loading": True, **LEAGUE.status()})
            else:
                self._json({"error": "Pro tento zápas není předpověď (možná už začal nebo skončil)."}, 404)
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
        if self.path != "/model/chat":
            self._send(404, b"Not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = 0
        if n <= 0 or n > 60000:
            self._json({"error": "Zpráva je moc dlouhá."}, 400)
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
        res = chat(body.get("messages") or [], gid)
        self._json(res, 200 if "reply" in res else 503)


def main():
    LEAGUE.ensure()  # začni stahovat data hned po startu
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Anton&family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
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
.cine h1 span { display: block; color: transparent; -webkit-text-stroke: 1.5px rgba(255,255,255,.55); background: none; }
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
</style>
</head>
<body>
<div id="progress"></div>
<div class="topbar"><div class="wrap">
  <a class="brand" href="#"><span class="puck">🏒</span><span>NHL Predikce<small>body · střely · vyloučení · kurzy</small></span></a>
  <nav class="nav">
    <a href="#" id="nav-over" class="on">🔮 Zápasy a předpovědi</a>
    <a href="#statistiky" id="nav-stats">📊 Statistiky týmů</a>
  </nav>
</div></div>

<div class="wrap">
  <section id="view-over"></section>
  <section id="view-game" class="hidden"></section>
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
    <div class="pname"><span>${esc(first)}</span>${esc(last)}</div>
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
    cine.querySelector("h1").style.transform = `translateY(${y * 0.35}px)`;
    cine.querySelector(".lead").style.transform = `translateY(${y * 0.2}px)`;
    cine.querySelector("h1").style.opacity = 1 - y / 700;
  }
}, { passive: true });
let SL = { kind: "shots.total" };
function slCfg() {
  const [grp, side] = SL.kind.split(".");
  const d = GAME;
  if (grp === "shots") { const b = d.shots[side]; return { normal: true, mean: b.mean ?? b.exp, sd: b.sd ?? 6, lab: "střel", step: 1 }; }
  if (grp === "goals") return { normal: false, mean: d.goals.total, lab: "gólů" };
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

let revealObs = null;
function reveal(root) {
  const els = root.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) { els.forEach((e) => e.classList.add("in")); return; }
  revealObs = revealObs || new IntersectionObserver((ents) => ents.forEach((en) => {
    if (en.isIntersecting) { en.target.classList.add("in"); revealObs.unobserve(en.target); }
  }), { threshold: 0.08 });
  els.forEach((e, i) => { e.style.transitionDelay = `${Math.min(i % 8, 7) * 60}ms`; revealObs.observe(e); });
}

// sortovatelná tabulka – formátovač dostává (hodnota, řádek)
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
  $("view-over").classList.toggle("hidden", !!game || stats);
  $("view-game").classList.toggle("hidden", !game);
  $("view-stats").classList.toggle("hidden", !stats);
  $("nav-over").classList.toggle("on", !stats);
  $("nav-stats").classList.toggle("on", stats);
  CHAT.gameId = game;
  updateChatCtx();
  if (stats) initStats();
  else if (game) openGame(game);
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
      <span class="nm">${esc(p.name)}${p.streak >= 3 ? ` <span class="pill hot">🔥${p.streak}</span>` : ""}<small>${esc(p.g.home)} – ${esc(p.g.away)} · ${fmtTime(p.g.start)}</small></span><span class="val">${val}</span></div>`;
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
    <section class="cine" id="cine">${RINK}<div class="spot"></div>
      <div class="kicker">NHL · předpovědi · ${dayIso ? fmtDay(dayIso) : ""}</div>
      <h1 class="display">Hokejová noc<span>v číslech</span></h1>
      <p class="lead">Model prošel ${(OV.status?.total || 0)} odehraných zápasů sezóny a spočítal, kdo dá bod, kolik padne střel, gólů a vyloučení
        ${best ? `– největší šanci na bod má dnes <b style="color:#fff">${esc(best.name)}</b> (${pc(best.pPoint)}).` : "."}</p>
      <div class="counters">
        <div><b data-count="${dayGames.length}">0</b><small>zápasů</small></div>
        <div><b data-count="${Math.round(sum((g) => g.goals.total))}">0</b><small>očekávaných gólů</small></div>
        <div><b data-count="${Math.round(sum((g) => g.shots.total))}">0</b><small>očekávaných střel</small></div>
        <div><b data-count="${Math.round(sum((g) => g.pens.total))}">0</b><small>očekávaných vyloučení</small></div>
      </div>
      <p class="help" style="position:relative">Čísla nahoře jsou součty za všechny zápasy vybraného dne. Den přepneš tlačítky níže.</p>
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
  reveal(el); interactive(el);
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
  { k: "name", t: "Hráč", cls: "left", f: (v, p) => `<b>${esc(v)}</b>${p.streak >= 3 ? ` <span class="pill hot">🔥 ${p.streak} v řadě</span>` : ""}${!p.active ? ` <span class="pill bad">nehrál ${p.lastGameAgo} z.</span>` : ""}` },
  { k: "team", t: "Tým", cls: "left pos" },
  { k: "pos", t: "Poz", cls: "left pos", f: (v) => posName[v] || v },
  { k: "pPoint", t: "Šance na bod", cls: "gold", f: (v) => `<span class="pbar" style="width:${Math.round(v * 60)}px"></span>${pc(v)}` },
  { k: "fairPoint", t: "Fér. kurz", v: (p) => 1 / p.pPoint, f: (v) => num(v) },
  { k: "pGoal", t: "Šance na gól", f: pc },
  { k: "expShots", t: "Oček. střely", f: (v) => `<b>${num(v, 1)}</b>` },
  { k: "p2Shots", t: "Střely 1.5+", f: pc },
  { k: "p3Shots", t: "Střely 2.5+", f: pc },
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

    <div class="statrow">
      <div class="card stat"><h4>🏒 Střely na branku</h4><div class="total">${d.shots.total.exp} <small>celkem</small></div>${cmp(d.shots.home.exp, d.shots.away.exp, (x) => x.toFixed(1))}
        <p class="help">Kolik <b>střel na branku</b> čekáme za oba týmy dohromady (velké číslo) a za každý tým zvlášť (pruhy). Ligový průměr je ${d.league.shots.toFixed(1)} střel na tým.</p></div>
      <div class="card stat"><h4>⛔ Vyloučení</h4><div class="total">${d.pens.total.exp.toFixed(1)} <small>celkem</small></div>${cmp(d.pens.home.exp, d.pens.away.exp, (x) => x.toFixed(1))}
        <p class="help">Kolik <b>trestů (vyloučení)</b> čekáme. Číslo u týmu = kolikrát půjdou <b>jeho</b> hráči na trestnou lavici. Ligový průměr je ${d.league.pens.toFixed(1)} na tým.</p></div>
      <div class="card stat"><h4>🥅 Góly</h4><div class="total">${d.goals.total.toFixed(1)} <small>celkem</small></div>${cmp(d.goals.home, d.goals.away, (x) => x.toFixed(2))}
        <p class="help">Kolik <b>gólů</b> čekáme celkem a za každý tým. Ligový průměr je ${d.league.goals.toFixed(2)} gólu na tým.</p></div>
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
        <button data-v="goals.total">Góly celkem</button><button data-v="pens.total">Vyloučení celkem</button><button data-v="pens.home">Vyloučení ${H}</button><button data-v="pens.away">Vyloučení ${A}</button></div>
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
        <br>U vítěze zápasu se kurzy z Tipsportu vyplní samy.</div>
      <div class="seg" id="mkseg" style="margin-bottom:12px">
        <button data-v="win" class="${GS.mkt === "win" ? "on" : ""}">Výsledek</button><button data-v="goals" class="${GS.mkt === "goals" ? "on" : ""}">Góly</button>
        <button data-v="shots" class="${GS.mkt === "shots" ? "on" : ""}">Střely</button><button data-v="pens" class="${GS.mkt === "pens" ? "on" : ""}">Vyloučení</button></div>
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
      <b>Střely 1.5+ / 2.5+</b> = šance na aspoň 2 / aspoň 3 střely · <b>Body posl. 5</b> = žluté čtverečky = zápasy, kdy bodoval (nejnovější vpravo), číslo = body celkem ·
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
