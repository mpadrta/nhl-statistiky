#!/usr/bin/env python3
"""
NHL statistiky týmu – kanadské body, góly, asistence, trestné minuty, střely...

Spuštění:   python nhl_statistiky.py
Pak se v prohlížeči otevře http://localhost:8765

Potřebuje jen Python 3.8+ (žádné další balíčky) a připojení k internetu.
Data bere z oficiálního NHL API (api-web.nhle.com).
"""

import datetime
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8765
NHL_API = "https://api-web.nhle.com/v1/"
CACHE_SECONDS = 300  # data se drží 5 minut, pak se stáhnou znovu

_cache = {}
_cache_lock = threading.Lock()


def http_get(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (nhl-statistiky)", "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read()


def nhl_get(path, base=NHL_API):
    """Stáhne JSON z NHL API (s krátkou cache)."""
    key = base + path
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1]
    data = http_get(key)
    with _cache_lock:
        _cache[key] = (now, data)
        if len(_cache) > 400:  # ať nám nepřeteče paměť
            for k in sorted(_cache, key=lambda k: _cache[k][0])[:200]:
                del _cache[k]
    return data


# =====================================================================
#  PREDIKČNÍ MODEL
# =====================================================================
# Jak to funguje (zjednodušeně):
#  * Pro oba týmy stáhneme všechny odehrané zápasy sezóny (boxscore + seznam vyloučení).
#  * Novější zápasy mají větší váhu (poločas rozpadu 12 zápasů).
#  * Každý tým má "útok" (střely/góly/vyloučení soupeře, které vynucuje) a "obranu"
#    (střely/góly povolené, vyloučení, kterých se dopouští) – zvlášť doma a venku.
#  * Očekávání = útok týmu × obrana soupeře / ligový průměr  (klasický multiplikativní model).
#  * Malé vzorky se "smršťují" k průměru, aby 2 zápasy neudělaly z průměrného hráče hvězdu.
#  * Šance hráče na bod: jeho bodový průměr a podíl zápasů s bodem, upravené o
#    doma/venku, aktuální roli (čas na ledě) a o to, kolik gólů má tým v tomto zápase čekat.

FINAL_STATES = ("FINAL", "OFF")
HALF_LIFE = 12.0
STATS_REST = "https://api.nhle.com/stats/rest/en/"
_perm = {}
_perm_lock = threading.Lock()
_pred_cache = {}


def get_json(path, base=NHL_API, permanent=False):
    key = base + path
    if permanent:
        with _perm_lock:
            if key in _perm:
                return _perm[key]
        data = json.loads(http_get(key))
        with _perm_lock:
            _perm[key] = data
        return data
    return json.loads(nhl_get(path, base))


def _abbr(x):
    if isinstance(x, dict):
        return x.get("default")
    return x


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


def _poisson_pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _poisson_over(line, lam):
    """P(X > line) pro Poissonovo rozdělení (line je x.5)."""
    k_max = int(math.floor(line))
    return 1 - sum(_poisson_pmf(k, lam) for k in range(k_max + 1))


def _normal_over(line, mean, sd):
    return 1 - _phi((line - mean) / sd)


def _lines(mean, step=1, count=3):
    c = math.floor(mean) + 0.5
    half = count // 2
    return [c + step * i for i in range(-half, half + 1)]


def count_penalties(landing):
    """Počet vyloučení podle týmů (bez osobních trestů a trestných střel)."""
    out = {}
    if not landing:
        return None
    for per in (landing.get("summary") or {}).get("penalties") or []:
        for p in per.get("penalties") or []:
            if p.get("type") in ("MIS", "GMIS", "PS"):
                continue
            ab = _abbr(p.get("teamAbbrev"))
            if ab:
                out[ab] = out.get(ab, 0) + 1
    return out


def load_game(gid):
    try:
        box = get_json(f"gamecenter/{gid}/boxscore", permanent=True)
    except Exception:
        return None, None
    try:
        land = get_json(f"gamecenter/{gid}/landing", permanent=True)
    except Exception:
        land = None
    return box, land


def league_averages():
    lg = {"shots": 28.5, "goals": 3.05, "pens": 3.4, "source": "odhad"}
    try:
        season = get_json("standings/now").get("standings", [{}])[0].get("seasonId")
    except Exception:
        season = None
    if not season:
        return lg
    exp = f"?cayenneExp=seasonId={season}%20and%20gameTypeId=2"
    try:
        rows = get_json("team/summary" + exp, base=STATS_REST).get("data", [])
        if rows:
            lg["shots"] = sum(r["shotsForPerGame"] for r in rows) / len(rows)
            lg["goals"] = sum(r["goalsForPerGame"] for r in rows) / len(rows)
            lg["source"] = "NHL"
    except Exception:
        pass
    try:
        rows = get_json("team/penalties" + exp, base=STATS_REST).get("data", [])
        vals = []
        for r in rows:
            gp = r.get("gamesPlayed") or 0
            if gp:
                n = (r.get("minors") or 0) + (r.get("majors") or 0) + \
                    (r.get("benchMinorPenalties") or 0) + (r.get("matchPenalties") or 0)
                vals.append(n / gp)
        if vals:
            lg["pens"] = sum(vals) / len(vals)
    except Exception:
        pass
    return lg


def team_records(team):
    """Všechny odehrané zápasy základní části týmu, od nejnovějšího."""
    sched = get_json(f"club-schedule-season/{team}/now")
    games = [g for g in sched.get("games", [])
             if g.get("gameType") == 2 and g.get("gameState") in FINAL_STATES]
    games.sort(key=lambda g: g.get("startTimeUTC", ""), reverse=True)
    with ThreadPoolExecutor(max_workers=10) as ex:
        loaded = list(ex.map(lambda g: load_game(g["id"]), games))
    recs = []
    for g, (box, land) in zip(games, loaded):
        if not box:
            continue
        is_home = box["homeTeam"]["abbrev"] == team
        side, oside = ("homeTeam", "awayTeam") if is_home else ("awayTeam", "homeTeam")
        me, op = box[side], box[oside]
        pens = count_penalties(land)
        pst = (box.get("playerByGameStats") or {}).get(side) or {}
        players = []
        for p in (pst.get("forwards") or []) + (pst.get("defense") or []):
            players.append({
                "id": p.get("playerId"), "name": _abbr(p.get("name")), "pos": p.get("position"),
                "g": p.get("goals", 0) or 0, "a": p.get("assists", 0) or 0,
                "pts": p.get("points", 0) or 0, "sog": p.get("sog", 0) or 0,
                "pim": p.get("pim", 0) or 0, "toi": _toi(p.get("toi")),
                "ppg": p.get("powerPlayGoals", 0) or 0, "hits": p.get("hits", 0) or 0,
            })
        starter = None
        for gk in pst.get("goalies") or []:
            if gk.get("starter"):
                starter = {"id": gk.get("playerId"), "name": _abbr(gk.get("name")),
                           "sa": gk.get("shotsAgainst", 0), "ga": gk.get("goalsAgainst", 0)}
        recs.append({
            "id": g["id"], "date": box.get("gameDate") or g.get("gameDate"),
            "home": is_home, "opp": op["abbrev"],
            "gf": me.get("score", 0) or 0, "ga": op.get("score", 0) or 0,
            "sf": me.get("sog", 0) or 0, "sa": op.get("sog", 0) or 0,
            "pt": pens.get(team, 0) if pens is not None else None,
            "pd": pens.get(op["abbrev"], 0) if pens is not None else None,
            "pim": sum(p["pim"] for p in players),
            "end": (box.get("gameOutcome") or {}).get("lastPeriodType", "REG"),
            "players": players, "starter": starter,
        })
    return recs


def _wstats(recs, key, venue=None):
    """Vážený průměr (novější zápasy víc) -> (průměr, součet vah, počet)."""
    sw = sv = 0.0
    n = 0
    for i, r in enumerate(recs):
        v = r.get(key)
        if v is None or (venue is not None and r["home"] != venue):
            continue
        w = 0.5 ** (i / HALF_LIFE)
        sw += w
        sv += v * w
        n += 1
    return (sv / sw if sw else None), sw, n


def _shrunk(recs, key, prior, venue=None, k=3.0):
    m, sw, _ = _wstats(recs, key, venue)
    if m is None:
        return prior
    return (m * sw + prior * k) / (sw + k)


def team_profile(recs, lg):
    pri = {"sf": lg["shots"], "sa": lg["shots"], "gf": lg["goals"], "ga": lg["goals"],
           "pt": lg["pens"], "pd": lg["pens"]}
    prof = {}
    for key, prior in pri.items():
        allv = _shrunk(recs, key, prior, k=3)
        prof[key] = {
            "all": allv,
            "home": _shrunk(recs, key, allv, venue=True, k=4),
            "away": _shrunk(recs, key, allv, venue=False, k=4),
        }
    # rozptyl střel týmu
    sf = [r["sf"] for r in recs]
    if len(sf) >= 5:
        m = sum(sf) / len(sf)
        sd = math.sqrt(sum((x - m) ** 2 for x in sf) / (len(sf) - 1))
    else:
        sd = 6.0
    prof["sf_sd"] = _clamp(sd, 4.5, 8.5)
    last5 = recs[:5]

    def rec_str(rs):
        w = sum(1 for r in rs if r["gf"] > r["ga"])
        l_reg = sum(1 for r in rs if r["gf"] < r["ga"] and r["end"] == "REG")
        l_ot = len(rs) - w - l_reg
        return f"{w}-{l_reg}-{l_ot}"

    prof["form"] = {
        "last5": rec_str(last5),
        "gf5": round(sum(r["gf"] for r in last5) / max(1, len(last5)), 2),
        "ga5": round(sum(r["ga"] for r in last5) / max(1, len(last5)), 2),
        "sf5": round(sum(r["sf"] for r in last5) / max(1, len(last5)), 1),
        "home": rec_str([r for r in recs if r["home"]]),
        "away": rec_str([r for r in recs if not r["home"]]),
        "gp": len(recs),
    }
    return prof


def player_projections(team, recs, prof, exp_goals, exp_shots, venue_home, roster):
    """Šance hráčů na bod / gól / střely v tomto zápase."""
    by_id = {}
    for i, r in enumerate(recs):
        w = 0.5 ** (i / HALF_LIFE)
        for p in r["players"]:
            by_id.setdefault(p["id"], []).append((i, w, r["home"], p))
    ctx_g = _clamp(exp_goals / max(0.5, prof["gf"]["all"]), 0.7, 1.4)
    ctx_s = _clamp(exp_shots / max(10, prof["sf"]["all"]), 0.75, 1.3)
    out = []
    for pid, apps in by_id.items():
        info = roster.get(pid)
        if roster and not info:
            continue  # hráč už v týmu není
        last = apps[0][3]
        pos = last["pos"] or "C"
        is_d = pos == "D"
        pri_p = 0.27 if is_d else 0.42
        pri_g = 0.06 if is_d else 0.16
        pri_s = 1.4 if is_d else 1.9
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
        # doma / venku
        mv, wv = wm(lambda p: p["pts"], venue_home)
        vf = _clamp(shr(mv, wv, ppg, 6) / ppg, 0.8, 1.25) if ppg > 0 else 1.0
        msv, wsv = wm(lambda p: p["sog"], venue_home)
        vfs = _clamp(shr(msv, wsv, spg, 6) / spg, 0.8, 1.25) if spg > 0 else 1.0
        # role (čas na ledě poslední 3 zápasy vs. sezóna)
        toi_all = wm(lambda p: p["toi"])[0] or 0
        recent = [a[3]["toi"] for a in apps[:3]]
        toi_rec = sum(recent) / len(recent) if recent else toi_all
        tf = _clamp(math.sqrt(toi_rec / toi_all), 0.85, 1.15) if toi_all > 0 else 1.0
        p_base = 0.5 * (prate + (1 - math.exp(-ppg)))
        p_point = 1 - (1 - p_base) ** (vf * tf * ctx_g)
        lam_g = gpg * vf * tf * ctx_g
        lam_s = spg * vfs * tf * ctx_s
        # forma
        last5 = apps[:5]
        streak = 0
        for a in apps:
            if a[3]["pts"] > 0:
                streak += 1
            else:
                break
        home_apps = [a[3]["pts"] for a in apps if a[2]]
        away_apps = [a[3]["pts"] for a in apps if not a[2]]
        name = info["name"] if info else last["name"]
        out.append({
            "id": pid, "name": name, "team": team, "pos": pos,
            "gp": len(apps), "g": sum(a[3]["g"] for a in apps), "a": sum(a[3]["a"] for a in apps),
            "pts": sum(a[3]["pts"] for a in apps),
            "pPoint": round(p_point, 4), "pGoal": round(1 - math.exp(-lam_g), 4),
            "expShots": round(lam_s, 2),
            "p2Shots": round(1 - math.exp(-lam_s) * (1 + lam_s), 4),
            "p3Shots": round(1 - math.exp(-lam_s) * (1 + lam_s + lam_s ** 2 / 2), 4),
            "form5": sum(a[3]["pts"] for a in last5), "form5gp": len(last5),
            "streak": streak,
            "homePpg": round(sum(home_apps) / len(home_apps), 2) if home_apps else None,
            "awayPpg": round(sum(away_apps) / len(away_apps), 2) if away_apps else None,
            "toi": round(toi_rec),
            "active": apps[0][0] <= 2,  # hrál v jednom z posledních 3 zápasů týmu
            "lastGameAgo": apps[0][0],
        })
    out.sort(key=lambda p: (-p["active"], -p["pPoint"]))
    return out


def goalie_info(team, recs, roster_goalies, game_date):
    starts = [r["starter"] for r in recs[:5] if r.get("starter")]
    counts = {}
    for s in starts:
        counts[s["id"]] = counts.get(s["id"], 0) + 1
    likely = None
    if starts:
        likely = max(counts, key=lambda k: (counts[k], -[s["id"] for s in starts].index(k)))
    b2b = False
    if recs and game_date:
        try:
            d0 = datetime.date.fromisoformat(recs[0]["date"])
            d1 = datetime.date.fromisoformat(game_date)
            b2b = (d1 - d0).days == 1
        except Exception:
            pass
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
            "gp": g.get("gamesPlayed"), "gs": g.get("gamesStarted"),
            "svp": g.get("savePercentage"), "gaa": g.get("goalsAgainstAverage"),
            "wins": g.get("wins"), "starts5": counts.get(g.get("playerId"), 0),
            "likely": g.get("playerId") == likely,
        })
    goalies.sort(key=lambda g: (-g["likely"], -(g["starts5"] or 0), -(g["gs"] or 0)))
    return {"goalies": goalies[:3], "b2b": b2b, "restDays": rest}


def predict(home, away, game_date=None):
    key = (home, away, game_date)
    hit = _pred_cache.get(key)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    lg = league_averages()
    with ThreadPoolExecutor(max_workers=2) as ex:
        fh, fa = ex.submit(team_records, home), ex.submit(team_records, away)
        rh, ra = fh.result(), fa.result()
    ph, pa = team_profile(rh, lg), team_profile(ra, lg)

    def roster(team):
        try:
            d = get_json(f"club-stats/{team}/now")
        except Exception:
            return {}, []
        r = {}
        for s in d.get("skaters", []):
            r[s["playerId"]] = {"name": f"{_abbr(s.get('firstName'))} {_abbr(s.get('lastName'))}"}
        return r, d.get("goalies", [])

    (rosH, gkH), (rosA, gkA) = roster(home), roster(away)

    # --- týmová očekávání ---
    sH = ph["sf"]["home"] * pa["sa"]["away"] / lg["shots"]
    sA = pa["sf"]["away"] * ph["sa"]["home"] / lg["shots"]
    gH = ph["gf"]["home"] * pa["ga"]["away"] / lg["goals"]
    gA = pa["gf"]["away"] * ph["ga"]["home"] / lg["goals"]
    nH = ph["pt"]["home"] * pa["pd"]["away"] / lg["pens"]
    nA = pa["pt"]["away"] * ph["pd"]["home"] / lg["pens"]

    gi_h = goalie_info(home, rh, gkH, game_date)
    gi_a = goalie_info(away, ra, gkA, game_date)
    # únava: druhý zápas za dva dny = trochu méně
    if gi_h["b2b"]:
        gH *= 0.96; sH *= 0.98; gA *= 1.03
    if gi_a["b2b"]:
        gA *= 0.96; sA *= 0.98; gH *= 1.03

    # výsledek (Poisson)
    p_home = p_away = p_draw = 0.0
    p_over_goals = {}
    grid = [[_poisson_pmf(i, gH) * _poisson_pmf(j, gA) for j in range(12)] for i in range(12)]
    for i in range(12):
        for j in range(12):
            if i > j:
                p_home += grid[i][j]
            elif i < j:
                p_away += grid[i][j]
            else:
                p_draw += grid[i][j]
    share = gH / (gH + gA)
    win_home = p_home + p_draw * share
    win_away = p_away + p_draw * (1 - share)
    for line in (4.5, 5.5, 6.5):
        p_over_goals[str(line)] = round(_poisson_over(line, gH + gA), 4)
    best = max(((i, j) for i in range(8) for j in range(8)), key=lambda t: grid[t[0]][t[1]])

    sd_tot = math.sqrt(ph["sf_sd"] ** 2 + pa["sf_sd"] ** 2)

    def shots_block(lam, sd, step=2):
        return {"exp": round(lam, 1),
                "lines": [{"line": l, "over": round(_normal_over(l, lam, sd), 4)} for l in _lines(lam, step)]}

    def pens_block(lam):
        return {"exp": round(lam, 2),
                "lines": [{"line": l, "over": round(_poisson_over(l, lam), 4)} for l in _lines(lam, 1)]}

    players = (player_projections(home, rh, ph, gH, sH, True, rosH) +
               player_projections(away, ra, pa, gA, sA, False, rosA))
    players.sort(key=lambda p: (-p["active"], -p["pPoint"]))

    h2h = [{"date": r["date"], "home": r["home"], "gf": r["gf"], "ga": r["ga"], "sf": r["sf"],
            "sa": r["sa"], "pt": r["pt"], "pd": r["pd"], "end": r["end"]}
           for r in rh if r["opp"] == away]

    def why(team, opp, p, po, is_home, s, n, g):
        v, ov = ("home", "away") if is_home else ("away", "home")
        vz, ovz = ("doma", "venku") if is_home else ("venku", "doma")
        return [
            f"{team} {vz} střílí v průměru {p['sf'][v]:.1f} střel, {opp} {ovz} povoluje {po['sa'][ov]:.1f} (liga {lg['shots']:.1f}) → čekáme {s:.1f}.",
            f"{team} {vz} dává {p['gf'][v]:.2f} gólu na zápas, {opp} {ovz} dostává {po['ga'][ov]:.2f} → čekáme {g:.2f}.",
            f"{team} {vz} chodí na trestnou lavici {p['pt'][v]:.1f}× za zápas, soupeři {opp} {ovz} vynucují {po['pd'][ov]:.1f}× → čekáme {n:.1f} vyloučení.",
            f"Forma posledních 5 zápasů: {p['form']['last5']} (V-P-PP), góly {p['form']['gf5']}:{p['form']['ga5']}, střely {p['form']['sf5']}.",
            f"Bilance {vz}: {p['form'][v]}.",
        ]

    result = {
        "home": home, "away": away, "date": game_date, "league": lg,
        "win": {"home": round(win_home, 4), "away": round(win_away, 4), "ot": round(p_draw, 4)},
        "goals": {"home": round(gH, 2), "away": round(gA, 2), "total": round(gH + gA, 2),
                  "over": p_over_goals, "likelyScore": f"{best[0]}:{best[1]}"},
        "shots": {"home": shots_block(sH, ph["sf_sd"]), "away": shots_block(sA, pa["sf_sd"]),
                  "total": shots_block(sH + sA, sd_tot)},
        "pens": {"home": pens_block(nH), "away": pens_block(nA), "total": pens_block(nH + nA)},
        "teams": {
            home: {"form": ph["form"], "why": why(home, away, ph, pa, True, sH, nH, gH), **gi_h},
            away: {"form": pa["form"], "why": why(away, home, pa, ph, False, sA, nA, gA), **gi_a},
        },
        "h2h": h2h,
        "players": players,
        "games": {"home": len(rh), "away": len(ra)},
    }
    _pred_cache[key] = (time.time(), result)
    return result


def upcoming_games():
    d = get_json("schedule/now")
    days = []
    for day in d.get("gameWeek", []):
        games = []
        for g in day.get("games", []):
            if g.get("gameType") not in (2, 3):
                continue
            games.append({
                "id": g["id"], "start": g.get("startTimeUTC"), "state": g.get("gameState"),
                "date": day.get("date"),
                "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
                "homeName": f"{_abbr(g['homeTeam'].get('placeName')) or ''} {_abbr(g['homeTeam'].get('commonName')) or ''}".strip(),
                "awayName": f"{_abbr(g['awayTeam'].get('placeName')) or ''} {_abbr(g['awayTeam'].get('commonName')) or ''}".strip(),
                "homeLogo": g["homeTeam"].get("darkLogo") or g["homeTeam"].get("logo"),
                "awayLogo": g["awayTeam"].get("darkLogo") or g["awayTeam"].get("logo"),
                "score": [g["homeTeam"].get("score"), g["awayTeam"].get("score")],
            })
        if games:
            days.append({"date": day.get("date"), "games": games})
    return {"days": days}


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
        if self.path.startswith("/model/"):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            try:
                if u.path == "/model/games":
                    data = upcoming_games()
                elif u.path == "/model/predict":
                    home = q.get("home", [""])[0].upper()
                    away = q.get("away", [""])[0].upper()
                    date = q.get("date", [None])[0]
                    if not (home.isalpha() and away.isalpha() and len(home) == 3 and len(away) == 3):
                        raise ValueError("Špatné týmy")
                    if date and not all(c.isdigit() or c == "-" for c in date):
                        date = None
                    data = predict(home, away, date)
                else:
                    self._send(404, b'{"error":"not found"}', "application/json")
                    return
                self._send(200, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
            except Exception as e:
                msg = json.dumps({"error": f"Výpočet se nepovedl: {e}"}, ensure_ascii=False).encode()
                self._send(502, msg, "application/json; charset=utf-8")
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
<title>NHL – statistiky a předpovědi</title>
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
  nav.tabs { display: flex; gap: 4px; padding: 14px 24px 0; border-bottom: 1px solid var(--line); }
  nav.tabs a { color: var(--muted); text-decoration: none; padding: 10px 16px; border-radius: 10px 10px 0 0;
               font-weight: 600; border: 1px solid transparent; border-bottom: none; }
  nav.tabs a.on { color: var(--text); background: var(--panel); border-color: var(--line); }
  .hidden { display: none !important; }
  /* předpovědi */
  .day h3 { font-size: 14px; color: var(--muted); margin: 18px 0 8px; text-transform: capitalize; }
  .glist { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; }
  .gcard { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px;
           cursor: pointer; display: flex; align-items: center; gap: 10px; }
  .gcard:hover { border-color: var(--accent); }
  .gcard img { height: 30px; width: 30px; }
  .gcard .vs { flex: 1; font-weight: 600; }
  .gcard .time { color: var(--muted); font-size: 13px; text-align: right; }
  .live { color: var(--bad); font-weight: 700; }
  .back { margin: 4px 0 14px; }
  .matchhead { display: flex; align-items: center; justify-content: center; gap: 22px; margin: 6px 0 6px; flex-wrap: wrap; }
  .matchhead .tm { text-align: center; min-width: 140px; }
  .matchhead img { height: 64px; }
  .matchhead .nm { font-weight: 700; font-size: 18px; }
  .matchhead .sub { color: var(--muted); font-size: 12px; }
  .matchhead .mid { text-align: center; }
  .matchhead .score { font-size: 34px; font-weight: 800; }
  .winbar { display: flex; height: 30px; border-radius: 8px; overflow: hidden; margin: 10px auto 4px; max-width: 700px; font-weight: 700; font-size: 13px; }
  .winbar div { display: flex; align-items: center; padding: 0 10px; }
  .winbar .h { background: var(--accent); justify-content: flex-start; }
  .winbar .a { background: #8a5cf6; justify-content: flex-end; }
  .centernote { text-align: center; color: var(--muted); font-size: 13px; }
  .pgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-top: 16px; }
  .pbox { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
  .pbox h4 { margin: 0 0 10px; font-size: 15px; }
  .pbox .row { display: flex; justify-content: space-between; padding: 4px 0; }
  .pbox .big { font-size: 22px; font-weight: 800; color: var(--gold); }
  .pbox .lines { margin-top: 8px; border-top: 1px solid var(--line); padding-top: 8px; font-size: 13px; color: var(--muted); }
  .pbox .lines span { color: var(--text); font-weight: 600; }
  .why { font-size: 13px; color: var(--muted); line-height: 1.55; padding-left: 18px; margin: 6px 0 0; }
  .pbar { display: inline-block; height: 8px; border-radius: 4px; background: var(--gold); vertical-align: middle; margin-right: 6px; }
  .tag { display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 6px; background: var(--panel2); color: var(--muted); margin-left: 4px; }
  .tag.hot { background: rgba(242,193,78,.18); color: var(--gold); }
  .tag.warn { background: rgba(255,107,107,.15); color: var(--bad); }
  td.dim, tr.dim td { opacity: .45; }
  .disclaimer { color: var(--muted); font-size: 12px; margin-top: 18px; line-height: 1.6; border-top: 1px solid var(--line); padding-top: 12px; }
  .spinner { display: inline-block; width: 18px; height: 18px; border: 3px solid var(--line); border-top-color: var(--accent);
             border-radius: 50%; animation: spin 1s linear infinite; vertical-align: middle; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<nav class="tabs">
  <a href="#" id="tab-stats" class="on">📊 Statistiky týmu</a>
  <a href="#predikce" id="tab-pred">🔮 Předpovědi zápasů</a>
</nav>

<section id="view-pred" class="hidden">
  <main id="pred"></main>
</section>

<section id="view-stats">
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
</section>

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

// =====================================================================
//  PŘEDPOVĚDI ZÁPASŮ
// =====================================================================
const P = { games: null, game: null, data: null, teamF: "all", showAll: false,
            sort: { key: "pPoint", dir: -1 } };
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pc = (v) => (v == null ? "–" : Math.round(v * 100) + " %");
const fmtTime = (iso) => new Date(iso).toLocaleTimeString("cs-CZ", { hour: "2-digit", minute: "2-digit" });
const fmtDay = (iso) => new Date(iso).toLocaleDateString("cs-CZ", { weekday: "long", day: "numeric", month: "numeric" });
const fmtDate = (s) => { const [y, m, d] = String(s).split("-"); return `${+d}. ${+m}.`; };

async function model(path) {
  const r = await fetch("/model/" + path);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || "Chyba " + r.status);
  return j;
}

function showView() {
  const pred = location.hash === "#predikce";
  $("view-pred").classList.toggle("hidden", !pred);
  $("view-stats").classList.toggle("hidden", pred);
  $("tab-pred").classList.toggle("on", pred);
  $("tab-stats").classList.toggle("on", !pred);
  if (pred && !P.games) loadGames();
}
window.addEventListener("hashchange", showView);

async function loadGames() {
  $("pred").innerHTML = `<div class="msg"><span class="spinner"></span>Načítám nadcházející zápasy…</div>`;
  try {
    P.games = await model("games");
    renderGames();
  } catch (e) {
    $("pred").innerHTML = `<div class="msg err">${esc(e.message)}</div>`;
  }
}

function renderGames() {
  P.game = null;
  const groups = new Map();
  for (const day of P.games.days) {
    for (const g of day.games) {
      if (["FINAL", "OFF"].includes(g.state)) continue;
      const k = fmtDay(g.start);
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(g);
    }
  }
  let html = `<h2 style="margin-top:18px">Vyber zápas a model spočítá předpověď</h2>
    <div class="centernote" style="text-align:left">Časy jsou v tvém časovém pásmu. Model vychází ze všech odehraných zápasů této sezóny.</div>`;
  if (!groups.size) html += `<div class="msg">Teď nejsou naplánované žádné zápasy.</div>`;
  for (const [day, games] of groups) {
    html += `<div class="day"><h3>${esc(day)}</h3><div class="glist">` + games.map((g) => `
      <div class="gcard" data-id="${g.id}">
        <img src="${esc(g.homeLogo || "")}" alt=""><div class="vs">${esc(g.home)} – ${esc(g.away)}
          <div class="sub" style="color:var(--muted);font-size:12px;font-weight:400">${esc(g.homeName)} vs ${esc(g.awayName)}</div></div>
        <img src="${esc(g.awayLogo || "")}" alt="">
        <div class="time">${["LIVE", "CRIT"].includes(g.state) ? '<span class="live">ŽIVĚ</span>' : fmtTime(g.start)}</div>
      </div>`).join("") + `</div></div>`;
  }
  $("pred").innerHTML = html;
  $("pred").querySelectorAll(".gcard").forEach((c) => (c.onclick = () => {
    const id = +c.dataset.id;
    for (const d of P.games.days) for (const g of d.games) if (g.id === id) openGame(g);
  }));
  window.scrollTo(0, 0);
}

async function openGame(g) {
  P.game = g; P.teamF = "all"; P.showAll = false;
  $("pred").innerHTML = `<button class="back">← Zpět na zápasy</button>
    <div class="msg"><span class="spinner"></span>Počítám předpověď ${esc(g.home)} – ${esc(g.away)}…<br>
    <small>Stahuju všechny odehrané zápasy obou týmů. Poprvé to může trvat až půl minuty.</small></div>`;
  $("pred").querySelector(".back").onclick = renderGames;
  window.scrollTo(0, 0);
  try {
    P.data = await model(`predict?home=${g.home}&away=${g.away}&date=${g.date}`);
    if (P.game === g) renderDetail();
  } catch (e) {
    if (P.game === g) $("pred").innerHTML = `<button class="back">← Zpět na zápasy</button><div class="msg err">${esc(e.message)}</div>`;
    const b = $("pred").querySelector(".back"); if (b) b.onclick = renderGames;
  }
}

function linesHtml(block, label) {
  return `<div class="lines">${label ? `<b style="color:var(--text)">${label}</b><br>` : ""}` +
    block.lines.map((l) => `${l.line}: přes <span>${pc(l.over)}</span> · pod <span>${pc(1 - l.over)}</span>`).join("<br>") + `</div>`;
}

const PL_COLS = [
  { k: "_rank", t: "#", cls: "rank", nosort: true },
  { k: "name", t: "Hráč", cls: "left", f: (v, p) => v },
  { k: "team", t: "Tým", cls: "left pos" },
  { k: "pos", t: "Poz", cls: "left pos", f: (v) => posName[v] || v },
  { k: "pPoint", t: "Šance na bod", cls: "pts", f: (v) => `<span class="pbar" style="width:${Math.round(v * 60)}px"></span>${pc(v)}` },
  { k: "pGoal", t: "Šance na gól", f: pc },
  { k: "expShots", t: "Oček. střely", f: (v) => num(v, 1) },
  { k: "p2Shots", t: "2+ střely", f: pc },
  { k: "p3Shots", t: "3+ střely", f: pc },
  { k: "form5", t: "Forma (B v posl. 5)" },
  { k: "streak", t: "Série s bodem" },
  { k: "homePpg", t: "Doma B/Z", f: (v) => num(v) },
  { k: "awayPpg", t: "Venku B/Z", f: (v) => num(v) },
  { k: "gp", t: "Z" }, { k: "g", t: "G" }, { k: "a", t: "A" }, { k: "pts", t: "B" },
  { k: "toi", t: "TOI posl. 3", f: toi },
];

function renderPlayers() {
  const d = P.data;
  let rows = d.players.filter((p) => (P.teamF === "all" || p.team === P.teamF) && (P.showAll || p.active));
  rows = rows.map((p) => {
    let tags = "";
    if (p.streak >= 3) tags += `<span class="tag hot">🔥 ${p.streak} v řadě</span>`;
    if (!p.active) tags += `<span class="tag warn">nehrál ${p.lastGameAgo} záp.</span>`;
    return { ...p, name: p.name, _nameHtml: esc(p.name) + tags };
  });
  const cols = PL_COLS.map((c) => (c.k === "name" ? { ...c, f: (v, p) => p._nameHtml } : c));
  renderTablePl($("pltable"), cols, rows, P.sort, renderPlayers);
}

// varianta renderTable, která předává celý řádek do formátovače
function renderTablePl(el, cols, rows, sort, onSort) {
  const sc = cols.find((c) => c.k === sort.key);
  rows = [...rows].sort((a, b) => {
    const x = a[sc.k], y = b[sc.k];
    if (typeof x === "string") return sort.dir * x.localeCompare(y, "cs");
    return sort.dir * ((x ?? -1e9) - (y ?? -1e9));
  });
  el.innerHTML = "<thead><tr>" + cols.map((c) =>
    `<th class="${c.cls || ""} ${c.k === sort.key ? "sorted" : ""}" data-k="${c.k}">${c.t}${c.k === sort.key ? (sort.dir < 0 ? " ▾" : " ▴") : ""}</th>`).join("") +
    "</tr></thead><tbody>" + (rows.length ? rows.map((p, i) => `<tr class="${p.active ? "" : "dim"}">` + cols.map((c) => {
      if (c.k === "_rank") return `<td class="rank">${i + 1}</td>`;
      const v = p[c.k];
      return `<td class="${c.cls || ""}">${c.f ? c.f(v, p) : (v ?? "–")}</td>`;
    }).join("") + "</tr>").join("") : `<tr><td colspan="${cols.length}" class="msg">Žádní hráči</td></tr>`) + "</tbody>";
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

function goalieHtml(team, t) {
  const g = t.goalies.find((x) => x.likely) || t.goalies[0];
  let s = `<div class="row"><b>${esc(team)}</b><span>${g ? esc(g.name) : "–"}</span></div>`;
  if (g) s += `<div class="row" style="color:var(--muted);font-size:13px"><span>%Z ${g.svp != null ? Number(g.svp).toFixed(3) : "–"} · GAA ${num(g.gaa)}</span><span>startů v posl. 5: ${g.starts5}</span></div>`;
  if (t.b2b) s += `<div class="row"><span class="tag warn">Hraje 2. den po sobě – může chytat náhradník</span></div>`;
  else if (t.restDays != null) s += `<div class="row" style="color:var(--muted);font-size:13px"><span>Odpočinek: ${t.restDays} ${t.restDays === 1 ? "den" : t.restDays >= 2 && t.restDays <= 4 ? "dny" : "dní"}</span></div>`;
  return s;
}

function renderDetail() {
  const g = P.game, d = P.data, H = d.home, A = d.away;
  const th = d.teams[H], ta = d.teams[A];
  const wh = Math.round(d.win.home * 100), wa = 100 - wh;
  const h2h = d.h2h.length ? d.h2h.map((r) => {
    const [l, rr] = r.home ? [`${H} ${r.gf}`, `${r.ga} ${A}`] : [`${A} ${r.ga}`, `${r.gf} ${H}`];
    const sh = r.home ? `${r.sf}:${r.sa}` : `${r.sa}:${r.sf}`;
    const pn = r.pt == null ? "–" : r.home ? `${r.pt}:${r.pd}` : `${r.pd}:${r.pt}`;
    return `<div class="row"><span>${fmtDate(r.date)}</span><span><b>${l}:${rr}</b>${r.end !== "REG" ? " " + r.end : ""}</span><span style="color:var(--muted)">střely ${sh} · vyl. ${pn}</span></div>`;
  }).join("") : `<div style="color:var(--muted)">Letos spolu ještě nehráli.</div>`;

  $("pred").innerHTML = `
    <button class="back">← Zpět na zápasy</button>
    <div class="matchhead">
      <div class="tm"><img src="${esc(g.homeLogo || "")}" alt=""><div class="nm">${esc(g.homeName)}</div><div class="sub">doma · ${th.form.home} · forma ${th.form.last5}</div></div>
      <div class="mid"><div class="sub">${fmtDay(g.start)} ${fmtTime(g.start)}</div>
        <div class="score">${d.goals.home.toFixed(1)} : ${d.goals.away.toFixed(1)}</div>
        <div class="sub">očekávané góly · nejpravděpodobnější výsledek ${d.goals.likelyScore}</div></div>
      <div class="tm"><img src="${esc(g.awayLogo || "")}" alt=""><div class="nm">${esc(g.awayName)}</div><div class="sub">venku · ${ta.form.away} · forma ${ta.form.last5}</div></div>
    </div>
    <div class="winbar"><div class="h" style="width:${wh}%">${H} ${wh} %</div><div class="a" style="width:${wa}%">${wa} % ${A}</div></div>
    <div class="centernote">Šance na výhru (včetně prodloužení a nájezdů) · šance, že zápas půjde do prodloužení: ${pc(d.win.ot)}</div>

    <div class="pgrid">
      <div class="pbox"><h4>🏒 Střely na branku</h4>
        <div class="row"><span>${H}</span><b>${d.shots.home.exp}</b></div>
        <div class="row"><span>${A}</span><b>${d.shots.away.exp}</b></div>
        <div class="row"><span>Celkem</span><span class="big">${d.shots.total.exp}</span></div>
        ${linesHtml(d.shots.total, "Celkem")}${linesHtml(d.shots.home, H)}${linesHtml(d.shots.away, A)}
      </div>
      <div class="pbox"><h4>⛔ Vyloučení (počet trestů)</h4>
        <div class="row"><span>${H}</span><b>${d.pens.home.exp}</b></div>
        <div class="row"><span>${A}</span><b>${d.pens.away.exp}</b></div>
        <div class="row"><span>Celkem</span><span class="big">${d.pens.total.exp}</span></div>
        ${linesHtml(d.pens.total, "Celkem")}${linesHtml(d.pens.home, H)}${linesHtml(d.pens.away, A)}
      </div>
      <div class="pbox"><h4>🥅 Góly</h4>
        <div class="row"><span>${H}</span><b>${d.goals.home}</b></div>
        <div class="row"><span>${A}</span><b>${d.goals.away}</b></div>
        <div class="row"><span>Celkem</span><span class="big">${d.goals.total}</span></div>
        <div class="lines">${Object.entries(d.goals.over).map(([l, v]) => `${l}: přes <span>${pc(v)}</span> · pod <span>${pc(1 - v)}</span>`).join("<br>")}</div>
        <h4 style="margin-top:16px">🧤 Pravděpodobní brankáři</h4>
        ${goalieHtml(H, th)}<div style="height:8px"></div>${goalieHtml(A, ta)}
      </div>
    </div>

    <div class="pgrid">
      <div class="pbox"><h4>Proč model čeká tohle – ${H}</h4><ul class="why">${th.why.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>
      <div class="pbox"><h4>Proč model čeká tohle – ${A}</h4><ul class="why">${ta.why.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>
      <div class="pbox"><h4>Vzájemné zápasy v této sezóně</h4>${h2h}</div>
    </div>

    <h2>Šance hráčů na kanadský bod</h2>
    <div class="controls" style="padding:0 0 10px">
      <div class="seg" id="plteam">
        <button data-v="all" class="on">Oba týmy</button><button data-v="${H}">${H}</button><button data-v="${A}">${A}</button>
      </div>
      <label style="display:flex;align-items:center;gap:6px;color:var(--muted);font-size:14px">
        <input type="checkbox" id="plall"> ukázat i hráče, kteří v posledních zápasech nehráli</label>
    </div>
    <div class="tablewrap"><table id="pltable"></table></div>
    <div class="disclaimer">
      <b>Jak to model počítá:</b> bere všechny odehrané zápasy obou týmů v této sezóně (${d.games.home} + ${d.games.away}),
      novější zápasy mají větší váhu. Útok týmu (střely, góly, vynucená vyloučení) kombinuje s obranou soupeře,
      zvlášť pro domácí a venkovní zápasy, a porovnává s ligovým průměrem (střely ${d.league.shots.toFixed(1)},
      góly ${d.league.goals.toFixed(2)}, vyloučení ${d.league.pens.toFixed(1)} na tým a zápas). U hráčů počítá s bodovým průměrem,
      podílem zápasů s bodem, formou, výkony doma/venku, časem na ledě v posledních zápasech a s tím, kolik gólů má jeho tým v tomto zápase čekat.
      Hráči, kteří nehráli v žádném z posledních 3 zápasů týmu (zranění, mimo sestavu), jsou skrytí.<br>
      Je to statistický odhad, ne jistota. Model neví o čerstvých zraněních, změnách sestav ani o tom, kdo skutečně nastoupí v brance.
    </div>`;
  $("pred").querySelector(".back").onclick = renderGames;
  segment("plteam", (v) => { P.teamF = v; renderPlayers(); });
  $("plall").onchange = (e) => { P.showAll = e.target.checked; renderPlayers(); };
  renderPlayers();
}

(async () => {
  showView();
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
