"""
Animated, photo-like weather background for Météo Clair (pure HTML + CSS + SVG filters, no images or JavaScript).

Layers, back to front: sky gradient -> sun/moon glow -> volumetric clouds (SVG turbulence) -> distant hills with
atmospheric haze -> trees that sway with the wind -> rain / mist / lightning -> film grain.
Every animated element is its own HTML element and only its transform/opacity is animated, so the browser draws the
expensive SVG filters once and then just moves the result (smooth even on phones).
"""
import math
import random

# sky (top, middle, horizon) for day scenes
SKY = {
    "sunny":   ("#2f7fd1", "#6fb2ea", "#cfe8f7"),
    "cloudy":  ("#5f84a8", "#97b4cc", "#d3dfe8"),
    "drizzle": ("#56687a", "#8698a8", "#b9c4cd"),
    "rain":    ("#3a4756", "#5f6d7b", "#8a96a1"),
    "storm":   ("#141a26", "#2a3140", "#4a5262"),
}
NIGHT_SKY = ("#050b1d", "#0f1d3d", "#26375c")
# hills: far, middle, near — haze makes far hills pale
HILLS = {
    "sunny":   ("#9ec1c9", "#6f9a7a", "#3f6e45"),
    "cloudy":  ("#a3b5bd", "#7a9583", "#4a6a4e"),
    "drizzle": ("#8e9ea6", "#6c8173", "#3f5844"),
    "rain":    ("#76838b", "#56685d", "#33473a"),
    "storm":   ("#3d4552", "#2c3a35", "#1c2a22"),
}
NIGHT_HILLS = ("#1c2840", "#142134", "#0c1622")


def kind_of(cat, p_rain):
    """Map today's rain category and probability to a scene."""
    if cat >= 3:
        return "storm"
    if cat == 2:
        return "rain"
    if cat == 1:
        return "drizzle"
    return "sunny" if p_rain < 0.2 else "cloudy"


def theme(kind, night):
    """Text colours for the glass cards and charts. Dark scenes get white text."""
    if night or kind in ("rain", "storm"):
        return dict(dark=True, fg="#ffffff", muted="rgba(255,255,255,0.80)", track="rgba(255,255,255,0.16)",
                    grid="rgba(255,255,255,0.14)", band=(0.03, 0.06, 0.09, 0.12))
    return dict(dark=False, fg="#14213a", muted="#33425b", track="rgba(255,255,255,0.45)",
                grid="rgba(20,33,58,0.10)", band=(0.10, 0.18, 0.26, 0.34))


# ---------------------------------------------------------------- SVG building blocks
DEFS = """
<svg width="0" height="0" style="position:absolute" aria-hidden="true">
 <filter id="mcCloud" x="-30%" y="-40%" width="160%" height="180%">
  <feTurbulence type="fractalNoise" baseFrequency="0.012" numOctaves="5" seed="4" result="n"/>
  <feDisplacementMap in="SourceGraphic" in2="n" scale="95" xChannelSelector="R" yChannelSelector="G" result="d"/>
  <feGaussianBlur in="d" stdDeviation="3.5"/>
 </filter>
 <filter id="mcCloudSoft" x="-30%" y="-40%" width="160%" height="180%">
  <feTurbulence type="fractalNoise" baseFrequency="0.009" numOctaves="4" seed="9" result="n"/>
  <feDisplacementMap in="SourceGraphic" in2="n" scale="140" xChannelSelector="R" yChannelSelector="G" result="d"/>
  <feGaussianBlur in="d" stdDeviation="9"/>
 </filter>
 <filter id="mcLeaves" x="-20%" y="-20%" width="140%" height="140%">
  <feTurbulence type="fractalNoise" baseFrequency="0.09" numOctaves="3" seed="2" result="n"/>
  <feDisplacementMap in="SourceGraphic" in2="n" scale="22" xChannelSelector="R" yChannelSelector="G"/>
 </filter>
 <filter id="mcHill" x="0" y="-20%" width="100%" height="140%">
  <feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves="3" seed="5" result="n"/>
  <feDisplacementMap in="SourceGraphic" in2="n" scale="14" xChannelSelector="R" yChannelSelector="G"/>
 </filter>
</svg>"""


def _cloud(rnd, y_pct, width, dur, delay, light, shade, opacity, soft):
    """A volumetric cloud: a darker underside and a bright top, both shaped by turbulence."""
    h = width * 0.42
    top, bottom = [], []
    n = rnd.randint(5, 8)
    for i in range(n):
        cx = width * (0.15 + 0.7 * i / max(n - 1, 1)) + rnd.uniform(-20, 20)
        cy = h * rnd.uniform(0.42, 0.62)
        rx = width * rnd.uniform(0.10, 0.17)
        ry = h * rnd.uniform(0.20, 0.33)
        top.append(f'<ellipse cx="{cx:.0f}" cy="{cy:.0f}" rx="{rx:.0f}" ry="{ry:.0f}"/>')
        bottom.append(f'<ellipse cx="{cx + 8:.0f}" cy="{cy + ry * 0.45:.0f}" rx="{rx * 1.05:.0f}" ry="{ry * 0.75:.0f}"/>')
    flt = "mcCloudSoft" if soft else "mcCloud"
    return (f'<svg class="sc-cloud" viewBox="0 0 {width:.0f} {h:.0f}" style="top:{y_pct:.1f}%;width:{width:.0f}px;'
            f'animation-duration:{dur:.0f}s;animation-delay:-{delay:.0f}s;opacity:{opacity:.2f}" aria-hidden="true">'
            f'<g filter="url(#{flt})"><g fill="{shade}">{"".join(bottom)}</g>'
            f'<g fill="{light}">{"".join(top)}</g></g></svg>')


def _hills(far, mid, near, rnd):
    def ridge(base, amp, n, phase):
        pts = []
        for i in range(n + 1):
            x = 1600 * i / n
            y = base - amp * (0.5 + 0.5 * math.sin(i * 0.9 + phase)) - rnd.uniform(0, amp * 0.35)
            pts.append(f"{x:.0f},{y:.0f}")
        return "M0,460 L" + " L".join(pts) + " L1600,460 Z"
    return (f'<svg class="sc-hills" viewBox="0 0 1600 420" preserveAspectRatio="none" aria-hidden="true">'
            f'<path d="{ridge(250, 90, 9, 0.3)}" fill="{far}" opacity="0.85" filter="url(#mcHill)"/>'
            f'<path d="{ridge(300, 70, 12, 1.7)}" fill="{mid}" filter="url(#mcHill)"/>'
            f'<path d="{ridge(350, 40, 16, 2.9)}" fill="{near}" filter="url(#mcHill)"/></svg>')


def _canopy(rnd, cx, cy, rx, ry, n, colors):
    out = []
    for _ in range(n):
        a = rnd.uniform(0, 2 * math.pi)
        rr = rnd.uniform(0, 1) ** 0.6
        x, y = cx + rx * rr * math.cos(a), cy + ry * rr * math.sin(a)
        r = rnd.uniform(16, 34)
        # lighter leaves on top (sun side), darker underneath
        shade = colors[0] if y < cy - ry * 0.2 else (colors[2] if y > cy + ry * 0.35 else colors[1])
        out.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r:.0f}" fill="{shade}"/>')
    return "".join(out)


def _tree(rnd, side, height_vh, sway_s, sway_deg, colors, trunk):
    """Tree = trunk + crown that sways, plus a highlight layer that flutters (leaves catching the wind)."""
    crown = _canopy(rnd, 200, 190, 150, 130, 95, colors)
    hi = _canopy(rnd, 185, 150, 120, 95, 35, (colors[0], colors[0], colors[1]))
    trunk_path = ('<path d="M190 520 C188 440 186 380 178 330 C168 290 140 262 112 240 M182 360 C200 320 232 290 268 262 '
                  'M184 300 C188 260 196 220 200 180" stroke="{t}" stroke-width="22" fill="none" stroke-linecap="round"/>'
                  '<path d="M190 520 C189 470 188 430 186 400" stroke="{t}" stroke-width="30" fill="none"/>').format(t=trunk)
    flip = "scaleX(-1)" if side == "right" else "scaleX(1)"
    common = f'height:{height_vh}vh;{side}:-3vw;--flip:{flip}'
    base = (f'<svg class="sc-tree" viewBox="0 0 400 520" style="{common};animation-duration:{sway_s:.2f}s;'
            f'--sway:{sway_deg:.1f}deg" aria-hidden="true">{trunk_path}<g filter="url(#mcLeaves)">{crown}</g></svg>')
    flutter = (f'<svg class="sc-tree sc-flutter" viewBox="0 0 400 520" style="{common};animation-duration:{sway_s * 0.45:.2f}s;'
               f'--sway:{sway_deg * 0.5 + 1.5:.1f}deg" aria-hidden="true"><g filter="url(#mcLeaves)" opacity="0.75">{hi}</g></svg>')
    return base + flutter


# ---------------------------------------------------------------- full scene
def scene_html(kind, wind_kmh, night=False, seed=11):
    rnd = random.Random(seed)
    kind = kind if kind in SKY else "cloudy"
    top, mid, bot = NIGHT_SKY if night else SKY[kind]
    far, hmid, near = NIGHT_HILLS if night else HILLS[kind]
    th = theme(kind, night)
    w = max(0.0, float(wind_kmh))
    sway_s = max(1.0, 6.0 - w / 6)          # stronger wind -> faster sway ...
    sway_deg = min(7.0, 0.8 + w / 5)        # ... and a wider swing
    cloud_dur = max(45, 220 - w * 6)        # clouds drift faster in wind
    slant = min(22, w * 0.7)                # rain slants with the wind

    p = [DEFS]
    # light source
    if night:
        stars = "".join(f'<i style="left:{rnd.uniform(0, 100):.1f}%;top:{rnd.uniform(0, 50):.1f}%;'
                        f'width:{rnd.choice([1, 1, 1.5, 2])}px;height:{rnd.choice([1, 1, 1.5, 2])}px;'
                        f'animation-delay:-{rnd.uniform(0, 5):.1f}s"></i>' for _ in range(120)) \
            if kind in ("sunny", "cloudy") else ""
        p.append(f'<div class="sc-stars">{stars}</div><div class="sc-moon"></div>')
    elif kind in ("sunny", "cloudy"):
        p.append('<div class="sc-sun"></div><div class="sc-glare"></div>')
        if kind == "sunny":
            p.append('<div class="sc-rays"></div>')

    # clouds: a soft high layer + fuller low clouds
    n_hi, n_lo = {"sunny": (3, 3), "cloudy": (4, 7), "drizzle": (4, 9), "rain": (5, 10), "storm": (5, 11)}[kind]
    light, shade = {"sunny": ("#ffffff", "#c9d6e2"), "cloudy": ("#f3f6f9", "#aebccb"), "drizzle": ("#dde3e8", "#98a5b1"),
                    "rain": ("#b9c2ca", "#6f7b87"), "storm": ("#6c7483", "#2e3542")}[kind]
    if night:
        light, shade = ("#5a6688", "#2b3552") if kind in ("sunny", "cloudy") else ("#3d465c", "#1d2333")
    for _ in range(n_hi):
        p.append(_cloud(rnd, rnd.uniform(0, 14), rnd.uniform(500, 900), cloud_dur * 1.6 * rnd.uniform(0.8, 1.2),
                        rnd.uniform(0, cloud_dur * 1.6), light, light, rnd.uniform(0.35, 0.6), True))
    for _ in range(n_lo):
        p.append(_cloud(rnd, rnd.uniform(4, 36), rnd.uniform(380, 720), cloud_dur * rnd.uniform(0.8, 1.25),
                        rnd.uniform(0, cloud_dur), light, shade, rnd.uniform(0.8, 0.97), False))
    if kind in ("rain", "storm"):
        p.append('<div class="sc-overcast"></div>')

    # landscape
    p.append(_hills(far, hmid, near, rnd))
    if night:
        leaves, trunk = ("#2a4a33", "#1d3826", "#12251a"), "#1e1712"
    elif kind in ("rain", "storm"):
        leaves, trunk = ("#4c8442", "#346a33", "#224a24"), "#3d2c20"
    else:
        leaves, trunk = ("#6aa850", "#3f7d3a", "#2a5a2a"), "#4b3627"
    p.append(_tree(rnd, "left", 52, sway_s, sway_deg, leaves, trunk))
    p.append(_tree(rnd, "right", 42, sway_s * 1.15, sway_deg * 0.85, leaves, trunk))
    p.append('<div class="sc-ground"></div>')

    # flying leaves when windy and dry
    if w >= 12 and kind in ("sunny", "cloudy", "drizzle"):
        n = int(min(16, 4 + (w - 12) / 2))
        p.append("".join(f'<span class="sc-fly" style="top:{rnd.uniform(40, 85):.0f}%;'
                         f'animation-duration:{max(3.0, 11 - w / 4) * rnd.uniform(0.8, 1.3):.1f}s;'
                         f'animation-delay:-{rnd.uniform(0, 10):.1f}s;background:{rnd.choice(leaves[:2])}"></span>'
                         for _ in range(n)))

    # rain in three depth layers + ground mist
    if kind in ("drizzle", "rain", "storm"):
        counts = {"drizzle": (30, 40, 0), "rain": (70, 80, 40), "storm": (90, 110, 60)}[kind]
        layers = [((10, 16), (1.0, 1.4), 0.35, 1.0),    # far: short, slow, faint
                  ((16, 28), (0.6, 0.9), 0.55, 1.4),    # middle
                  ((30, 48), (0.38, 0.55), 0.75, 2.0)]  # near: long, fast, bright
        for i, (count, (length, speed, op, width)) in enumerate(zip(counts, layers)):
            if not count:
                continue
            drops = "".join(
                f'<i style="left:{rnd.uniform(-15, 110):.1f}%;height:{rnd.uniform(*length):.0f}px;width:{width}px;'
                f'animation-duration:{rnd.uniform(*speed):.2f}s;animation-delay:-{rnd.uniform(0, 1.5):.2f}s"></i>'
                for _ in range(count))
            p.append(f'<div class="sc-rain sc-rain{i}" style="transform:rotate({slant:.0f}deg);opacity:{op}">{drops}</div>')
        p.append('<div class="sc-mist"></div>')
    if kind == "storm":
        p.append('<svg class="sc-bolt" viewBox="0 0 120 420" aria-hidden="true"><path d="M70 0 L40 150 L72 150 L30 300 '
                 'L64 300 L20 420" stroke="#f5f7ff" stroke-width="4" fill="none" stroke-linejoin="bevel"/></svg>'
                 '<div class="sc-flash"></div>')

    p.append('<div class="sc-grain"></div><div class="sc-vignette"></div>')

    dark = th["dark"]
    hdr = ("text-shadow: 0 1px 10px rgba(0,0,0,0.35);" if dark else "text-shadow: 0 1px 14px rgba(255,255,255,0.6);")
    g1, g2, gb = (0.16, 0.05, 0.22) if dark else (0.42, 0.16, 0.50)
    css = f"""
<style>
:root {{ --fg: {th['fg']}; --fg-muted: {th['muted']}; --g1: {g1}; --g2: {g2}; --gb: {gb};
  --tile: {'rgba(255,255,255,0.10)' if dark else 'rgba(255,255,255,0.34)'};
  --line: {'rgba(255,255,255,0.14)' if dark else 'rgba(20,33,58,0.08)'}; }}
.mc-name, .mc-date {{ {hdr} }}
.mc-scene {{position: fixed; inset: 0; z-index: -2; overflow: hidden; pointer-events: none;
  background: linear-gradient(180deg, {top} 0%, {mid} 50%, {bot} 78%, {bot} 100%);}}
.mc-scene > * {{will-change: transform;}}
.sc-sun {{position: absolute; right: 12%; top: 9%; width: 90px; height: 90px; border-radius: 50%;
  background: radial-gradient(circle, #ffffff 0%, #fffbe6 35%, #ffe9a8 60%, rgba(255,233,168,0) 72%);
  box-shadow: 0 0 80px 40px rgba(255,244,200,0.55), 0 0 260px 140px rgba(255,220,140,0.28);}}
.sc-glare {{position: absolute; inset: 0;
  background: radial-gradient(circle at 86% 14%, rgba(255,250,225,0.45) 0%, rgba(255,250,225,0) 38%),
              radial-gradient(circle at 60% 40%, rgba(255,255,255,0.10) 0 3%, transparent 3.5%),
              radial-gradient(circle at 48% 54%, rgba(255,230,180,0.08) 0 5%, transparent 5.5%);}}
.sc-rays {{position: absolute; right: calc(12% - 360px); top: calc(9% - 360px); width: 810px; height: 810px;
  background: repeating-conic-gradient(from 0deg, rgba(255,250,220,0.10) 0 4deg, transparent 4deg 14deg);
  -webkit-mask: radial-gradient(circle, #000 8%, transparent 62%); mask: radial-gradient(circle, #000 8%, transparent 62%);
  animation: sc-spin 140s linear infinite;}}
.sc-moon {{position: absolute; right: 12%; top: 9%; width: 70px; height: 70px; border-radius: 50%;
  background: radial-gradient(circle at 38% 36%, #fffef6 0%, #ece7d2 55%, #cbc3a6 100%);
  box-shadow: 0 0 50px 14px rgba(230,236,255,0.25), 0 0 180px 80px rgba(150,170,230,0.12);}}
.sc-stars {{position: absolute; inset: 0;}}
.sc-stars i {{position: absolute; border-radius: 50%; background: #fff; animation: sc-twinkle 5s ease-in-out infinite;}}
.sc-cloud {{position: absolute; left: -950px; animation: sc-drift linear infinite;}}
.sc-overcast {{position: absolute; inset: 0 0 40% 0; background: linear-gradient(180deg, rgba(40,48,60,0.35), rgba(40,48,60,0));}}
.sc-hills {{position: absolute; left: 0; right: 0; bottom: 0; width: 100%; height: 32vh;}}
.sc-ground {{position: absolute; left: 0; right: 0; bottom: 0; height: 5vh;
  background: linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,0,0,0.25) 100%);}}
.sc-tree {{position: absolute; bottom: -1vh; transform-origin: 47% 100%; animation: sc-sway ease-in-out infinite alternate;
  filter: drop-shadow(0 10px 18px rgba(0,0,0,0.25));}}
.sc-flutter {{filter: none;}}
.sc-fly {{position: absolute; left: -40px; width: 13px; height: 7px; border-radius: 80% 0 80% 0; opacity: 0.9;
  animation: sc-blow linear infinite;}}
.sc-rain {{position: absolute; inset: -15% -15% 0 -15%; transform-origin: top center;}}
.sc-rain i {{position: absolute; top: -60px; border-radius: 2px;
  background: linear-gradient(180deg, rgba(220,232,245,0) 0%, rgba(225,236,248,0.9) 100%); animation: sc-fall linear infinite;}}
.sc-rain0 i {{filter: blur(0.6px);}}
.sc-mist {{position: absolute; left: -20%; right: -20%; bottom: 0; height: 40vh;
  background: radial-gradient(ellipse at 30% 100%, rgba(220,228,236,0.45), transparent 60%),
              radial-gradient(ellipse at 75% 100%, rgba(220,228,236,0.35), transparent 55%);
  animation: sc-mist 26s ease-in-out infinite alternate;}}
.sc-bolt {{position: absolute; left: 62%; top: 4%; height: 46vh; opacity: 0;
  filter: drop-shadow(0 0 8px #cfe0ff) drop-shadow(0 0 24px #9fbfff); animation: sc-bolt 9s infinite;}}
.sc-flash {{position: absolute; inset: 0; background: #e8efff; opacity: 0; animation: sc-flash 9s infinite;}}
.sc-grain {{position: absolute; inset: 0; opacity: 0.07; mix-blend-mode: overlay;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='g'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/></filter><rect width='100%' height='100%' filter='url(%23g)'/></svg>");}}
.sc-vignette {{position: absolute; inset: 0; background: radial-gradient(ellipse at 50% 40%, transparent 55%, rgba(0,0,0,0.22) 100%);}}
@keyframes sc-spin {{to {{transform: rotate(360deg);}}}}
@keyframes sc-drift {{from {{transform: translateX(0);}} to {{transform: translateX(calc(100vw + 1900px));}}}}
@keyframes sc-fall {{from {{transform: translateY(0);}} to {{transform: translateY(120vh);}}}}
@keyframes sc-sway {{from {{transform: var(--flip) rotate(calc(var(--sway) * -0.6));}} to {{transform: var(--flip) rotate(var(--sway));}}}}
@keyframes sc-twinkle {{0%,100% {{opacity: 0.25;}} 50% {{opacity: 1;}}}}
@keyframes sc-mist {{from {{transform: translateX(-4%);}} to {{transform: translateX(4%);}}}}
@keyframes sc-flash {{0%,86%,100% {{opacity: 0;}} 87% {{opacity: 0.55;}} 88% {{opacity: 0.05;}} 89.5% {{opacity: 0.4;}} 91% {{opacity: 0;}}}}
@keyframes sc-bolt {{0%,86%,100% {{opacity: 0;}} 87% {{opacity: 1;}} 88% {{opacity: 0.2;}} 89.5% {{opacity: 0.9;}} 91% {{opacity: 0;}}}}
@keyframes sc-blow {{0% {{transform: translate(0,0) rotate(0);}} 50% {{transform: translate(55vw,-10vh) rotate(380deg);}}
  100% {{transform: translate(112vw,5vh) rotate(760deg);}}}}
@media (prefers-reduced-motion: reduce) {{ .mc-scene * {{animation: none !important;}} }}
@media (max-width: 760px) {{ .sc-tree {{height: 30vh !important;}} .sc-hills {{height: 22vh;}} }}
</style>"""
    return css + f'<div class="mc-scene mc-{kind}{" mc-night" if night else ""}">{"".join(p)}</div>'