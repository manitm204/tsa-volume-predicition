# Brag Plan: TSA · EDGE

## What is this app?
A "quant terminal" dashboard that forecasts weekly U.S. airport TSA checkpoint volume with a trained model, compares it against live Kalshi prediction-market prices, and outputs a graded, Kelly-sized trade recommendation — for a market about how many people go through airport security.

## The angle
Play it 100% straight, like a real fintech product demo. The joke is that all the machinery of a serious trading terminal — signal grades, Kelly sizing, confidence bars, order books, edge highlighting — has been pointed at TSA checkpoint throughput. Nobody in the video acknowledges this is funny. That's the bit.

## Hook (first 2-3 seconds)
Hard cut to black. Bold display type slams in: "THIS IS A TRADING TERMINAL." Beat. Second line slams under it: "FOR AIRPORT SECURITY LINES." Dead stop, no music swell yet — let the absurdity sit for a beat before the terminal reveals.

## Key moments (the middle)
- The **Signal Hero** materializes: emerald neon-conic border animates in, then the giant gradient-text recommendation "BUY YES 2.85M" slams in with its BUY YES badge, Edge/EV chips, and the confidence bar filling.
- The **Signal Strength grade pill** (A+) pops in next to it with its emerald glow, and the "Kelly Size" dollar figure counts up in the glass panel beside it — this is the "we have a system" money shot.
- The **Probability Ladder** rows sweep in one by one (staggered), model bars filling left-to-right in emerald/blue with the dashed amber Kalshi marker crossing each bar, edge percentages appearing on the right.

## Outro / punchline
Cut back to black. Type slams in: "PAST PERFORMANCE ≠ FUTURE RESULTS." Beat. Smaller line underneath, deadpan: "neither is TSA wait times." Logo/brand card: "TSA · EDGE — Quant Terminal" with the emerald Zap mark, then out.

## User flow worth showing
1. Model produces a weekly volume forecast vs. Kalshi's implied market price (Signal Hero comparison stats: Model Weekly Avg vs Kalshi Implied vs Edge).
2. System grades the opportunity and sizes a real trade (Grade pill + Kelly Size + BUY YES/SELL recommendation).
3. Model vs market is shown across every strike price at once (Probability Ladder bars with edge highlighting) — the receipts for why the trade was made.

## Tone
- Preset: yc-parody
- Creative direction: fake-serious fintech launch demo — deadpan confidence, zero self-awareness about the premise
- Interpretation: Structured 4-5 scene pacing, minimal hard cuts (no crossfades), big confident display type held long enough to read, product UI treated with total institutional seriousness.

## Format: landscape — 1920x1080
## Duration: 20s

## Visual identity (from the project)
- Background: `#03060e` (near-black navy), glass cards `hsl(222 20% 8%)` w/ blue-tinted borders `rgba(99,140,255,0.10-0.20)`
- Accent: emerald `#22d3a4` (primary signal/buy color), secondary blue `#60a5fa`, warn amber `#f59e0b` (Kalshi marker), danger red `#ef5466` (sell/down)
- Text: `#e2e8f0` / `#cbd5e1` body, gradient-text emerald (`#60a5fa → #818cf8 → #22d3a4`) for hero numbers
- Display font: Space Grotesk (headlines/display numbers)
- Body font: Inter; mono figures in JetBrains Mono
- Strongest visual element: the `.neon-border` animated conic-gradient glow around the Signal Hero card, and the emerald "grade pill" (A+) with glow shadow

## Share copy (draft)
I built a quant trading terminal for TSA checkpoint lines. It has Kelly sizing, a signal grade, and an order book. It is unreasonably serious about airport security.

## Audio direction
- Role: dense rhythmic layer, restrained — energetic bed under a deadpan script
- Music: `happy-beats-business-moves-vol-1-by-ende-dot-app.mp3` (120.19 BPM, upbeat corporate-electronic)
- Music treatment: starts quiet/muted under the black-screen hook, opens up once the terminal reveals at ~3s, steady through highlights, drops out briefly for the outro's deadpan punchline, one final swell under the logo card
- Music cue guidance: preset read from bundled track. Strong cues at 3.02s/4.02s/5.03s available near the reveal; using scene-level alignment rather than exact snap since this video's early scenes (0-6s) precede the track's detailed 17-25s cue cluster — treat 3-6s beats as the reveal anchor, align probability-ladder row reveals to the ~0.5s beat grid, hold each row visible across at least one full beat before the next arrives.
- Audio-reactive treatment: subtle — the neon border's glow intensity may breathe faintly with the beat once the hero card is on screen; nothing else reacts.
- SFX posture: sparse, motion-matched — a soft synth "hit" on each hard cut/text slam, a light data-tick on the grade pill and Kelly counter, a subtle ladder "tick" per row reveal (quantized, not literal per-row sound spam)
- Audio-coupled moments: hook lines slam on downbeats; grade pill pop and Kelly counter land on a strong beat; ladder rows stagger-reveal on the beat grid, holding each row long enough to read the strike/edge numbers
- Restraint rule: no laugh-track cues, no cartoon SFX, no waveform visualizations — the joke is that the audio treats this exactly like a real fintech launch video

## Storyboard

### Scene 1 — The Claim — 3s
Black screen. "THIS IS A TRADING TERMINAL." slams in (display, huge, white). Beat (~0.6s hold). "FOR AIRPORT SECURITY LINES." slams in below it, smaller, in emerald.
Sequential/interaction: yes — two lines slam in sequentially, each held to read (line 1 ~1.2s, line 2 ~1.4s)
Audio intent: dry, confident, almost silent — one dull thud per line, music entering muted/filtered underneath
Audio-coupled idea: each line slam gets a single low-key hit; no music vocal drop yet
Music: enters muted/filtered under scene, opens up at scene transition
Transition mood: hard cut → Scene 2

### Scene 2 — Reveal — 3s
Hard cut to the full dashboard: dark sidebar with "TSA · EDGE / Quant Terminal" brand mark visible, nav items listed (Command, Tomorrow, Week, Model, Ensemble). Signal Hero card's neon-conic border animates its sweep in as the card materializes.
Sequential/interaction: yes — sidebar snaps in first, then the hero card border animates its conic sweep as it fades/scales up
Audio intent: music opens up full volume here — this is the "reveal" beat
Audio-coupled idea: hero card entrance lands on a strong beat
Music: full energy kicks in
Transition mood: clean cut → Scene 3

### Scene 3 — The Trade Call — 5s
Inside the Signal Hero: "BUY YES 2.85M" slams in as huge emerald gradient-text, BUY YES badge and Edge/EV chips pop in beside it, confidence bar fills left-to-right.
Sequential/interaction: yes — headline first, then badge+chips pop in together, then confidence bar fills over ~0.8s
Audio intent: build energy, purposeful
Audio-coupled idea: badge pop and bar-fill motion-matched to beat ticks
Music: steady beat, building
Transition mood: hard cut (whip) → Scene 4

### Scene 4 — Receipts — 5s
Grade pill (A+) pops in with emerald glow next to the Kelly Size panel; the Kelly dollar figure counts up (e.g. $0 → $62). Cut/whip to Probability Ladder: 4-5 rows stagger-reveal, bars filling with the amber Kalshi marker crossing each, edge % landing on the right.
Sequential/interaction: yes — grade pill + counter first (both land on-beat), then ladder rows reveal one by one on the beat grid, each held to read
Audio intent: peak energy and confidence — the "proof" moment
Audio-coupled idea: counter tick sound during count-up; soft tick per ladder row arrival
Music: full beat, most rhythmic density of the video
Transition mood: hard cut → Scene 5

### Scene 5 — Punchline / Outro — 4s
Cut to black. "PAST PERFORMANCE ≠ FUTURE RESULTS." slams in, holds. Music drops out almost entirely. Smaller deadpan line fades in beneath: "neither is TSA wait times." Cut to logo card: emerald Zap mark + "TSA · EDGE — Quant Terminal," then out.
Sequential/interaction: yes — headline, then smaller punchline line, then logo card, each sequential
Audio intent: the music drop is the joke's timing — silence sells the deadpan line, then one final swell under the logo
Audio-coupled idea: music ducks hard under the punchline line, one small musical accent when the logo card lands
Music: near-silent, then final soft swell on logo
Transition mood: hard cut → hard cut → end

**Music mood for this video:** parody (deadpan corporate-electronic)
**Audio summary:** An upbeat 120 BPM corporate-electronic bed that opens with restraint under the hook, hits full energy through the dashboard reveal and trade-call/receipts scenes, then drops out almost entirely to sell the deadpan punchline before a small final swell on the logo card.
