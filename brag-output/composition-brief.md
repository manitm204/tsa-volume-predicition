# Hyperframes Composition Brief: TSA · EDGE

## Objective
Create a short launch-style brag video for TSA · EDGE.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 20 seconds

## Source Material
- Project root: `/home/manit/Desktop/fun_projects/tsa`
- Primary files read: `dashboard/frontend/src/index.css`, `dashboard/frontend/src/components/SignalHero.tsx`, `ProbabilityLadder.tsx`, `MarketsPanel.tsx`, `layout/Sidebar.tsx`
- Product name: TSA · EDGE (subtitle: "Quant Terminal")
- Tagline / strongest claim: A trading terminal that forecasts TSA airport-security checkpoint volume, compares it to Kalshi market prices, and outputs a graded, Kelly-sized trade
- Key UI or visual moment to recreate: the `SignalHero` card — animated conic-gradient neon border, giant gradient-text trade call ("BUY YES 2.85M"), BUY YES badge + Edge/EV chips, confidence bar, A+ grade pill, Kelly Size dollar counter — plus the `ProbabilityLadder` rows (model bar vs. dashed Kalshi marker, edge % readout)
- Copy that must appear verbatim:
  - "THIS IS A TRADING TERMINAL."
  - "FOR AIRPORT SECURITY LINES."
  - "BUY YES 2.85M"
  - "TSA · EDGE"
  - "Quant Terminal"
  - "PAST PERFORMANCE ≠ FUTURE RESULTS."
  - "neither is TSA wait times."

## Creative Direction
- Tone preset: yc-parody
- Creative direction: fake-serious fintech launch demo — deadpan confidence, zero self-awareness about the premise
- Interpretation: Structured 5-scene pacing, hard cuts only (no crossfades), big confident display type held long enough to read, product UI treated with total institutional seriousness
- Angle: All the machinery of a serious trading terminal — signal grades, Kelly sizing, confidence bars, order books, edge highlighting — pointed at TSA checkpoint throughput, played completely straight
- Hook: Black screen, "THIS IS A TRADING TERMINAL." slams in, beat, "FOR AIRPORT SECURITY LINES." slams in below
- Outro / punchline: "PAST PERFORMANCE ≠ FUTURE RESULTS." then smaller deadpan "neither is TSA wait times." then logo card
- Avoid:
  - Generic SaaS language
  - Abstract filler visuals
  - Unrelated visual redesign — must match the existing dark/glass/emerald aesthetic exactly

## Visual Identity
- Background: `#03060e` (near-black navy)
- Text: `#e2e8f0` body / `#cbd5e1` secondary; gradient-text emerald `#60a5fa → #818cf8 → #22d3a4` for hero display numbers
- Accent: emerald `#22d3a4` (buy/positive), blue `#60a5fa`, amber `#f59e0b` (Kalshi marker / warn), red `#ef5466` (sell/negative)
- Glass cards: `linear-gradient(145deg, rgba(13,21,38,0.85), rgba(8,13,24,0.85))`, border `rgba(99,140,255,0.10)`, `backdrop-filter: blur(14px)`, radius 16-20px
- Display font: Space Grotesk (headlines, hero numbers, grade pill)
- Body font: Inter; monospace figures in JetBrains Mono
- Visual references from the project: `.neon-border` animated conic-gradient sweep (emerald → blue → violet → emerald), `.grade-aplus` glowing pill (`linear-gradient(135deg,#0d503e,#11785e)` with emerald glow shadow), `.badge-buy` (emerald, glowing), `.ladder-fill` gradient bar with amber Kalshi tick marker

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:
1. The Claim — 3s — black screen, two stacked headline slams (white then emerald), must be fully readable
2. Reveal — 3s — dashboard sidebar snaps in (TSA · EDGE brand + nav list), Signal Hero card's neon-conic border animates its sweep in as the card materializes
3. The Trade Call — 5s — "BUY YES 2.85M" gradient-text slam, BUY YES badge + Edge/EV chips pop in, confidence bar fills
4. Receipts — 5s — A+ grade pill pops with glow, Kelly dollar figure counts up, whip to Probability Ladder rows stagger-revealing with amber Kalshi marker crossing each bar
5. Punchline / Outro — 4s — black screen, "PAST PERFORMANCE ≠ FUTURE RESULTS." then smaller "neither is TSA wait times.", then logo card (Zap mark + "TSA · EDGE — Quant Terminal")

## Audio
- Audio role: dense rhythmic layer, restrained — energetic bed under a deadpan script
- Audio arc: quiet/filtered under the black-screen hook → opens to full energy at the dashboard reveal → holds through trade-call and receipts scenes (highest density) → drops out almost entirely for the punchline → small final swell under the logo card
- Music: `happy-beats-business-moves-vol-1-by-ende-dot-app.mp3`
- Music treatment: muted/filtered start, full volume from the Scene 2 reveal, hard duck under the Scene 5 punchline line, small swell back up under the logo card, fade out at the end
- Music cue guidance: bundled preset for this track (120.19 BPM). Early scenes (0-6s) precede the track's densest cue cluster (16-25s) — treat the beat grid (~0.5s spacing from 3.02s onward) as the timing reference for the Scene 2 reveal and the Scene 4 sequential ladder-row/counter reveals rather than snapping to a specific named strongCue. Use natural timing if it reads better.
- Audio-reactive treatment: subtle — the Signal Hero neon border's glow intensity may breathe faintly with music energy while the card is on screen; nothing else reacts.
- Audio-coupled moments:
  - Scene 1 headline slams — one dry low-key hit per line
  - Scene 2 hero card reveal — lands on a strong beat, music opens to full volume here
  - Scene 3 badge pop + confidence-bar fill — motion-matched to beat ticks
  - Scene 4 grade pill pop + Kelly counter count-up — landed on-beat; ladder rows tick in sequentially on the beat grid, each held long enough to read
  - Scene 5 punchline — music ducks hard under the deadpan line; small musical accent when the logo card lands
- SFX selection guidance: sparse and motion-matched — soft synth hit on hard cuts/text slams, light data-tick on the grade pill and Kelly counter, subtle tick per ladder row (quantized, not one-per-row spam). No cartoon SFX, no laugh-track stings, no waveform/equalizer visuals — the joke is that the audio treats this exactly like a real fintech launch video.
- SFX analysis guidance: use `sfx-analysis.md` from the brag skill assets; prefer lower high-frequency-risk sounds for the repeated ladder-row ticks.
- Exact SFX choice: Hyperframes chooses exact filenames/timestamps/density from the implemented animation.
- Audio files: copy `happy-beats-business-moves-vol-1-by-ende-dot-app.mp3` into `brag-output/composition/assets/music/`.

## Hyperframes Instructions
Load the composition-building Hyperframes domain skills — `hyperframes-core`, `hyperframes-animation`, `hyperframes-creative`, `hyperframes-keyframes`, `hyperframes-cli`. This is the `/brag` workflow: do not enter the `hyperframes` entry-point intent interview and do not route into its generic promo/launch-video workflow. Prefer native Hyperframes conventions over anything in `/brag`.

Requirements:
- Show at least one real UI, copy, or visual element from the source project (Signal Hero card, Probability Ladder, TSA · EDGE brand mark).
- Keep all text readable in the final render.
- Keep the video within 15-25 seconds (target 20s).
- Include the planned music/SFX layer.
- Treat `/brag` audio notes as guidance, not a fixed cue sheet; choose SFX after the visual animation exists.
- Treat music cue metadata as optional timing hints; ignore cues that hurt readability, scene pacing, or story.
- Use 1-3 strong cue locks across the video; align Scene 4's sequential reveals to the beat grid, holding each row long enough to read.
- Use SFX to support motion/interaction with restraint given the deadpan yc-parody tone.
- Honor the planned music treatment: quiet start, full-volume reveal, hard duck under the punchline, small swell on the logo.
- Use audio-reactive treatment on the neon border glow if extraction is available; otherwise skip and note it.
- Run `hyperframes check` before render — it is brag's single gate.
