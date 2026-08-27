# Fall-warning dataset (assembled 2026-08-27)

This directory is a **reorganized academic collection** for an elderly fall-warning prototype.
It is **not** a newly captured elderly-home corpus.

**Assembled / last merged:** 2026-08-27 14:38 UTC+08:00 (UTC+8)
**Root:** `/workspace/fall-warning-data/data`

## Important deviation (read this first)

Public fall-detection video sets are almost entirely **young / healthy actors performing simulated falls**. **We cannot film real elderly people.** No clip here is a confirmed real-world elderly fall.

| Spec need | What we actually have |
|---|---|
| Real elderly subjects | **None confirmed.** CAUCAFall paper: mixed ages; per-subject age not in Dataset details.xlsx (left unknown). |
| Bathroom slip | **None in official academic zips.** Bathroom not invented from living-room footage. |
| Bed-exit fall | **Partial.** GMDCSA descriptions mentioning a bed mapped to bedroom. Not a clinical bed-exit protocol. |
| Night IR / night wandering | **Partial night lighting.** CAUCAFall S6/S8 = 0 lux; GMDCSA CSV Night. Academic zips have no wandering class. |
| >=1280x720 | **Partial.** URFD 640x240; Le2i ~320x240; CAUCAFall official AVI **720x480** (paper camera 1080x960; not upsampled). GMDCSA typically 1280x720. |
| Clips 5 s-3 min | **Partial.** Many falls are 2-8 s (original length kept). |
| Normal >= 2 hours | **Not met.** Short takes, not continuous ADL. |

Empty category folders contain only `.gitkeep`.

## Counts (scene x action_label)

| scene | action_label | n_clips |
|---|---|---|
| livingroom | fall | 280 |
| livingroom | nearfall | 54 |
| livingroom | normal | 115 |
| livingroom | risk_behavior | 3 |
| bedroom | fall | 24 |
| bedroom | nearfall | 17 |
| bedroom | normal | 24 |
| bedroom | risk_behavior | 0 |
| bathroom | fall | 1 |
| bathroom | nearfall | 0 |
| bathroom | normal | 1 |
| bathroom | risk_behavior | 0 |

**Total clips:** 519
**Total duration:** 5067.0 s (84.5 min)
**Duration by label:** fall=2744.0s, nearfall=709.6s, normal=1541.5s, risk_behavior=72.0s
**Disk (data/):** 1.3G
**Night clips (lighting=night):** 50
**Clips >=720p:** 168
**Empty categories (pre-ncam table):** bathroom/nearfall, bathroom/risk_behavior. After ncam: bedroom/risk_behavior has 1 clip (see Wall-cam section).

Sources: {'GMDCSA-24': 160, 'Le2i': 130, 'URFD': 100, 'CAUCAFall': 100, 'WebVideo': 29}
Falls/nearfalls with official timestamps: 300; falls with timestamps left blank (not invented): 32
Frame-bbox JSON files: 230
Copied IMU/sync CSVs: 200

## Obtained sources

1. **URFD** — CC BY-NC-SA 4.0. https://fenix.ur.edu.pl/~mkepski/ds/uf.html — Kwolek & Kepski, CMPB 2014, doi:10.1016/j.cmpb.2014.09.005
2. **Le2i / IMVIA** — CC BY-NC-SA. doi:10.25666/DATAUBFC-2024-04-09 — Home + Coffee room only.
3. **CAUCAFall** — CC BY 4.0. Mendeley v5 doi:10.17632/7w7fccy7ky.5 — 100 AVI + YOLO txt (PNGs skipped). Official AVI 720x480. Lighting from Dataset details.xlsx: S1-4 natural; S5,7,9,10 artificial; S6 and S8 0 lux = night. Hop = normal; kneel/sit/pick-up = nearfall; scene livingroom. Paper: Data in Brief 45 (2022) 108610.
4. **GMDCSA-24** — CC BY 4.0. Zenodo v2.1 doi:10.5281/zenodo.13354453 — 160 clips, 4 actors, 3 homes, typically 1280x720. Bedroom iff description mentions a bed. Night from CSV Time of Recording. Paper: Data in Brief 2024 110892.

## Skipped (not pirated)

Multiple Cameras Fall Dataset (Anubis bot-wall); UP-Fall (Google Drive login); TST v2 (IEEE DataPort login); TsetFall rar (optional, skipped).

## Mapping / timestamps / boxes

- Scenes: livingroom / bedroom / bathroom only. Bathroom never invented. CAUCAFall -> livingroom. GMDCSA bedroom only when a bed is named.
- Timestamps never fabricated. CAUCAFall fall times from YOLO class 1 frames. GMDCSA from CSV class spans. URFD cam1 blank.
- Boxes: Le2i official xyxy; CAUCAFall YOLO->xyxy; URFD/GMDCSA none.
- ffmpeg: remux H.264 copy; transcode AVI libx264, no scale.

## Remaining gaps

Bathroom slips in official zips; clinical bed-exit protocol (bedroom fall clips = 24); night wandering / risk_behavior in academic zips; real elderly; 2 h inactivity; uniform >=720p.

URFD/Le2i are CC BY-NC-SA; CAUCAFall/GMDCSA-24 are CC BY 4.0. Non-commercial academic/prototype use unless NC sources are dropped. Cite original authors.

## Wall-cam / SmartHome-Bench (assembled 2026-08-27)

Public **smart-home / wall-mounted indoor camera** clips from **SmartHome-Bench** (Zhao et al., arXiv:2506.12992). Filenames use an `ncamNN` number slot so they do **not** collide with URFD/Le2i/CAUCA/GMDCSA/`wNN` sequences. Existing rows/files were **appended to**, not replaced.

- Paper: https://arxiv.org/abs/2506.12992
- Repo: https://github.com/Xinyi-0724/SmartHome-Bench-LLM
- URL list: https://github.com/Xinyi-0724/SmartHome-Bench-LLM/blob/main/Videos/Video_url.csv
- Annotations: https://github.com/Xinyi-0724/SmartHome-Bench-LLM/blob/main/Videos/Video_Annotation.csv
- HuggingFace card: https://huggingface.co/datasets/violetcliff/SmartHome-Bench

Filter: indoor home rooms (living room, bedroom, kitchen/dining mapped to livingroom, hallway mapped to livingroom). Skipped baby-only, pet-only, outdoor doorbell/porch/street, ceiling-only office, thermal, first-person, 3D, dashcam, children as primary subjects, voyeuristic bathroom results.

JSON notes include source URL, `people_count` (`1p` or `2p+`), `camera_angle=wall_mount`. `person_bbox` is null. Fall timestamps are approximate (`0.3` / `duration-0.2`), not frame-accurate. Lighting is `daylight` / `night` / `unknown`. ffmpeg H.264, original resolution kept, audio stripped, long sources split to 10–40 s.

### NCAM-only counts (this section)

| scene | action_label | people_count | n_clips |
|---|---|---|---|
| livingroom | normal | 1p | 11 |
| livingroom | normal | 2p+ | 3 |
| livingroom | fall | 2p+ | 3 |
| livingroom | nearfall | 1p | 2 |
| livingroom | risk_behavior | 1p | 1 |
| bedroom | risk_behavior | 2p+ | 1 |
| bathroom | (any) | | 0 |

**NCAM clips:** 21 (all from SmartHome-Bench public YouTube; extra YouTube search downloads hit bot-check). Duration ~508 s (~8.5 min).

### Source URLs (provenance)

SmartHome-Bench IDs used (one source may yield multiple event clips):

- `smartbench_1002` `3-dvsT8pqbQ` — living-room elderly trip + walking pre-roll — https://www.youtube.com/watch?v=3-dvsT8pqbQ
- `smartbench_0022` `iF3N61nr6JA` — Ring indoor living room man+dog; indoor portion only — https://www.youtube.com/watch?v=iF3N61nr6JA
- `smartbench_1008` `wCehkLbs2q4` — indoor stairwell elderly couple fall/walk — https://www.youtube.com/watch?v=wCehkLbs2q4
- `smartbench_0449` `LQ4ugbhm_Tw` — Ring IR night, woman lying on living-room floor — https://www.youtube.com/watch?v=LQ4ugbhm_Tw
- `smartbench_1013` `HbKd881dcy8` — caregiver + elderly walker/rollator in bedroom — https://www.youtube.com/shorts/HbKd881dcy8
- `smartbench_1012` `o_RzQo8ci-Q` — elderly woman walking with cane indoors — https://www.youtube.com/watch?v=o_RzQo8ci-Q
- `smartbench_0388` `XK1NSHrPuIQ` — Ring indoor living room sit/stand — https://www.youtube.com/watch?v=XK1NSHrPuIQ
- `smartbench_0259` `_Psi3NnuNiw` — Ring indoor living/kitchen two people — https://www.youtube.com/watch?v=_Psi3NnuNiw
- `smartbench_0036` `c4BiMNV5dME` — Ring indoor woman on couch — https://www.youtube.com/watch?v=c4BiMNV5dME
- `smartbench_0307` `evVrht5KnYI` — Ring indoor living room standing — https://www.youtube.com/watch?v=evVrht5KnYI
- `smartbench_0962` `7gSYic4_JBA` — sitting on sofa at night (400x224 kept) — https://www.youtube.com/watch?v=7gSYic4_JBA
- `smartbench_0661` `g5P1hj_cUyA` — Ring indoor walking living/dining — https://www.youtube.com/watch?v=g5P1hj_cUyA
- `smartbench_0983` `gTt5XatfDpc` — wall-cam home office/study walking — https://www.youtube.com/watch?v=gTt5XatfDpc
- `smartbench_0336` `vrMiyFpCky8` — Ring indoor kitchen/dining standing — https://www.youtube.com/watch?v=vrMiyFpCky8
- `smartbench_0100` `vP_TtmwnK2Q` — indoor hallway walking — https://www.youtube.com/watch?v=vP_TtmwnK2Q
- `smartbench_0189` `wvbu9zKQpoo` — indoor hallway walking — https://www.youtube.com/watch?v=wvbu9zKQpoo

### Failures / skipped

- **YouTube bot-check** after the first SmartHome-Bench batch: `8aPMaaGPmr8` (elderly man walker + caregiver hallway, 2p+; high-value miss) and all secondary `ytsearch*` downloads (`pc8eQZW9E8Q`, `2vsnfzzBz0k`, `R2oRP8Jcpro`, `YNd8jNrCASU`, `bZxZE_imKlA`, `Ixjy2jDj6tg`, `Fr-VlFvmHx4`, `yweiixlWbs0`, `8YjvUFX6A_I`, `ndD0nw4deB0`, Kami/Eufy/Wyze sample-footage IDs, etc.). Skipped rather than using cookies.
- **Visual skip (not wall-mount 15–40° into a room):** `kHsRbQW7_ac` close-up bed-exit; `PSouFxTaIf8` / `Q2sa-BnIc34` eye-level portraits of elderly sitting; `y1QitvnIWm4` chest-height phone-like view.
- **CSV filter skip:** baby/kid primary, pet-only, outdoor doorbell/porch/wildlife, ceiling office, burglary hallway, Santa costume, mall escalator.
- **Bathroom search:** `ytsearch5:bathroom home camera person walking` returned non-home / joke / sexual-adjacent titles — **none downloaded**.

### Remaining gaps (wall-cam)

- **Bathroom** still 0 from this pass (no usable fixture-and-person wall-cam after safety filter).
- **Bedroom** only 1 clip (caregiver+walker); no bedroom normal ADL or bed-exit fall from this pass.
- **2p+ normal ADL** only 3 clips; target was ~15.
- Extra YouTube sample-footage (YI/Eufy/Blink/Wyze living-room walking) blocked by bot-check; would have filled 1p normal 720p+.
- Many SmartHome-Bench public clips are outdoor Ring doorbell / wildlife / pets, so indoor-adult yield is thin even before bot-check.
