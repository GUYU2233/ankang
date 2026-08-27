# Fall-warning dataset (assembled 2026-08-27)

This directory is a **reorganized academic collection** for an elderly fall-warning prototype.
It is **not** a newly captured elderly-home corpus.

**Assembled:** 2026-08-27 13:10 UTC+08:00 (UTC+8)
**Root:** `/workspace/fall-warning-data/data`

## Important deviation (read this first)

Public fall-detection video sets are almost entirely **young / healthy actors performing simulated falls** in labs or mocked-up rooms. **We cannot film real elderly people.** No clip here is a confirmed real-world elderly fall.

| Spec need | What we actually have |
|---|---|
| Real elderly subjects | **None.** Demographics unknown in official files; literature describes young volunteers. |
| Bathroom slip | **None.** No public clip in the obtained sets is a true bathroom scene. Folder left empty. |
| Bed-exit fall | **None officially labeled.** No `fall_from_bed` ground truth. Bedroom folder empty. |
| Night IR / night wandering | **None.** `risk_behavior` folders empty. |
| ≥1280×720 | **Not met.** URFD Kinect RGB is **640×240**; Le2i is **320×240**. Files were **not upsampled** (that would fake detail). |
| Clips 5 s–3 min | **Partially met.** Many URFD falls are 2–5 s (original length kept). |
| Normal ≥ 2 hours | **Not met.** Public sets are short takes, not continuous ADL. See totals below. |

Empty category folders contain only `.gitkeep`.

## Counts (scene × action_label)

| scene | action_label | n_clips |
|---|---|---|
| livingroom | fall | 159 |
| livingroom | nearfall | 16 |
| livingroom | normal | 55 |
| livingroom | risk_behavior | 0 |
| bedroom | fall | 0 |
| bedroom | nearfall | 0 |
| bedroom | normal | 0 |
| bedroom | risk_behavior | 0 |
| bathroom | fall | 0 |
| bathroom | nearfall | 0 |
| bathroom | normal | 0 |
| bathroom | risk_behavior | 0 |

**Total clips:** 230
**Total duration:** 2204.5 s (36.7 min)
**Duration by label:** fall=1387.6s, nearfall=149.9s, normal=667.0s
**Disk (data/):** 140M

Sources: {'Le2i': 130, 'URFD': 100}
Falls with official timestamps: 145; falls with timestamps left blank (not invented): 30
Le2i official frame-bbox JSON files: 130
Copied IMU/sync CSVs: 200

## What was downloaded vs failed

### Obtained

1. **UR Fall Detection Dataset (URFD)** — University of Rzeszow
   - Page: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
   - Files: official `fall-*-cam0.mp4`, `fall-*-cam1.mp4`, `adl-*-cam0.mp4`, `*-data.csv`, `*-acc.csv`, `urfall-cam0-falls.csv`, `urfall-cam0-adls.csv`
   - **Not downloaded:** RGB/depth PNG zip sequences (MP4 already provided by the authors)
   - Scale: 30 falls × 2 cameras + 40 ADL = 100 clips
   - License: **CC BY-NC-SA 4.0**, non-commercial academic use. Commercial use requires contacting mkepski@ur.edu.pl
   - Citation: Bogdan Kwolek, Michal Kepski, *Human fall detection on embedded platform using depth maps and wireless accelerometer*, Computer Methods and Programs in Biomedicine, 117(3), 2014, pp. 489–501. https://doi.org/10.1016/j.cmpb.2014.09.005

2. **Le2i / IMVIA Fall Detection Dataset** (University of Bourgogne / dataUBFC)
   - Record: https://search-data.ubfc.fr/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html
   - Direct zip: https://search-data.ubfc.fr/dl_data.php?file=101 (FallDataset.zip, advertised 8.95 GB)
   - DOI: 10.25666/DATAUBFC-2024-04-09
   - License: **CC BY-NC-SA**
   - Used subsets: **Home** and **Coffee room** only (have `Annotation_files`). Office and Lecture room were skipped (not home scenes; livingroom already covered).
   - Citation: Julien Dubois, Johel Miteran (2014): Fall Detection Dataset. dataUBFC. doi:10.25666/DATAUBFC-2024-04-09
   - Related paper: Charfi et al., *Optimised spatio-temporal descriptors for real-time fall detection*, JEI 2013, doi:10.1117/1.JEI.22.4.041106

### Failed / skipped (not pirated, not scraped from shady hosts)

3. **Multiple Cameras Fall Dataset** (Université de Montréal, Auvinet et al., Tech. Report 1350)
   - Official pages https://www.iro.umontreal.ca/~labimage/Dataset/ and https://www-labs.iro.umontreal.ca/~labimage/Dataset/ are behind an **Anubis bot-wall**. wget/curl cannot fetch the videos. **Skipped.** No unofficial mirrors used.

4. **UP-Fall Detection** (Universidad Panamericana)
   - Site: https://sites.google.com/up.edu.mx/har-up/
   - Complete dumps are hundreds of GB; file links are Google Drive `drive.google.com/a/up.edu.mx/...` (university-hosted, login). **Skipped** (registration / size).

5. **TST Fall Detection v2**
   - IEEE DataPort https://ieee-dataport.org/documents/tst-fall-detection-dataset-v2 — **login required**. Depth+skeleton, not RGB home video. **Skipped.**

## Scene mapping

Only three scene buckets are allowed: `livingroom`, `bedroom`, `bathroom`.

- URFD: indoor laboratory with furniture. Mapped to **livingroom**. Camera 0 = side view parallel to the floor; camera 1 = ceiling. Not a real apartment.
- Le2i Home: mapped to **livingroom** (source scene=Home; we did **not** relabel as bedroom/bathroom without a dataset scene tag).
- Le2i Coffee room: mapped to **livingroom**.
- Le2i Office / Lecture: **not included**.
- **Bathroom:** zero clips. Do not relabel living-room footage as bathroom.
- **Bedroom:** zero clips. No official bed-exit label.

## Event types and labels

| action_label | How it was assigned |
|---|---|
| `fall` | URFD `fall-*` sequences; Le2i videos whose Annotation_files start/end frames are non-zero |
| `nearfall` | URFD ADL sequences whose official `urfall-cam0-adls.csv` contains posture label `1` (lying on the ground). Event type `lie_down`. **Not invented from unlabeled walking ADLs.** |
| `normal` | Remaining URFD ADL (posture always `-1`); Le2i videos with start=end=0 (no fall) |
| `risk_behavior` | **Zero.** Public sets do not include night wandering or long sitting. |

URFD fall **type** (forward / syncope / from chair) is **not named per file** in the official release → `event_type=fall`.

## How timestamps were derived

**URFD cam0 falls and lie-down ADLs**

- `urfall-cam0-falls.csv` / `urfall-cam0-adls.csv` columns: sequence, frame, label, … where label `-1` = not lying, `0` = falling (temporary pose), `1` = lying on the ground (author documentation on the dataset page).
- `fall_start_ts` = time of the first label-`0` frame.
- `fall_end_ts` = time of the first label-`1` frame after that (onset of lying), using last-`0` then first-`1`.
- Frame index is converted with official `*-data.csv` column 2 (milliseconds since sequence start) / 1000.

**URFD cam1 falls**

- Official feature labels are **cam0 only**. Cameras are **not strictly synchronized**. `fall_start_ts` / `fall_end_ts` are **left blank** (not copied from cam0).

**Le2i Home / Coffee**

- Annotation file: beginning-of-fall frame, end-of-fall frame (dataset README), then per-frame boxes.
- Timestamp = `frame / fps` with fps from ffprobe (dataset states 25 fps). Start=end=0 → not a fall.

**Never fabricated.** If unknown, the field is JSON `null` / CSV blank.

## Bounding boxes

- URFD: **no person bbox** in the files we downloaded (feature CSV has ratios, not pixel boxes). `person_bbox` is `null`.
- Le2i: official `Annotation_files` begin with start/end frames, then per-frame rows. The dataset README says height/width/center; **the files themselves** are `frame,flag,xmin,ymin,xmax,ymax`. We store those four numbers as `xyxy` without converting. Full tracks: `annotations/frame_bbox/<stem>.json`. Event JSON `person_bbox` is the box at the fall-start frame when present; otherwise `null`.
- A few Home_02 clips probe as 320×180 (original), not 320×240.

No skeleton files were present in the obtained packages.

## Conversion (ffmpeg)

- URFD MP4s were already H.264; remuxed with `ffmpeg -c copy -an -movflags +faststart` (no re-encode, no scale).
- Le2i sources are typically AVI at 320×240 / 25 fps; transcoded with `ffmpeg -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -an` **without scaling**.
- Original resolutions are recorded in each JSON and in `manifest.csv`.

## Manifest and tools

- `manifest.csv` columns are exactly: video_path, scene, event_type, action_label, subject_id, age_group, gender, lighting, camera_angle, occlusion, resolution, fps, duration_s, fall_start_ts, fall_end_ts, annotation_file, sensor_file, note
- Every video has a row and a matching `annotations/action_events/<same_basename>.json`.
- Regenerate the manifest:

```
python3 /workspace/fall-warning-data/tools/build_manifest.py
```

- `meta/subject_info.csv` — desensitized; ids are `unknown` (authors did not publish per-clip subject ids, names, or contact data).
- `meta/environment.csv` — lighting/angle/distance when known; otherwise `unknown`.
- `meta/sensor/` — URFD accelerometer (`*_acc.csv`) and sync (`*_sync.csv`).
- `meta/source_mapping.csv` — new filename ↔ original file.

## Coverage vs product spec

Present: simulated indoor falls (side + ceiling Kinect, home/coffee-room RGB), some lie-down ADLs, short walking ADLs, IMU on URFD.

Missing (do **not** treat empty folders as “no falls in bathrooms” medically — it is a **data gap**):

- bathroom slips, showers, wet-floor
- bed-exit / out-of-bed
- night IR, low-light, night wandering
- real elderly, mobility aids, walkers, wheelchairs (not labeled)
- long-duration sitting / 2 h inactivity
- ≥720p video

## License reminder

Both obtained video sets are **CC BY-NC-SA**. This assembly is for **non-commercial academic / prototype** use. Redistribute only under the same terms and cite the original authors. Do not use commercially without permission from the dataset owners.
