# From SORT to our scene-cut-aware tracker

Project-specific learning guide, checked against the code and results on 24 September 2026.

This document answers five questions: how SORT works, why it fails, what OC-SORT changes, what Deep OC-SORT adds, and exactly what **our** extension does. It separates published algorithms from our implementation and measured results from unverified tracker decisions. The original sources are linked where each method is introduced.

## First, the mental model

Multi-object tracking is not one model recognizing a person forever. It is a sequence of decisions:

```text
video frame
  -> detector: "here are person boxes and confidence scores"
  -> tracker: "which box continues which recent trajectory?"
  -> optional appearance model: "which crops look alike?"
  -> our cut-aware memory: "after a cut, can a new local track reuse an old global ID?"
```

The detector decides **where** a person is. A tracker decides **which detection belongs to which track**. A ReID model supplies a **similarity cue**, not an identity certificate. A scene-cut detector decides when the old camera-coordinate motion model should no longer be trusted. We process people only: no ball, jersey OCR, team classification, or training.

Terms used throughout:

| Term | Meaning |
|---|---|
| Detection/observation | A detector-produced bounding box for the current frame, usually `(x1, y1, x2, y2, confidence, class)`. |
| Track/tracklet | A time-ordered chain of detections believed to be the same object. |
| Local ID | The ID assigned by one running tracker instance. It can disappear or restart after a reset. |
| Global ID | Our persistent display/evaluation ID, managed separately from the tracker. |
| Prediction | Where a motion model expects a tracked box to be in the next frame. It is **not** a new detection. |
| Association | One-to-one matching of current detections to existing tracks. |
| ID switch | Against identity-labeled ground truth, a person who was matched to one predicted ID is later matched to another. It is not simply the count of IDs created. |
| Occlusion | A person is partly/fully hidden or the detector misses them for some frames. |
| Hard cut | A discrete editing jump from one shot to another; pixel coordinates and motion can change instantly. |

### The three mathematical objects that matter

Suppose a tracker predicts boxes `T1, T2` and YOLO detects boxes `D1, D2`. Their overlaps form an IoU matrix:

```text
             D1     D2
T1          0.70   0.05
T2          0.10   0.65
```

`IoU = area(intersection) / area(union)`, from 0 (no overlap) to 1 (same box). The Hungarian algorithm finds a **globally consistent one-to-one assignment** for the whole matrix, not an independent greedy choice for each track. Here it chooses `T1-D1` and `T2-D2`. A low-IoU pair is rejected even if assignment selected it. This is an optimization problem, not neural-network training.

A Kalman filter maintains a state estimate and uncertainty. In original SORT, the state is approximately

`x = [u, v, s, r, u_dot, v_dot, s_dot]^T`,

where `(u,v)` is box center, `s` is area, `r` is aspect ratio, and the dotted terms are velocities. The detector gives a measurement `z = [u,v,s,r]^T`. Original SORT assumes approximately constant velocity and constant aspect ratio. With state-transition matrix `F`, observation matrix `H`, covariance `P`, process noise `Q`, and measurement noise `R`:

```text
predict: x_prior = F x_previous
         P_prior = F P_previous F^T + Q

correct: K = P_prior H^T (H P_prior H^T + R)^(-1)
         x_new = x_prior + K (z - H x_prior)
         P_new = (I - K H) P_prior
```

These are the “matrix multiplications” behind SORT/OC-SORT. `F` moves the estimate forward; `H` maps the hidden state to the measurable box; `K` determines how much to trust the new detector box versus the prediction. If no box matches, there is no real measurement correction and uncertainty grows. In plain English: *predict where the player should be, then correct that guess using the observed player box*. OC-SORT retains this basic Kalman machinery but changes how observations are used around association and gaps. [SORT paper, Sections 3.2–3.3](https://arxiv.org/pdf/1602.00763); [OC-SORT paper, Sections 3–4](https://arxiv.org/html/2203.14360).

## 1. How original SORT works

SORT means **Simple Online and Realtime Tracking**. “Online” means it uses current and earlier frames, not future frames. The 2016 paper paired its tracker with **Faster R-CNN person detections**, not YOLO. The tracker itself is detector-agnostic; feeding it YOLO boxes changes the detector, not the SORT association principle. [Original SORT paper, Section 3](https://arxiv.org/pdf/1602.00763).

For each frame:

1. A detector produces person boxes. SORT does not discover pixels or draw a box around a missed person by itself.
2. Every live track's Kalman filter predicts a box in the current frame.
3. It computes pairwise IoU between predicted boxes and detections.
4. Hungarian assignment selects one-to-one pairs; an IoU gate rejects implausible matches.
5. Matched tracks receive a real measurement update and keep their IDs.
6. Unmatched detections start tentative new tracks with new IDs.
7. Unmatched tracks age; a track is eventually deleted if detections do not return. Original SORT used a one-frame lost-track lifetime in its experiments and gave a reappearing person a new ID. Confirmation/age details vary by implementation; our BoxMOT OC-SORT uses `max_age: 30` and `min_hits: 3`, which are **not** the original paper's exact setup.

Concrete example: a player detected at center `x=100` and then `x=104` is estimated to move roughly four pixels per frame. SORT predicts near `x=108` next. If the next player box overlaps there, it keeps the ID. If a cut moves that same player's image to `x=700`, the predicted box and new box probably have zero IoU, so SORT starts another ID. It has no appearance memory that says “this looks like the earlier person.”

SORT's strength is its simple, fast, transparent tracking-by-detection loop. Its paper's reported **260 Hz refers to its tracking component**, not a guaranteed end-to-end detector-plus-tracker video rate on our CPU. The paper explicitly showed that changing detection quality could materially change tracking results. [SORT abstract and experiments](https://arxiv.org/abs/1602.00763).

## 2. What goes wrong with SORT, and why

The important point is to identify the failing **layer**, not just say “IDs are bad.”

| Failure | Mechanism | What you see in our videos |
|---|---|---|
| Detector miss or poor box | No reliable observation exists to correct the track; a tracker cannot create a correct current person box from nothing. | Tiny/blurred football players or dark film frames lack boxes or have partial boxes. |
| Nonlinear movement | Constant-velocity extrapolation points the box the wrong way after a turn, acceleration, or abrupt pose/camera change. | Fast runs and crossovers create low IoU or wrong associations. |
| Prolonged occlusion | Repeated predictions without real corrections accumulate position/velocity error; the old track may expire. | A player passes behind another and returns with a new ID. |
| Ambiguous overlap | Two similar-sized boxes cross; geometry alone may assign the wrong detection. | Two nearby players swap displayed IDs. |
| Hard scene cut | Camera coordinates and object motion are discontinuous. The old prediction has no physical meaning in the new shot. | Same actor/player can restart with a new ID at a cut. |
| Similar appearances | SORT has no appearance model at all. | It cannot tell two nearby football players apart by their crops. |

The [OC-SORT paper's analysis](https://arxiv.org/html/2203.14360) sharpens the middle three issues: (1) velocity estimated from nearby noisy boxes is sensitive to state noise, (2) errors amplify while observations are missing, and (3) an *estimation-centric* tracker can trust its own drifted predictions too much. Keeping a track alive with `max_age` does **not** guarantee a visible output box or the correct ID when a person returns; the new detection still has to associate successfully.

### Why “just increase `max_age`” is not a fix

After 20 missing frames, the Kalman track may still exist internally. But if its predicted box is far from the real person, the IoU gate blocks the match. Larger `max_age` extends how long a hypothesis survives; it does not make a bad prediction accurate. Likewise, a larger OSNet weight inside an IoU-gated association cannot force a zero-overlap match.

## 3. OC-SORT: what it fixes and how

OC-SORT means **Observation-Centric SORT**. The change in philosophy is: detector observations are the reliable anchors; motion estimates are useful but can drift, especially through gaps. It is still an online tracker and the original OC-SORT is **motion/geometry-based, not person-ReID-based**. The paper reports three mechanisms: OCM, OCR, and ORU. [OC-SORT paper, Sections 3–4](https://arxiv.org/html/2203.14360).

### OCM — Observation-Centric Momentum

SORT uses predicted-box overlap to decide who is who. OC-SORT additionally asks whether a proposed match continues the **observed direction of travel**. It estimates a direction from actual earlier observations separated by a short interval `delta_t`, rather than trusting only the latest noisy Kalman velocity. Direction agreement adds a weighted term to the IoU association matrix before Hungarian assignment.

Example: two players' boxes overlap similarly after crossing. If player A's observed trajectory was moving right and one candidate detection continues right, OCM favors that association. It is still a *soft cue*; it does not recognize a face or jersey number. In our config, `delta_t: 3` and `inertia: 0.20` set the look-back and direction term for the BoxMOT implementation. [OCM section](https://arxiv.org/html/2203.14360).

### OCR — Observation-Centric Recovery

After the normal association pass, OCR makes another attempt for unmatched tracks and detections, comparing against the track's **last actual observed box** rather than only its possibly drifted prediction. This helps if a person paused or briefly disappeared while the Kalman prediction moved away. The match still needs plausible spatial overlap. [OCR description](https://arxiv.org/html/2203.14360).

### ORU — Observation-Centric Re-Update

If a previously unmatched track **is successfully re-associated** with a real detection, ORU revisits the missed interval. It uses the last real detection before the gap and the newly recovered real detection as endpoints, interpolates *virtual* intermediate observations, and replays Kalman predict/update steps. This repairs a motion state that would otherwise remain contaminated by its blind predictions.

Important ordering: OCR/normal association first finds a plausible return; **then** ORU repairs the filter. ORU is not a magic cross-camera identity search. It cannot repair a track that never re-associated. The early Deep OC-SORT paper calls this component **OOS** (Observation-Centric Online Smoothing); the OC-SORT authors later renamed OOS to **ORU**. [ORU section](https://arxiv.org/html/2203.14360); [official OC-SORT repository rename note](https://github.com/noahcao/OC_SORT).

### Why OC-SORT was sensible for football

Players move, accelerate, overlap, and become temporarily occluded. OC-SORT directly targets the weakness of a naive constant-velocity tracker in those situations while remaining relatively simple and detector-flexible. It is a good *baseline to explain and test*, not a claim that it is universally best. Football also exposes its limits: distant players are hard to detect, same-team uniforms are alike, camera pans distort apparent motion, and broadcast edits create true discontinuities. OC-SORT does not solve any of those simply by being observation-centric.

## 4. What is special about Deep OC-SORT?

Deep OC-SORT starts with OC-SORT and adds **appearance-aware association**. Its paper identifies three new modules: Camera Motion Compensation (CMC), Dynamic Appearance (DA), and Adaptive Weighting (AW). It is **not DeepSORT** with a renamed detector, and it is **not** the same thing as “OC-SORT + an OSNet threshold.” [Deep OC-SORT paper](https://arxiv.org/pdf/2302.11813).

### CMC — Camera Motion Compensation

If the camera pans right, many people's image coordinates shift left even if they did not move that way on the field. CMC estimates a global frame-to-frame image transform and adjusts track positions/motion accordingly before association. This addresses camera-induced apparent motion *within a continuous shot*. A hard edit is different: there may be no meaningful transform between the two shots. Our BoxMOT Deep OC-SORT config has `cmc_off: false`, so **CMC is enabled**; its installed implementation uses a sparse-optical-flow CMC method. [Deep OC-SORT, Section 3.2](https://arxiv.org/pdf/2302.11813).

### DA — Dynamic Appearance

For each tracked person, a ReID network converts the detected crop to a feature vector (an embedding). Deep OC-SORT keeps a smoothed appearance description, roughly

`track_embedding_new = alpha * old_embedding + (1 - alpha) * current_detection_embedding`.

When detector confidence is low, a crop may be blurred, occluded, or badly localized. DA increases `alpha`, so that dubious new feature changes the track appearance less. With a confident crop it accepts more new information. Our base `alpha_fixed_emb` is `0.95`. This protects the short-term appearance model; it does **not** create an eternal identity memory. [Deep OC-SORT, Section 3.3](https://arxiv.org/pdf/2302.11813).

### AW — Adaptive Weighting

Deep OC-SORT combines geometric overlap, OC-SORT's direction cue, and appearance similarity. With normalized vectors, cosine similarity is just their dot product: `similarity = e_detection · e_track`. The proposed AW idea uses appearance more when one candidate is clearly more similar than alternatives, and less when everyone looks alike. Conceptually, each possible track/detection pair receives an affinity such as

`affinity = IoU + direction_term + weighted_appearance_similarity`.

Hungarian assignment operates on the resulting matrix. This is **association scoring**, not retraining YOLO or matrix multiplication between whole images. In our installed BoxMOT version, appearance contribution is zeroed for non-overlapping candidate pairs, and the final match is rejected if IoU is below `0.30`. Therefore, even a strong OSNet similarity cannot reconnect a zero-overlap track in the normal Deep OC-SORT matching stage. [Deep OC-SORT, Section 3.4](https://arxiv.org/pdf/2302.11813); see installed `boxmot/utils/association.py` and our [diagnosis](TRACKING_DIAGNOSIS.md).

### What OSNet actually contributes

OSNet is a pretrained **person re-identification feature extractor**. Its omni-scale blocks process a person crop through multiple receptive-field scales and fuse those features with input-dependent gating. Small details (shoes, clothing patches) and broader appearance (whole outfit) can both affect its feature vector. We use `osnet_x0_25_msmt17.pt` on CPU, without training it on our clips. The *original Deep OC-SORT paper* used YOLOX and an SBS50 fast-reid model, **not our YOLO11/OSNet combination**. [OSNet paper](https://arxiv.org/html/1905.00953); [Deep OC-SORT implementation section](https://arxiv.org/pdf/2302.11813).

OSNet does not emit “this is player 8” as an absolute fact. It emits numbers that can be compared. The same person can look different after a viewpoint/lighting/scale change; two different people in matching uniforms can look very similar. A cosine value of `0.75` is a threshold chosen by our system, **not** a probability of a correct ID and **not** an OSNet guarantee. An in-domain validation set is needed to calibrate it.

### Four project columns: exact interpretation

| Column | Shared detector | Association and state | Cut behavior |
|---|---|---|---|
| `ocsort` | YOLO11n person detections | OC-SORT, geometry/motion; no OSNet | No reset or long-term ReID memory. |
| `deep_ocsort` | The **same cached** YOLO11n detections | BoxMOT Deep OC-SORT + OSNet, appearance weight `0.50`, AW and CMC enabled | No cut-aware reset or long-term memory. |
| `deep_ocsort_osnet` | The **same cached** YOLO11n detections | Still BoxMOT **Deep OC-SORT**, not a separate pure “OC-SORT + OSNet” algorithm; weight `1.0`, AW disabled, DA and CMC retained | Still IoU-gated; no cut-aware reset/memory. |
| `deep_ocsort_scenecut` | The **same cached** YOLO11n detections | Standard Deep OC-SORT within shots, plus our external global-ID memory | On accepted hard cut, reset short-term tracker/CMC state; attempt conservative global-ID recovery. |

The third column's name is presentation shorthand. It is **not DeepSORT**. Nor does `w_association_emb: 1.0` mean appearance is the only criterion: the installed matcher still uses spatial gates. The original SORT, OC-SORT, and Deep OC-SORT papers are distinct; our configuration is a separate engineering experiment.

## 5. Our improvement: cut-aware, selective identity recovery

### Problem statement

Within a shot, old and new coordinates are related well enough for motion/IoU and CMC to be useful. Across a hard cut, the previous coordinate system is invalid. Continuing the old Kalman prediction can create false matches or lose the same person's ID. Blindly resetting everything guarantees the old ID is lost. So we split identity state into two lifetimes:

```text
short-term state (discard at cut)        long-term state (preserve at cut)
Kalman estimates and covariance          global ID -> recent OSNet embeddings
active OC-SORT/Deep OC-SORT tracks         normalized gallery prototype
within-shot local IDs                    next global ID counter
camera-motion state
```

This separation is the central engineering improvement. It is an **extension built for this project**, not a published claim that we invented ReID or solved all cross-shot tracking.

### What happens on every frame

1. Decode the frame. The five-video runner uses a **single YOLO11n detection cache per video**, shared by all four columns; it also caches OSNet embeddings for comparable/repeatable Deep OC-SORT runs. YOLO proposals are retained from confidence `0.05`.
2. For the cut-aware column, PySceneDetect proposes a visual change using content threshold `27.0`. The current guard accepts a hard cut only if the candidate also has HSV histogram distance at least `0.40`, normalized pixel difference at least `0.12`, and sufficient spacing from the last *accepted* cut (`15` frames). These are heuristics, not an oracle.
3. **No accepted cut:** Deep OC-SORT updates normally. We update each global ID's appearance gallery with an embedding from its matched person, but a new local track does **not** search old identities outside the cut-recovery window.
4. **Accepted cut:** Clear active Deep OC-SORT/Kalman and CMC motion state; start a fresh local tracker generation. Preserve the global-ID gallery. Record the cut and open a `15`-frame recovery window.
5. Detect/track people in the new shot. For each newly emitted local track, compare its normalized OSNet embedding with candidate global-ID prototypes using cosine similarity. A prototype is the normalized average of up to `20` recent embeddings for that ID.
6. Run Hungarian assignment over the similarity matrix for a one-to-one proposal. Reuse a prior global ID **only if** the assigned similarity is at least `0.75`; otherwise allocate a new global ID. A recovered old ID can be claimed only once in that recovery window. Keep the local-to-global mapping for subsequent frames in the shot.
7. Save annotated video, MOT-format tracks, CSV tracks, cut/identity event logs, plots, and summary diagnostics. The same logic is generic to person videos; football is one evaluation setting.

The sequence above is implemented in [cut detection](../src/scenecut_tracking/scene_cut.py), [Deep OC-SORT runner](../src/scenecut_tracking/deep_runner.py), [tracker reset](../src/scenecut_tracking/deep_ocsort_tracker.py), and [identity memory](../src/scenecut_tracking/identity_memory.py). There is also a lighter OC-SORT + cut-aware path in [runner.py](../src/scenecut_tracking/runner.py), but the **four-column final experiment** uses the Deep OC-SORT cut-aware path.

### A concrete cross-cut example

Before a cut, local track `7` maps to global person `G12`; memory holds `G12`'s recent normalized OSNet vectors. At the cut, local tracker state is cleared; `G12` remains in memory. In the new shot, the same person may appear as local track `1` at a completely different pixel position. If its OSNet similarity to `G12` is `0.81` and this is the one-to-one selected match, the display uses global ID `G12`. If the best accepted assignment is `0.68`, the system creates a new global ID instead. If two players both resemble `G12`, at most one can claim it; nevertheless, the chosen one **can still be wrong**. No claim of correct recovery can be made without ground truth.

The memory currently considers *all historical identities with a valid prototype* at a cut, not just identities in the immediately previous shot. This helps when a person returns after several shots, but increases the chance of a stale lookalike match. Accepted matches also update the reused ID's gallery; a false recovery can contaminate its future prototype. These are important research limitations, not hidden implementation details.

### What our improvement cannot do

- It is **selective**: cross-cut memory only searches after an accepted cut. An ordinary same-shot disappearance can still cause a new global ID.
- A missed cut means no reset/recovery; a false cut causes unnecessary reset and risky matching. The histogram/pixel guard reduces false triggers but is not perfect.
- A poor YOLO box produces a poor crop and potentially a misleading OSNet vector. Tracking does not fix detector geometry.
- Similar uniforms, low resolution, lighting changes, extreme pose differences, and heavy occlusion can defeat pretrained person ReID.
- A high embedding score can be an impostor, and a low score can be the same person. `0.75` needs labeled in-domain validation.
- CMC handles within-shot camera motion; it cannot meaningfully warp arbitrary edited shots together.
- “Accepted ReID decisions” in logs means *the algorithm reused an ID*, not that a human or ground truth confirmed it.

### Why the boxes/IDs were imperfect in this project

The [diagnosis](TRACKING_DIAGNOSIS.md) traced a combination of low-confidence person proposals making many short tracks, IoU-gated appearance matches, false scene cuts, and appearance ambiguity. The current [CPU experiment config](../configs/ultimate_cpu.yaml) keeps YOLO at `0.05` so candidate detections are not lost early, but starts OC-SORT tracks at `0.25` and uses lower-confidence `0.10–0.25` detections for **existing OC-SORT tracks** via BYTE association. The Deep OC-SORT wrapper also starts tracks at `0.25`; do not assume its low-confidence handling is identical to the separate OC-SORT `use_byte` path. `min_hits: 3` can delay a visible label; `max_age: 30` does not guarantee ID continuity.

We visually improved drawing, but the rendered track boxes mostly follow their assigned YOLO geometry. A sharper outline can make a rectangle readable; it cannot make a partial/missed detector box correct. This is why the real path to better football boxes may require a better domain detector or labeled detector analysis, not another tracking parameter.

## Evaluation: how to say what was actually proven

The project calculates HOTA, DetA, AssA, IDF1, MOTA, FP, FN, true ID switches, and fragmentation with TrackEval when MOT-format identity ground truth exists. See [evaluation.py](../src/scenecut_tracking/evaluation.py) and the [TrackEval project](https://github.com/JonathonLuiten/TrackEval).

| Measure | What it tests | Direction / caution |
|---|---|---|
| DetA | Detection quality component of HOTA: finding/localizing people. | Higher is better; a tracker cannot compensate for persistent misses. |
| AssA | Association quality component of HOTA: keeping correctly detected people on consistent tracks. | Higher is better; useful for identity reasoning. |
| HOTA | Balanced summary over localization/detection and association across multiple IoU thresholds. | Higher is better; compare on the **same dataset, detector, protocol**. |
| IDF1 | Identity-aware precision/recall balance over the sequence. | Higher is better; requires identity GT. |
| MOTA | Primarily penalizes false positives, false negatives, and ID switches. | Higher is better, but can hide association problems when detection dominates. |
| IDs | Ground-truth-defined changes in the predicted ID matched to a person. | Lower is better; **not** equal to “unique IDs created.” |
| FP / FN | Extra predicted boxes / missed GT boxes. | Lower is better. |
| Processing FPS | Frames processed divided by measured runtime for a specified pipeline segment. | Higher is faster, not more accurate; cached-tracker FPS is not full live YOLO+ReID FPS. |
| Cross-cut recovery rate | Fraction of GT people present around annotated cuts whose predicted global ID is preserved. | Higher is better, but requires cut timing **and** person-ID GT on both sides. |

Our one annotated **SportsMOT training sequence** has 875 frames and was used for exploratory calibration. With the same cached detector/appearance inputs, old -> tuned results were: OC-SORT HOTA `33.66 -> 43.66`, IDF1 `29.22 -> 42.83`, IDs `421 -> 262`; Deep OC-SORT HOTA `38.74 -> 50.52`, IDF1 `33.50 -> 50.39`, IDs `381 -> 169`. These are **measured for that one training sequence**, not a held-out benchmark and not proof of generalization. The separate appearance-weight-1.0 column and full numbers are in [TRACKING_DIAGNOSIS.md](TRACKING_DIAGNOSIS.md).

The five custom clips (`football-1`, `football-2`, `football-3`, `MI6`, `Joker`) have no person identity ground truth. We can report frames processed, FPS, number of accepted cuts, number of ID assignments, and how many ReID matches passed our threshold. We **cannot** honestly report true ID switches, IDF1, or verified correct cross-cut recoveries for those videos. After adding the cut guard, accepted cuts across those clips fell from `55` to `28`, and accepted ReID decisions from `221` to `42`; **42 is not “42 correct recoveries.”** Check manually annotated cut/person pairs before making an accuracy claim.

## Source-code map and reproduction

| File | Why it exists |
|---|---|
| [ultimate_cpu.yaml](../configs/ultimate_cpu.yaml) | Canonical CPU-only settings, input paths, model weights, thresholds, output location. |
| [detector.py](../src/scenecut_tracking/detector.py), [detection_cache.py](../src/scenecut_tracking/detection_cache.py) | Run YOLO11n person detection once per input, save identical detections for every method. |
| [ocsort_tracker.py](../src/scenecut_tracking/ocsort_tracker.py) | BoxMOT OC-SORT adapter and motion reset. |
| [deep_ocsort_tracker.py](../src/scenecut_tracking/deep_ocsort_tracker.py) | BoxMOT Deep OC-SORT/OSNet adapter, local track embeddings, cut reset. |
| [appearance_cache.py](../src/scenecut_tracking/appearance_cache.py), [reid.py](../src/scenecut_tracking/reid.py) | Extract and reuse OSNet crop embeddings. |
| [scene_cut.py](../src/scenecut_tracking/scene_cut.py) | PySceneDetect candidate plus hard-cut visual guards, with histogram fallback. |
| [identity_memory.py](../src/scenecut_tracking/identity_memory.py) | Global ID gallery, recovery window, one-to-one thresholded matching. |
| [runner.py](../src/scenecut_tracking/runner.py), [deep_runner.py](../src/scenecut_tracking/deep_runner.py) | Frame loops, local/global ID mapping, annotated outputs and event logs. |
| [evaluation.py](../src/scenecut_tracking/evaluation.py) | Ground-truth-based tracking/detection evaluation and cross-cut recovery hook. |
| [run_all_experiments.py](../scripts/run_all_experiments.py) | Resumable five-video, four-method execution and shared-cache orchestration. |
| [Ultimate notebook](../notebooks/Ultimate_YOLO11_Tracking_Comparison.ipynb) | Presentation figures, tables, result navigation, limitations. |
| [TRACKING_DIAGNOSIS.md](TRACKING_DIAGNOSIS.md) | Root-cause evidence, measured calibration, unannotated-clip diagnostics. |

From the project root in PowerShell, rerun and validate with:

```powershell
.\.venv\Scripts\python.exe scripts\run_all_experiments.py
.\.venv\Scripts\python.exe scripts\validate_outputs.py
.\.venv\Scripts\python.exe scripts\benchmark_sportsmot.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

The runner reads `configs/ultimate_cpu.yaml`, reuses current caches/results, and rebuilds stale parts when relevant settings change. Outputs are under `outputs/experiments`; the browser gallery is `outputs/reports/video_gallery.html`. The benchmark command evaluates the annotated SportsMOT sequence, **not** the unannotated custom videos. Do not add `--force` unless you intentionally want to repeat costly CPU inference. [README](../README.md) gives the current environment and output instructions.

## Oral-defense version: what to tell your examiner

> “SORT tracks detector boxes using a constant-velocity Kalman filter, IoU matching, and Hungarian assignment. Its weakness is that predictions drift through misses and do not recognize appearance or scene cuts. OC-SORT makes the tracker observation-centric: it adds observed-motion direction to matching, retries using last real boxes, and repairs Kalman state after a track is reacquired. Deep OC-SORT adds appearance embeddings, confidence-aware embedding updates, adaptive appearance weighting, and camera-motion compensation. We used YOLO11n detections and pretrained OSNet with a BoxMOT Deep OC-SORT implementation; we did not train either model. Our extension detects hard cuts, clears invalid short-term motion state, but keeps a long-term embedding gallery and restores old global IDs only when one-to-one similarity clears a conservative threshold. This helps a specific failure mode—shot changes—but it cannot guarantee identity without labeled evaluation, especially with similar football kits.”

Questions you should be able to answer without reading the code:

1. **What is multiplied?** Kalman matrices predict/correct box state; a detection-track IoU matrix (plus direction/appearance scores in later methods) feeds Hungarian assignment; normalized OSNet vectors are dotted to calculate cosine similarity.
2. **Why reset at a cut?** The camera coordinate/motion relationship across an edit is invalid; preserving old Kalman predictions can force bad matches. Global appearance identity can still be useful, so it is preserved separately.
3. **Is Deep OC-SORT the same as DeepSORT?** No. Deep OC-SORT builds on OC-SORT's observation-centric motion method and adds DA/AW/CMC. Our repository does not run a separate DeepSORT method.
4. **Does a strong OSNet score guarantee a correct ID?** No. Similar uniforms and domain shift create false matches; a threshold is a heuristic until evaluated against labeled pairs.
5. **Did you prove cross-cut improvement?** We measured tracker gains on one annotated football training sequence and logged cut/recovery behavior on five custom clips. We **have not** measured true cross-cut correctness on those custom clips because they lack identity labels.

## Primary references

- Bewley et al., [SORT: Simple Online and Realtime Tracking](https://arxiv.org/pdf/1602.00763). Read Sections 3.2–3.4 for state, IoU/Hungarian association, and track lifecycle.
- Cao et al., [Observation-Centric SORT](https://arxiv.org/html/2203.14360). Read Sections 3–4 for SORT failure analysis and ORU/OCM; OCR is described at the end of Section 4.
- Maggiolino et al., [Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification](https://arxiv.org/pdf/2302.11813). Read Section 3 for CMC, DA, and AW, and Section 4 for the original YOLOX/SBS50 setup.
- Zhou et al., [Omni-Scale Feature Learning for Person Re-Identification](https://arxiv.org/html/1905.00953). Read Section 3 for OSNet's multi-scale residual block and unified aggregation gate.
- Luiten et al., [TrackEval metric implementations](https://github.com/JonathonLuiten/TrackEval). Use the evaluation implementation/protocol when interpreting HOTA, CLEAR, and ID metrics.
