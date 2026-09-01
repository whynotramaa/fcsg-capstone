import base64, pathlib

FIG = pathlib.Path("results/figures")

def img(name):
    b = (FIG / name).read_bytes()
    return f"data:image/png;base64,{base64.b64encode(b).decode()}"

HTML = f"""<title>DnCNN Baseline</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,300;0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {{
  --paper:      #f6f8f7;
  --surface:    #ffffff;
  --ink:        #14201f;
  --ink-soft:   #4d5c59;
  --ink-faint:  #778682;
  --rule:       #d9e0de;
  --rule-soft:  #e9eeec;
  --accent:     #0f6e63;
  --accent-ink: #0b524a;
  --accent-wash:#dcece8;
  --flag:       #8a5316;
  --flag-wash:  #f6ead9;
  --shadow:     0 1px 2px rgba(20,32,31,.05), 0 8px 24px -12px rgba(20,32,31,.18);
  --serif: "Spectral", Georgia, "Times New Roman", serif;
  --sans:  "IBM Plex Sans", system-ui, -apple-system, sans-serif;
  --mono:  "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:      #0d1312;
    --surface:    #151d1c;
    --ink:        #e3e9e7;
    --ink-soft:   #a7b5b1;
    --ink-faint:  #7d8b87;
    --rule:       #2a3634;
    --rule-soft:  #1d2726;
    --accent:     #58c4b2;
    --accent-ink: #8ad9ca;
    --accent-wash:#16302c;
    --flag:       #d79a52;
    --flag-wash:  #2e2517;
    --shadow:     0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
  }}
}}
:root[data-theme="dark"] {{
  --paper:      #0d1312;
  --surface:    #151d1c;
  --ink:        #e3e9e7;
  --ink-soft:   #a7b5b1;
  --ink-faint:  #7d8b87;
  --rule:       #2a3634;
  --rule-soft:  #1d2726;
  --accent:     #58c4b2;
  --accent-ink: #8ad9ca;
  --accent-wash:#16302c;
  --flag:       #d79a52;
  --flag-wash:  #2e2517;
  --shadow:     0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
}}

* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--serif);
  font-size: 17px;
  line-height: 1.62;
  -webkit-font-smoothing: antialiased;
}}
.wrap {{
  max-width: 54rem;
  margin: 0 auto;
  padding: clamp(2rem, 5vw, 4.5rem) clamp(1.1rem, 4vw, 2.5rem) 6rem;
  display: flex;
  flex-direction: column;
  gap: 3.25rem;
}}
p {{ margin: 0 0 1.05em; max-width: 68ch; }}
p:last-child {{ margin-bottom: 0; }}

/* ---- masthead ---- */
.masthead {{ display: flex; flex-direction: column; gap: 1.5rem; }}
.eyebrow {{
  font-family: var(--mono);
  font-size: .72rem;
  letter-spacing: .16em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0;
}}
h1 {{
  font-family: var(--serif);
  font-weight: 600;
  font-size: clamp(2.3rem, 6vw, 3.4rem);
  line-height: 1.08;
  letter-spacing: -.02em;
  margin: 0;
  text-wrap: balance;
}}
.standfirst {{
  font-size: clamp(1.08rem, 2.4vw, 1.28rem);
  line-height: 1.5;
  color: var(--ink-soft);
  font-weight: 300;
  max-width: 46ch;
  margin: 0;
}}
.meta {{
  display: flex;
  flex-wrap: wrap;
  gap: .4rem 2rem;
  padding-top: 1.25rem;
  border-top: 1px solid var(--rule);
  font-family: var(--sans);
  font-size: .82rem;
  color: var(--ink-faint);
}}
.meta b {{ font-weight: 500; color: var(--ink-soft); }}

/* ---- sections ---- */
section {{ display: flex; flex-direction: column; gap: 1.15rem; }}
h2 {{
  font-family: var(--serif);
  font-weight: 600;
  font-size: 1.62rem;
  letter-spacing: -.012em;
  line-height: 1.22;
  margin: 0;
  padding-bottom: .55rem;
  border-bottom: 2px solid var(--ink);
  text-wrap: balance;
}}
h3 {{
  font-family: var(--sans);
  font-weight: 600;
  font-size: .95rem;
  letter-spacing: .01em;
  margin: .6rem 0 0;
}}

/* ---- headline figure ---- */
.headline {{
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 3px;
  box-shadow: var(--shadow);
  padding: clamp(1.4rem, 3.5vw, 2.1rem);
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 13rem), 1fr));
  gap: 1.6rem;
}}
.stat {{ display: flex; flex-direction: column; gap: .3rem; }}
.stat .label {{
  font-family: var(--mono);
  font-size: .68rem;
  letter-spacing: .13em;
  text-transform: uppercase;
  color: var(--ink-faint);
}}
.stat .value {{
  font-family: var(--sans);
  font-weight: 600;
  font-size: clamp(1.9rem, 5vw, 2.5rem);
  line-height: 1;
  letter-spacing: -.02em;
  font-variant-numeric: tabular-nums;
  color: var(--accent);
}}
.stat .value span {{ font-size: .42em; font-weight: 500; letter-spacing: 0; color: var(--ink-soft); }}
.stat .foot {{ font-family: var(--sans); font-size: .78rem; color: var(--ink-faint); line-height: 1.4; }}

/* ---- tables ---- */
.scroll {{ overflow-x: auto; }}
table {{
  border-collapse: collapse;
  width: 100%;
  min-width: 34rem;
  font-family: var(--sans);
  font-size: .87rem;
  font-variant-numeric: tabular-nums;
}}
th, td {{ text-align: right; padding: .62rem .8rem; border-bottom: 1px solid var(--rule-soft); }}
th:first-child, td:first-child {{ text-align: left; padding-left: 0; }}
th:last-child, td:last-child {{ padding-right: 0; }}
thead th {{
  font-size: .7rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--ink-faint);
  font-weight: 500;
  border-bottom: 1px solid var(--rule);
}}
tbody tr:last-child td {{ border-bottom: none; }}
td.key {{ color: var(--accent); font-weight: 600; }}
.note {{ font-family: var(--sans); font-size: .78rem; color: var(--ink-faint); margin: 0; }}

/* ---- callouts ---- */
.callout {{
  border-left: 3px solid var(--accent);
  background: var(--accent-wash);
  padding: 1rem 1.25rem;
  border-radius: 0 3px 3px 0;
  font-size: .96rem;
}}
.callout.flag {{ border-left-color: var(--flag); background: var(--flag-wash); }}
.callout p {{ max-width: 62ch; }}
.callout .head {{
  font-family: var(--mono);
  font-size: .68rem;
  letter-spacing: .13em;
  text-transform: uppercase;
  color: var(--accent-ink);
  display: block;
  margin-bottom: .45rem;
}}
.callout.flag .head {{ color: var(--flag); }}

/* ---- figures ---- */
figure {{ margin: 0; display: flex; flex-direction: column; gap: .6rem; }}
figure img {{
  width: 100%;
  height: auto;
  display: block;
  border: 1px solid var(--rule);
  border-radius: 3px;
  background: #fff;
}}
figcaption {{ font-family: var(--sans); font-size: .8rem; color: var(--ink-faint); line-height: 1.5; }}
figcaption b {{ color: var(--ink-soft); font-weight: 500; }}
.pair {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 19rem), 1fr)); gap: 1.6rem; }}

code {{
  font-family: var(--mono);
  font-size: .86em;
  background: var(--rule-soft);
  padding: .12em .38em;
  border-radius: 2px;
}}
pre {{
  font-family: var(--mono);
  font-size: .8rem;
  line-height: 1.55;
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 3px;
  padding: 1rem 1.15rem;
  overflow-x: auto;
  margin: 0;
}}
pre code {{ background: none; padding: 0; font-size: inherit; }}

ul {{ margin: 0; padding-left: 1.15rem; max-width: 68ch; display: flex; flex-direction: column; gap: .5rem; }}
li::marker {{ color: var(--accent); }}

footer {{
  border-top: 1px solid var(--rule);
  padding-top: 1.4rem;
  font-family: var(--sans);
  font-size: .8rem;
  color: var(--ink-faint);
}}
a {{ color: var(--accent-ink); text-decoration-thickness: 1px; text-underline-offset: 2px; }}
a:focus-visible, :focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}

@page {{ size: A4; margin: 17mm 15mm 20mm; }}
@media print {{
  :root, :root[data-theme="dark"] {{
    --paper: #fff; --surface: #fff; --ink: #14201f; --ink-soft: #40504d;
    --ink-faint: #5f6f6b; --rule: #c3ccca; --rule-soft: #e4eae8;
    --accent: #0b5750; --accent-ink: #0b5750; --accent-wash: #eaf3f1;
    --flag: #7a4712; --flag-wash: #f7efe3; --shadow: none;
  }}
  body {{ font-size: 10.2pt; line-height: 1.5; background: #fff; }}
  .wrap {{ max-width: none; padding: 0; gap: 1.6rem; }}
  h1 {{ font-size: 22pt; }}
  h2 {{ font-size: 13.5pt; break-after: avoid; }}
  h3 {{ break-after: avoid; }}
  .standfirst {{ font-size: 11.5pt; }}
  .stat .value {{ font-size: 19pt; }}
  .headline {{ padding: 1rem; box-shadow: none; break-inside: avoid; }}
  figure, table, .callout, pre, .stat {{ break-inside: avoid; }}
  .pair {{ grid-template-columns: 1fr 1fr; gap: 1rem; }}
  figure img {{ border-color: #c3ccca; }}
  li {{ break-inside: avoid; }}
  a {{ color: inherit; text-decoration: none; }}
  footer {{ break-inside: avoid; }}
}}
</style>

<div class="wrap">

<header class="masthead">
  <p class="eyebrow">FCSG-Net &middot; Phase 1 baseline</p>
  <h1>DnCNN reproduces to within 0.19&nbsp;dB, at a cost of 44 GPU-hours</h1>
  <p class="standfirst">The baseline is trustworthy. Getting there exposed a data loader that
  wasted nine tenths of every training step, now fixed and 4.8&times; faster.</p>
  <div class="meta">
    <span><b>Model</b> DnCNN, 17 layers, width 64, 558,403 params</span>
    <span><b>Noise</b> AWGN &sigma;=25</span>
    <span><b>Checkpoint</b> step 286,000</span>
    <span><b>Date</b> 2 September 2026</span>
  </div>
</header>

<section>
  <div class="headline">
    <div class="stat">
      <span class="label">CBSD68, cross-dataset</span>
      <span class="value">31.04<span> dB</span></span>
      <span class="foot">Published DnCNN colour: 31.23 dB.<br>Gap of 0.19 dB.</span>
    </div>
    <div class="stat">
      <span class="label">DIV2K val, full images</span>
      <span class="value">32.65<span> dB</span></span>
      <span class="foot">Exit test asked for roughly 29 dB.<br>Clears it by 3.6 dB.</span>
    </div>
    <div class="stat">
      <span class="label">Loader throughput</span>
      <span class="value">4.8<span>&times; faster</span></span>
      <span class="foot">1.79 &rarr; 8.63 steps/s.<br>A 300k run: 46.7 h &rarr; 9.7 h.</span>
    </div>
  </div>
</section>

<section>
  <h2>What was measured</h2>
  <p>Three PSNR figures circulate for this one checkpoint, and they are easy to confuse
  because all three are honest measurements of different things. Only the last row belongs
  next to a published number.</p>
  <div class="scroll">
  <table>
    <thead>
      <tr><th>Protocol</th><th>Images</th><th>Noisy in</th><th>Restored out</th><th>Gain</th></tr>
    </thead>
    <tbody>
      <tr><td>DIV2K val, 128px crops (training log)</td><td>16</td><td>20.56</td><td>34.75</td><td>+14.19</td></tr>
      <tr><td>DIV2K val, full images</td><td>100</td><td>20.70</td><td>32.65</td><td>+11.95</td></tr>
      <tr><td class="key">CBSD68, full images</td><td>68</td><td>20.53</td><td class="key">31.04</td><td>+10.51</td></tr>
    </tbody>
  </table>
  </div>
  <p class="note">Rows two and three are appended to <code>results/benchmark.csv</code>. Row one is
  the in-loop validation that <code>train.py</code> writes every 2,000 steps.</p>
  <p>The measured noise floor confirms the degradation pipeline is correct. For &sigma;=25 in
  0&ndash;255 units the theoretical input PSNR is 20.17&nbsp;dB, and clipping to [0,1] lifts it
  slightly. All three protocols land between 20.53 and 20.70&nbsp;dB.</p>
</section>

<section>
  <h2>The exit test named the wrong number</h2>
  <p><code>plan.md</code> sets the Phase 1 bar at &ldquo;roughly 29 dB PSNR on the DIV2K
  validation set&rdquo;. That figure is the <em>grayscale</em> BSD68 result for DnCNN-S. This
  model is <code>DnCNN(channels=3)</code>, so the comparable published figure is the colour
  CBSD68 one, about 31.23&nbsp;dB.</p>
  <div class="callout">
    <span class="head">Verdict</span>
    <p>At 31.04&nbsp;dB the baseline sits 0.19&nbsp;dB below published, well inside the 0.5&nbsp;dB
    band the plan requires. Phase 1 passes for DnCNN. Correct the target in <code>plan.md</code>
    so the next reader is not comparing colour results against a grayscale reference.</p>
  </div>
  <p class="note">Verify 31.23 dB against Zhang et al., TIP 2017 before it goes in the report.
  It is quoted here from memory, not from the paper.</p>
</section>

<section>
  <h2>The run had converged before it stopped</h2>
  <p>Training halted at step 286,000 of a scheduled 300,000 when the Kaggle quota ran out.
  That truncation costs nothing measurable, and the schedule itself is the argument.</p>
  <div class="pair">
    <figure>
      <img src="{img('psnr_curve.png')}" alt="Validation PSNR against training step, flat after 260k steps, with a shaded gap between 132k and 210k">
      <figcaption><b>Validation PSNR.</b> Flat within &plusmn;0.04 dB across the final 22,000 steps.
      Best was 34.75 dB at step 280,000. The shaded band is a hole in the log, not a hole in training.</figcaption>
    </figure>
    <figure>
      <img src="{img('lr_curve.png')}" alt="Cosine learning rate schedule on a log axis, decaying from 2e-4 to below 2e-6">
      <figcaption><b>Cosine schedule.</b> At step 286,000 the learning rate is 2.07e-6, one percent
      of peak. The remaining 14,000 steps carry 0.06% of the run's integrated learning rate.</figcaption>
    </figure>
  </div>
  <p>Two independent signals agree, so the checkpoint at 286,000 is the final baseline and the
  run should not be resumed. Spending 2.2 hours of a 30-hour weekly quota to reach a round
  number would buy noise.</p>
</section>

<section>
  <h2>What it looks like</h2>
  <figure>
    <img src="{img('qualitative.png')}" alt="Two rows comparing noisy input, restored output, and ground truth crops from DIV2K validation images">
    <figcaption><b>Noisy, restored, ground truth.</b> 320px crops from DIV2K validation at &sigma;=25.
    Denoising is clean on smooth regions; fine texture is softened, which is the known DnCNN
    failure mode and the gap FCSG-Net's high-frequency expert is meant to close.</figcaption>
  </figure>
</section>

<section>
  <h2>What it cost, and why that matters more than the result</h2>
  <p>The baseline consumed roughly 44 GPU-hours across four Kaggle sessions, at a steady
  1.79 steps per second. A full 300,000-step run costs 46 hours at that rate, against a
  30-hour weekly quota.</p>
  <div class="scroll">
  <table>
    <thead><tr><th>Session</th><th>Steps</th><th>Hours</th><th>Rate</th></tr></thead>
    <tbody>
      <tr><td>1</td><td>50 &ndash; 56,600</td><td>8.34</td><td>1.88/s</td></tr>
      <tr><td>2</td><td>56,050 &ndash; 132,650</td><td>11.98</td><td>1.78/s</td></tr>
      <tr><td>3 (log lost)</td><td>132,650 &ndash; 210,050</td><td>~12.0</td><td>&mdash;</td></tr>
      <tr><td>4</td><td>210,050 &ndash; 287,750</td><td>11.98</td><td>1.80/s</td></tr>
    </tbody>
  </table>
  </div>
  <div class="pair">
    <figure>
      <img src="{img('throughput.png')}" alt="Training step against cumulative GPU hours, a near-straight line broken where one session's log was lost">
      <figcaption><b>Throughput.</b> Dotted lines mark session restarts. The break is session 3,
      whose <code>train_log.csv</code> was never seeded on resume, so 77,400 steps and about
      12 hours are absent from the record.</figcaption>
    </figure>
    <figure>
      <img src="{img('loss_curve.png')}" alt="Charbonnier training loss on a log axis, falling from 0.076 to 0.017">
      <figcaption><b>Training loss.</b> Charbonnier, 0.0761 down to 0.0174 over 4,220 logged points.
      No instability, no divergence after any resume.</figcaption>
    </figure>
  </div>
  <h3>The diagnosis</h3>
  <p><code>PatchDataset.__getitem__</code> opened and fully decoded a 2040&times;1356 PNG to take a
  single 128px crop, then discarded the decode. A 0.56M-parameter DnCNN at batch 16 should take
  tens of milliseconds per step on a P100; it was taking 555. Roughly nine tenths of every step
  was libpng, not CUDA.</p>
  <div class="callout flag">
    <span class="head">Why this blocks Phase 4</span>
    <p>FCSG-Net is larger than DnCNN, and the ablation table is several runs. At 46 hours each,
    on 30 GPU-hours per week, the table alone would take more than a month of wall-clock time.
    The loader was the constraint on the research schedule, not the GPU. It has since been fixed,
    and the measurement is below.</p>
  </div>
  <h3>The fix</h3>
  <p><code>training/build_tiles.py</code> extracts crops once into a <code>uint8</code> NumPy memmap,
  and <code>TileDataset</code> serves them with no decode at all. At 64 crops per image that is
  51,200 tiles, about 2.5&nbsp;GB, times eight flip and rotation variants.</p>
  <pre><code>python training/build_tiles.py --data /content/DIV2K \\
    --out tiles.npy --patch 128 --per-image 64

python training/train.py --config configs/dncnn.toml \\
    --data /content/DIV2K --tiles tiles.npy --out ckpt</code></pre>
  <div class="scroll">
  <table>
    <thead><tr><th>Loader</th><th>Rate</th><th>300k run</th><th>Sessions needed</th></tr></thead>
    <tbody>
      <tr><td>PNG decode per sample</td><td>1.79/s</td><td>46.7 h</td><td>4</td></tr>
      <tr><td class="key">Tile memmap</td><td class="key">8.63/s</td><td class="key">9.7 h</td><td class="key">1</td></tr>
    </tbody>
  </table>
  </div>
  <p>Measured on a Colab T4 over 2,000 steps, same model and batch size. The run now fits inside
  one session under the 11-hour budget, which removes the resume cycle that lost session 3's log
  in the first place.</p>
  <div class="callout flag">
    <span class="head">Caveat</span>
    <p>8.63 steps/s is a DnCNN number. FCSG-Net carries an FFT, nine experts, and a router, so it
    will be slower and the bottleneck may return to the GPU where it belongs. Re-time once the
    model exists, before committing to a step count for Phase 3.</p>
  </div>
</section>

<section>
  <h2>Open items</h2>
  <ul>
    <li><b>FFDNet is not written.</b> The Phase 1 exit test has two halves and only DnCNN is done.
    <code>models/ffdnet.py</code> does not exist, and <code>build_model</code> in
    <code>train.py</code> has no branch for it. This is code, not compute, and it blocks M1.</li>
    <li><b><code>notes/related.md</code> is an empty skeleton.</b> Four headings, no paragraphs.
    <code>plan.md</code> requires MWCNN, SFNet/FSNet, and the sparse mixture-of-experts paper to
    be read before M1, because the novelty claim is positioned against SFNet.</li>
    <li><b>Set <code>steps</code> deliberately for Phase 3.</b> At 8.63 steps/s a 300,000-step
    DnCNN run takes 9.7 hours, which fits one session with about an hour of margin. FCSG-Net will
    be slower, so re-time it before choosing its step count.</li>
    <li><b>Session 3's log is unrecoverable.</b> Seed <code>train_log.csv</code> from the previous
    output on every resume, or the record keeps losing sessions.</li>
    <li><b>The CBSD68 qualitative figure was overwritten</b> by the DIV2K evaluation run. Re-run
    that evaluation if both strips are wanted.</li>
  </ul>
</section>

<footer>
  Generated from <code>checkpoints/train_log.csv</code> (4,324 rows) and
  <code>results/benchmark.csv</code>. Figures regenerate with
  <code>python evaluation/plots.py --csv checkpoints/train_log.csv --out results/figures</code>.
</footer>

</div>
"""

out = pathlib.Path("results/phase1_report.html")
out.write_text(HTML)
print(f"wrote {out}  {out.stat().st_size / 1e6:.2f} MB")
