# Related work

This review positions FCSG-Net against frequency selection and expert routing.
Parameter counts refer to the named variant, not an entire family of models.
Published scores are background evidence, not results from this repository.

## Foundations

Zhang et al.'s DnCNN learns the noise residual and subtracts it from the input.
Batch normalization stabilizes training of the convolutional stack. This
repository's RGB variant has 17 layers, width 64, and 558,403 parameters. It
provides a baseline for the data and optimization code, but has no explicit
noise map or frequency-conditioned router. The original paper also studies
blind denoising and multiple restoration tasks; DnCNN is not inherently limited
to one fixed noise level. [Paper](https://arxiv.org/abs/1608.03981).

FFDNet conditions restoration on a noise-level map and rearranges pixels into
four subimages before convolution. Its colour model uses 12 layers and 96
channels. The author's KAIR inference variant has 852,108 parameters, with batch
normalization absorbed into convolutions. Table V reports 31.21 dB on CBSD68 at
sigma 25. That protocol adds unclipped Gaussian noise. A clipped-noise model is
a separate variant, so the existing clipped DnCNN scores cannot establish
agreement with this published setting. FFDNet supplies the second baseline,
but its noise map does not describe blur or JPEG compression.
[Paper](https://arxiv.org/pdf/1710.04026),
[author implementation and protocol](https://github.com/cszn/KAIR/blob/master/main_test_ffdnet.py).

## Frequency-domain restoration

Liu et al.'s MWCNN replaces pooling and upsampling with discrete wavelet and
inverse wavelet transforms in a modified U-Net. These fixed transforms add no
learned parameters; the convolutional backbone supplies the model capacity.
The paper specifies the network in Figure 3 rather than reporting one total
parameter count. MWCNN processes subbands jointly, preserving their
dependencies. It explicitly contrasts this with separate processing of each
subband. Its ablations therefore motivate cross-band fusion in FCSG-Net.
MWCNN already establishes frequency decomposition for restoration, but does
not use the proposed spatial top-k mixture of different convolution kernels.
[Paper, sections 3.2 and 4.5](https://arxiv.org/pdf/1805.07071).

Chi et al.'s Fast Fourier Convolution combines local convolution, a semi-global
Fourier branch, and a global Fourier branch. Updates in Fourier space give a
large spatial receptive field. FFC is an operator rather than one restoration
model, so its parameter cost depends on the backbone and the fraction of
channels assigned to the spectral branch. Table 4 reports 26.7M parameters for
FFC-ResNet-50, a classifier. This is not a denoising baseline result. FCSG-Net
instead inverse-transforms masked bands before applying spatial experts.
[Paper](https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf).

Cui et al.'s SFNet is the closest comparison for the frequency-selection claim.
Its MDSF module uses content-dependent local decomposition and channel attention
to select frequency information. Its MCSF module uses global and window pooling
with learned modulation to broaden the receptive field at lower cost than
another convolutional branch. Both modules sit in a U-shaped restoration
network. Parameter cost includes that backbone and varies with the task variant;
the author's README does not report a count, so no unverified count is assigned
here. SFNet already supports local, adaptive frequency selection. The proposed
distinction is explicit top-k routing across shared 7x7, 5x5, and 3x3 experts,
with routing measured against known degradation parameters.
[Authors' description and code](https://github.com/c-yn/SFNet),
[paper](https://openreview.net/forum?id=tyZ1ChGZIKO).

Cui et al.'s FSNet extends this approach to six restoration tasks and combines
multiple receptive-field scales in one U-shaped network. Its frequency modules
remain content dependent, so fixed Fourier bands alone cannot be claimed as an
advance over it. Parameter count depends on the released task configuration;
the author's summary does not provide a verified total. FSNet's channel
modulation is a close conceptual comparison, but its described modules do not
implement this project's explicit three-expert top-k routing maps. Any claim
that the routing improves restoration requires the uniform-gate and global-gate
ablations. [Authors' description and code](https://github.com/c-yn/FSNet).

Kong et al.'s FFTformer computes attention-related operations in the frequency
domain and uses a frequency-aware feed-forward module for deblurring. Its
GoPro model has 16.6M parameters in Table 1. Local Fourier processing also makes
the transform's spatial context part of the architecture, which cautions
against treating arbitrary FFT tiling as exact. FFTformer establishes another
strong frequency-based restoration method, but does not provide the same
band-by-expert routing interpretation or the project's under-5M budget.
[Paper](https://arxiv.org/pdf/2211.12250).

Rahaman et al. show that ReLU networks tend to learn lower-frequency functions
first and study how data geometry affects that tendency. This is an analysis
of learning dynamics, not a restoration architecture with a fixed parameter
count. It motivates examining frequency-specific errors, but does not prove
that a low-frequency image band needs a large kernel or that separate experts
will specialize. Those remain empirical questions for FCSG-Net.
[Paper](https://arxiv.org/abs/1806.08734).

## Mixture of experts and gating

Shazeer et al. use noisy top-k gating to choose a small subset of experts for
each example. Their largest reported architecture has up to 137 billion
parameters, while each example activates only a fraction. Importance and load
penalties discourage routing collapse across examples. These penalties are
not equivalent to maximizing each example's routing entropy: uniform weights
at every location can prevent specialization. FCSG-Net uses spatial top-2
weights and exposes the dense probabilities for later entropy experiments.
The current implementation evaluates every expert before mixing, so sparse
weights do not yet save computation through conditional dispatch.
[Paper, sections 2 and 4](https://arxiv.org/pdf/1701.06538).

Fedus et al.'s Switch Transformer selects one expert per token and addresses
capacity, load balancing, and numerical stability. The study scales to
trillion-parameter language models, not a fixed-size restoration network.
Section 2 is relevant to routing precision and collapse, but its distributed
dispatch mechanism does not transfer automatically to spatial convolutions
whose outputs depend on neighboring pixels. Sparse image routing needs an
explicit treatment of that spatial context before a speedup can be claimed.
[Paper](https://arxiv.org/abs/2101.03961).

Hu et al.'s squeeze-and-excitation block pools features globally, compresses
their channel representation, and predicts channel scales. Ignoring biases,
two fully connected projections add approximately 2*C*C/r parameters for C
channels and reduction ratio r. FCSG-Net uses a 9-to-3-to-9 attention block
and a 9-to-3 projection, totaling 96 parameters including biases. SE supplies
cross-band channel interaction, but its global pooling does not replace a
spatial routing map. [Paper](https://arxiv.org/abs/1709.01507).

## Modern baselines and evaluation

Chen et al.'s NAFNet removes conventional nonlinear activations from restoration
blocks while retaining multiplicative gating and simplified channel attention.
Its competitive results show why architecture complexity alone is weak evidence
of a contribution. The GoPro comparison in FFTformer's Table 1 lists NAFNet at
67.9M parameters; this is a particular model, not every NAFNet configuration.
NAFNet does not supply the proposed frequency-band routing analysis, but a
smaller parameter count alone would not demonstrate better restoration.
[Paper](https://arxiv.org/html/2204.04676v1),
[parameter comparison](https://arxiv.org/pdf/2211.12250).

Zamir et al.'s Restormer combines transposed channel attention with a gated
feed-forward network and progressive patch training for high-resolution
restoration. The model in FFTformer's comparison has 26.1M parameters.
Restormer provides context for accuracy and model size, but its scores from
separate tasks or datasets cannot enter a matched archival-degradation
comparison. It also illustrates why patch size and inference context must be
reported. [Paper](https://arxiv.org/pdf/2111.09881),
[parameter comparison](https://arxiv.org/pdf/2211.12250).

Zhang et al.'s LPIPS measures distance between normalized deep features and
learns channel weights from perceptual judgments. Parameter cost depends on
the chosen feature backbone and its calibration layers; LPIPS is an evaluation
model, so its parameters must not be added to the restoration model's count.
It can disagree with PSNR because perceptual resemblance and exact pixel
agreement measure different properties. Phase 3 must record the backbone,
input normalization, and library version when adding LPIPS.
[Paper](https://arxiv.org/abs/1801.03924).

Agustsson and Timofte introduce DIV2K through the NTIRE super-resolution
challenge. It contains 800 training, 100 validation, and 100 test images.
A dataset has no learned parameter count. This project uses its clean images
to synthesize restoration pairs; DIV2K's original task does not validate the
realism of that synthetic archival pipeline. Training crops and the D2 snapshot
must use only the training split. [Dataset and paper links](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

## Implementation references

PyTorch's FFT API supplies real-input transforms and paired normalization modes.
`rfft2` stores only the nonredundant half-spectrum. Masks built with `fftfreq`
and `rfftfreq` already match its unshifted ordering; shifting them again would
misalign the bands. These operators add no learned parameters. Float32 FFTs
avoid CUDA half-precision restrictions on non-power-of-two dimensions.
[FFT reference](https://docs.pytorch.org/docs/stable/fft.html).

PyTorch's AMP recipe combines autocast with gradient scaling for float16
training. Scaling protects small gradients; it does not repair an invalid
operation or wrong loss. AMP adds no model parameters, though its scaler
state must survive checkpoint resume. FCSG-Net keeps the FFT in float32 and
the notebook checks forward and backward passes on odd image sizes under
autocast. [AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html).

## Scope of the research claim

FCSG-Net tests spatial top-k mixing of distinct shared experts after explicit
frequency decomposition, with routing related to recorded degradation strength.
Frequency decomposition, local frequency selection, and expert routing already
exist in the cited work. This combination is a project hypothesis, not a claim
of exhaustive novelty. Sparse weights currently provide an analysis mechanism;
all expert computation is included in the resource budget. Specialization and
restoration gains remain unmeasured until Phases 3 and 4.
