"""
Build the DocPilot demo corpus: ~85 passages of XR / remote-assist documentation.

Run:
    python data/build_demo_corpus.py

Outputs:
    data/corpus.json        -- passages indexed by DocPilot
    data/qa_dataset.json    -- SQuAD v2.0 format for fine-tuning / eval
"""

import collections
import json
import random
import re
import string
from pathlib import Path

random.seed(42)
OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Corpus content -- written to read like actual technical documentation.
# Each passage is self-contained so the QA model can extract an answer span
# without needing cross-passage context.
# ---------------------------------------------------------------------------

PASSAGES: dict[str, list[str]] = {
    "display_optics": [
        (
            "Waveguide displays are the dominant architecture for see-through AR glasses. "
            "A compact projector engine, typically an LCoS or micro-OLED microdisplay, injects "
            "light into a thin glass slab through an in-coupling diffractive grating. The light "
            "propagates by total internal reflection until it reaches an out-coupling grating that "
            "redirects it toward the eye. Modern single-layer waveguides achieve a diagonal "
            "field-of-view between 40 and 55 degrees and an exit pupil of 8 to 12 mm, which is wide "
            "enough that the user does not need to position the headset precisely. Multi-layer stacks "
            "use three separate waveguide plates, one per colour channel, to reduce chromatic dispersion, "
            "though this adds weight and manufacturing cost. Waveguide coupler efficiency is typically "
            "10 to 20 percent, so the projector must output over 5,000 nits to maintain visibility in "
            "bright ambient light. Waveguide scratches degrade image uniformity; field service "
            "procedures require replacing the optical module rather than attempting cleaning of "
            "internal surfaces."
        ),
        (
            "Pancake lens optics use a polarisation-based folded light path to shrink the distance "
            "between the display panel and the eye. In a conventional Fresnel VR headset the "
            "lens-to-panel distance is roughly 40 mm; pancake designs fold this to 15 to 20 mm. "
            "Light from the panel passes through a linear polariser, reflects off a curved partial "
            "mirror, passes through a quarter-wave plate, and reflects off a reflective polariser "
            "before reaching the eye. This achieves a form factor reduction of roughly 40 percent "
            "compared to Fresnel, with better edge sharpness because the lens profile can be "
            "continuous rather than stepped. The tradeoff is optical efficiency: pancake stacks "
            "transmit only 15 to 25 percent of emitted light, requiring brighter display panels "
            "and consuming more battery. Thermal management is a bigger concern in pancake-based "
            "headsets, particularly during extended enterprise use sessions of more than two hours."
        ),
        (
            "Micro-LED arrays are being developed as the next-generation emissive display for AR. "
            "Each pixel is an individually addressed inorganic LED, enabling peak brightness above "
            "1,000,000 nits, which is orders of magnitude beyond OLED's practical limit of around "
            "1,000 nits. This brightness is necessary for AR overlays to remain visible in direct "
            "sunlight. The key manufacturing challenge is mass transfer: placing millions of "
            "sub-10-micron LED chips onto a backplane with high yield. Current research prototypes "
            "achieve pixel pitches of 3 to 5 micrometres, enabling 4K resolution in a 0.5-inch "
            "diagonal die. Colour micro-LED displays typically use separate red, green, and blue "
            "sub-pixels, though some approaches use a monochromatic blue array with colour-converting "
            "phosphor caps. The technology is not yet in volume production for headsets as of 2025 "
            "but targets the 2026 to 2028 timeframe. For enterprise XR, the high brightness would "
            "improve outdoor usability in field service scenarios significantly."
        ),
        (
            "Foveated rendering exploits the structure of the human visual system. The fovea, "
            "covering roughly 5 degrees of visual angle, is the only part of the retina with high "
            "cone density. Outside that region, spatial resolution falls off sharply. An eye tracker "
            "embedded in the headset streams gaze data at 90 to 120 Hz with sub-degree accuracy, "
            "allowing the renderer to apply full shading and anti-aliasing only in a 10 to 15 degree "
            "region around the gaze point and progressively reduce quality toward the periphery. "
            "The GPU workload reduction depends on scene complexity, but 2 to 3x frame time "
            "improvement is typical in dense outdoor environments. The main engineering challenge "
            "is gaze latency: the eye moves at up to 700 degrees per second during saccades, so "
            "the foveal region must be predicted 1 to 2 frames ahead. Mispredictions are perceptible "
            "when eye tracking latency exceeds 12 ms from camera exposure to render decision. RTX "
            "3000-series GPUs support hardware variable rate shading that can implement foveated "
            "rendering with minimal driver overhead."
        ),
        (
            "The vergence-accommodation conflict (VAC) is one of the core unsolved problems in "
            "stereoscopic displays. Vergence refers to the inward rotation of both eyes to fixate "
            "on a virtual object at a given depth; accommodation refers to the lens changing shape "
            "to focus at that depth. In natural vision these two responses are tightly coupled. In "
            "a fixed-focal-plane stereoscopic display both eyes must accommodate to the display "
            "surface, typically at 1 to 2 m optical distance, regardless of where the virtual "
            "object appears in depth. This mismatch causes visual fatigue that becomes significant "
            "after 20 to 30 minutes of use. Varifocal displays address this by mechanically or "
            "optically moving the focal plane to match gaze depth as measured by the eye tracker. "
            "Liquid lens and voice-coil actuator designs can shift focal distance by 2 to 3 diopters "
            "in under 2 ms. Lightfield displays are a non-mechanical alternative but require "
            "significantly more compute and sacrifice resolution."
        ),
        (
            "Motion-to-photon latency is the total delay from a head movement to the corresponding "
            "update appearing on the display. For comfortable VR this needs to be below 20 ms; most "
            "modern headsets target 10 to 15 ms end-to-end. The budget breaks down as: sensor readout "
            "and pose integration taking 1 to 2 ms; application frame generation occupying 8 to 11 ms; "
            "display scanout and pixel response adding 2 to 5 ms. The biggest lever is asynchronous "
            "timewarp: after the application frame is submitted, the compositor reprojects it using "
            "the most recent pose estimate from just before scanout, taking approximately 0.5 ms. "
            "Asynchronous timewarp can correct for up to 8 degrees of rotation mismatch. This allows "
            "applications running at 45 fps to deliver smooth 90 Hz motion, because the timewarp "
            "fill-frame corrects most of the error. Without asynchronous timewarp, a late application "
            "frame causes visual tearing or dropped frames that are immediately perceptible as judder."
        ),
        (
            "Mixed reality passthrough cameras capture the real world so virtual content can be "
            "composited over it. Consumer headsets typically use greyscale cameras at 30 to 60 fps, "
            "while higher-end enterprise devices use colour cameras at 90 fps. The key quality metric "
            "is passthrough latency: the delay from a photon entering the camera to the corresponding "
            "pixel appearing on the display. Most headsets achieve 10 to 15 ms, though some achieve "
            "under 7 ms. Spatial calibration between cameras and the virtual scene is critical for "
            "mixed reality: even a 0.5 mm error in camera extrinsics causes noticeable misalignment "
            "between virtual objects and real surfaces. Enterprise deployments of remote assist tools "
            "depend on high-quality passthrough so remote experts can see what the field technician "
            "is seeing with minimal latency and colour distortion."
        ),
        (
            "Display calibration procedures for XR headsets address inter-unit variation in colour "
            "temperature, brightness uniformity, and geometric distortion. Factory calibration stores "
            "a per-unit polynomial warp mesh in the headset firmware to correct lens distortion so "
            "that straight lines remain straight in the rendered image. Brightness uniformity "
            "calibration maps regions that are brighter or dimmer than average and applies per-pixel "
            "gain correction at runtime. Colour calibration stores a 3x3 colour transformation matrix "
            "per display unit, ensuring that white has a consistent colour temperature, usually D65, "
            "across a production batch. For enterprise deployments with many headsets, field "
            "calibration drift is a real maintenance concern: OLED displays shift white point and "
            "brightness over thousands of hours of use. Device management systems should include "
            "calibration monitoring and scheduling to maintain consistent visual quality across a "
            "fleet of devices."
        ),
    ],

    "tracking_slam": [
        (
            "Inside-out tracking eliminates external base stations by performing all localisation "
            "using sensors on the headset itself. A typical implementation fuses data from four "
            "wide-angle monochrome cameras with a six-axis IMU running at 1000 Hz. The IMU provides "
            "high-frequency pose updates between camera frames, which run at 30 Hz for the tracking "
            "pipeline. A visual-inertial odometry algorithm propagates the IMU measurements forward "
            "and corrects accumulated drift each time a new camera frame is processed. Sparse feature "
            "tracking using ORB or FAST descriptors maintains a map of several hundred feature points "
            "visible in the current keyframe set. The drift rate for a well-calibrated system is under "
            "1 mm per metre of movement, though fast rotation and textureless environments degrade "
            "this significantly. Enterprise deployments in environments with repetitive patterns, such "
            "as warehouse racking or factory floors, can cause loop closure failures; adding AprilTag "
            "fiducials at key locations is a common mitigation."
        ),
        (
            "Simultaneous localisation and mapping (SLAM) maintains a persistent map of the "
            "environment and localises the headset within it across sessions. The map consists of "
            "a sparse point cloud of visual features, their 3D positions, and associated descriptors. "
            "When the headset is restarted in a known space, the relocalisation module attempts to "
            "match current camera observations to the stored map. This typically uses a bag-of-visual-"
            "words index for fast candidate retrieval, followed by geometric verification with "
            "PnP plus RANSAC. Successful relocalisation takes 0.5 to 2 seconds. Map maintenance is "
            "important for long-running enterprise deployments: the map grows as new areas are "
            "explored, and old map points corresponding to moved objects must be invalidated. Some "
            "enterprise MDM platforms provide a map sync service that distributes a shared SLAM map "
            "to all headsets in a facility, ensuring consistent coordinate frame alignment."
        ),
        (
            "Hand tracking in XR headsets uses the same cameras as the inside-out tracker but runs "
            "a separate neural network pipeline. The pipeline has two stages: a palm detection "
            "network runs on a full-resolution crop at 15 Hz to initialise tracking, then a second "
            "network crops the palm region and regresses 21 3D keypoints at 60 to 90 Hz. The "
            "keypoints include the wrist, all finger joints, and fingertips. Typical sub-centimetre "
            "accuracy is achieved for the index fingertip in well-lit conditions. Accuracy degrades "
            "when hands are occluded by objects, when lighting is below about 50 lux, or when hands "
            "move faster than 2 m/s. For enterprise remote assist, hand tracking allows experts to "
            "see a skeletal overlay of the technician's hands without requiring them to hold a "
            "controller, which is important when both hands are occupied with a task."
        ),
        (
            "Tracking degradation happens in several well-understood scenarios. Textureless "
            "environments such as blank white walls or concrete floors provide few visual features, "
            "leading to increased pose uncertainty and drift. Low light below approximately 20 lux "
            "makes feature extraction unreliable even with infrared illuminators. Fast head rotation "
            "above 150 degrees per second causes motion blur in camera frames, dropping feature "
            "matches. Highly reflective surfaces such as polished metal or wet floors create specular "
            "highlights that appear as false features. In each case the IMU continues to provide "
            "short-term stability, but without regular camera updates drift accumulates within 2 to "
            "3 seconds. Recovery requires the user to look at a textured, well-lit region to "
            "re-establish tracking. Enterprise deployment guidelines should identify these risk "
            "zones in a facility and suggest mitigations such as adding textured floor stickers or "
            "improving lighting in low-visibility areas."
        ),
        (
            "Multi-device spatial anchors allow several headsets to share a common world coordinate "
            "frame so virtual content placed by one device appears in the same physical location on "
            "all others. Meta's Shared Spatial Anchors use a cloud service to persist anchor poses "
            "and distribute them to other headsets in the same session. Microsoft's Azure Spatial "
            "Anchors works cross-platform across HoloLens, Android ARCore, and iOS ARKit by storing "
            "a visual anchor descriptor in the cloud that each device resolves independently. The key "
            "quality metric is co-localisation accuracy: how closely two independently tracking "
            "devices agree on the position of a shared anchor. For enterprise remote assist, "
            "co-localisation accuracy of 2 to 3 cm is sufficient to align annotation overlays with "
            "physical objects. Accuracy degrades if the facility SLAM map has not been updated "
            "recently or if the two devices are tracking in different sections of a large space."
        ),
        (
            "IMU calibration determines the intrinsic parameters including scale, axis alignment, "
            "and bias, as well as the camera-IMU extrinsic transform. Gyroscope bias, a constant "
            "offset in the angular velocity reading, accumulates into rotation error over time and "
            "must be estimated at startup by holding the headset still for 0.5 to 1 second. "
            "Accelerometer scale errors cause incorrect gravity estimation, leading to tilt drift. "
            "Factory calibration determines these parameters for each unit and stores them in the "
            "firmware. Over time, mechanical shock and temperature cycling can cause calibration "
            "drift, which manifests as a slowly drifting horizon or objects that appear to breathe "
            "slightly. Enterprise MDM software should flag headsets with high calibration residuals "
            "for recalibration. IMU recalibration typically takes 5 minutes and can be performed "
            "in the field using the headset's built-in calibration wizard."
        ),
    ],

    "ml_inference": [
        (
            "ONNX (Open Neural Network Exchange) is the standard format for exporting trained neural "
            "networks to a runtime-independent representation. Exporting from PyTorch uses "
            "torch.onnx.export(), which traces the model's computation graph and writes it as a "
            "sequence of ONNX operators. Key export parameters: opset_version should match what "
            "the inference runtime supports; ONNX Runtime 1.17 supports up to opset 19. "
            "dynamic_axes specifies which dimensions should be variable at inference time rather "
            "than fixed to the training shape. After export, onnx.checker.check_model() validates "
            "the graph. A common issue is non-exportable control flow where Python if/else depends "
            "on tensor values; these must be replaced with torch.where or similar operators. "
            "Exporting a RoBERTa-base model produces a file of approximately 500 MB. The exported "
            "model should be validated by comparing outputs between the PyTorch model and ONNX "
            "Runtime on the same input; output differences above 1e-4 indicate a tracing error."
        ),
        (
            "INT8 quantization reduces a model's weight precision from 32-bit floats to 8-bit "
            "integers, shrinking model size by roughly 4x and improving CPU inference throughput "
            "by 2 to 3x because modern CPUs have wider SIMD units for integer operations than for "
            "float. ONNX Runtime's dynamic quantization quantizes only the weights at model load "
            "time; activations remain float32 during inference. This requires no calibration dataset "
            "and introduces minimal accuracy loss for transformer models because their weight "
            "distributions are roughly symmetric. Static quantization also quantizes activations "
            "and requires a small calibration dataset of 50 to 100 representative inputs to determine "
            "activation ranges. For a RoBERTa-base QA model, dynamic INT8 quantization with ONNX "
            "Runtime reduces P95 inference latency from approximately 185 ms at FP32 on CPU to "
            "60 to 70 ms at INT8 on a modern laptop CPU with 8 or more cores, representing roughly "
            "a 3x speedup."
        ),
        (
            "TensorRT is NVIDIA's inference optimizer for GPU deployment. It takes an ONNX model, "
            "profiles which layers are most efficiently fused, and generates a serialised engine "
            "file specific to the target GPU architecture. FP16 precision halves memory bandwidth "
            "and is supported natively on all RTX-class GPUs; for most NLP models FP16 reduces "
            "accuracy by less than 0.5 percent F1. INT8 with TensorRT requires a calibration dataset "
            "to compute per-layer activation scales; a 100-sample calibration set is typically "
            "sufficient. On an RTX 3060, a FP16 TensorRT engine for RoBERTa-base with sequence "
            "length 384 achieves throughput of 40 to 60 inferences per second at batch size 1, "
            "roughly 5x faster than FP32 ONNX on the same GPU. TensorRT engines are not portable: "
            "the engine built for an RTX 3060 will not run on an RTX 4090. Engine build time is "
            "2 to 5 minutes for a transformer model; this must be included in deployment procedures."
        ),
        (
            "On-device neural network inference on XR headsets is constrained by a thermal design "
            "power budget of 5 to 10 W for standalone headsets compared to 30 to 150 W for a laptop "
            "GPU. Qualcomm's Snapdragon XR2 and XR2+ processors include a dedicated Hexagon DSP "
            "with tensor acceleration optimised for 8-bit integer matrix multiply. Models must be "
            "quantised to INT8 and have operations expressible in the Hexagon instruction set; some "
            "PyTorch ops are not supported and must be replaced. For a QA inference model, the "
            "practical constraint is memory bandwidth: the Snapdragon XR2 has 12 GB LPDDR5 shared "
            "between CPU, GPU, and DSP, with a total bandwidth of 51 GB/s. A RoBERTa-base INT8 "
            "model reading its 120 MB of weights per inference requires 120 MB divided by 51 GB/s, "
            "or approximately 2.4 ms just for weight loading, giving a theoretical minimum latency "
            "of roughly 5 ms beyond what compute alone would suggest."
        ),
        (
            "ONNX Runtime execution providers determine which hardware ONNX operations run on. "
            "CUDAExecutionProvider offloads operators to NVIDIA GPU via CUDA; "
            "TensorrtExecutionProvider wraps TensorRT for additional fusion and INT8 support; "
            "CPUExecutionProvider is the fallback. Providers are specified in priority order and "
            "ONNX Runtime assigns each operator to the highest-priority provider that supports it. "
            "For a RoBERTa model running on CUDA, nearly all operators use CUDAExecutionProvider. "
            "DirectMLExecutionProvider targets Windows DirectX 12 devices including integrated "
            "Intel and AMD GPUs, which is relevant for XR headsets with integrated graphics. "
            "For the INT8 CPU path targeting XR headsets and edge deployment, CPUExecutionProvider "
            "with intra_op_num_threads set to the core count and ORT_ENABLE_ALL graph optimisation "
            "gives the best single-query latency without requiring GPU drivers."
        ),
        (
            "Benchmarking inference correctly requires controlling for cold-start effects and "
            "measuring the metric that matters for the application. Cold-start latency on the first "
            "inference includes model weight loading, CUDA kernel compilation, and cache warming, "
            "and is typically 5 to 10 times higher than steady-state latency. Benchmarks should "
            "use at least 50 warm-up inferences before measuring. For interactive QA, the relevant "
            "metric is single-query latency at batch size 1, not throughput at large batch sizes. "
            "P95 and P99 latency are more useful than mean latency because they characterise "
            "worst-case user experience. When comparing FP32 versus INT8, the benchmark must use "
            "the same input length and ONNX opset, as different sequence lengths can change the "
            "throughput ranking. On an i9-11900H with 16 threads and ONNX Runtime INT8, RoBERTa-base "
            "QA at sequence length 384 typically achieves P50 of 55 to 65 ms and P95 of 70 to 85 ms."
        ),
    ],

    "networking": [
        (
            "WebRTC is the protocol stack used for peer-to-peer audio, video, and data streaming in "
            "browser-based and native XR remote assist applications. The media pipeline uses RTP "
            "over UDP, with SRTP providing encryption. The signalling channel uses an SDP offer/answer "
            "exchange that happens out-of-band over WebSockets or HTTP. NAT traversal uses ICE with "
            "STUN to discover public IP and port mappings and TURN servers for relay when direct "
            "connectivity fails. For enterprise deployments on corporate networks with strict firewall "
            "policies, TURN over TCP port 443 is often the fallback path, adding 10 to 30 ms of "
            "additional round-trip time compared to direct peer-to-peer. WebRTC data channels provide "
            "both reliable and unreliable message delivery; XR applications use unreliable channels "
            "for pose updates and reliable channels for annotation commands and file transfers."
        ),
        (
            "Video encoding for XR remote assist must balance image quality, bitrate, and latency. "
            "H.264 Baseline or Main profile at 4 to 10 Mbps is the common choice for compatibility. "
            "H.265 offers roughly 40 percent better quality at the same bitrate but has higher encoder "
            "and decoder compute cost. AV1 is emerging as the royalty-free alternative to H.265 with "
            "similar quality. The critical latency-affecting parameter is keyframe interval: longer "
            "intervals improve compression but mean more data must be resent after packet loss. "
            "XR remote assist typically uses 1 to 2 second keyframe intervals with reference frames "
            "that can be refreshed on demand when the scene changes drastically. Hardware-accelerated "
            "encoding on the XR headset, available on Snapdragon XR2 and later, keeps encoding "
            "latency under 10 ms. For passthrough streaming where the remote expert makes decisions "
            "based on what they see, block artefacts from fast motion must be minimised."
        ),
        (
            "Multi-access edge computing reduces round-trip latency for cloud-rendered XR by placing "
            "compute infrastructure within the 5G radio access network. A standard data centre is "
            "50 to 100 ms away from a mobile device; a MEC server co-located at the 5G base station "
            "can be 5 to 10 ms away. For cloud rendering of XR where the render loop includes a "
            "round trip of pose sent to server and compressed frame received back, MEC makes the "
            "total pose-to-pixel latency competitive with on-device rendering. For remote assist "
            "specifically, MEC can host the annotation service and document retrieval backend so "
            "that query latency is consistently low regardless of WAN conditions. Designing the "
            "backend to work with 10 ms MEC latency rather than 100 ms cloud latency changes which "
            "operations can be made synchronous during a technician's workflow."
        ),
        (
            "Network quality of service for XR traffic requires differentiating XR packets from "
            "bulk data on the enterprise network. The key flows to prioritise are voice at low "
            "latency and low bandwidth using DSCP Expedited Forwarding, video at moderate latency "
            "tolerance and high bandwidth using DSCP Assured Forwarding, and pose and control data "
            "at the lowest latency using DSCP Expedited Forwarding. Enterprise Wi-Fi should enable "
            "WMM on the access points so that DSCP markings are honoured over the air interface. "
            "The uplink bandwidth requirement for a field worker streaming 1080p30 passthrough "
            "video to a remote expert is 5 to 8 Mbps. The downlink requirement for annotation "
            "overlays is under 1 Mbps. Link utilisation above 70 percent causes queueing latency "
            "that degrades XR quality even with QoS because queues fill up at high load."
        ),
    ],

    "spatial_audio": [
        (
            "Head-related transfer functions (HRTFs) model how a sound's spectrum is modified by "
            "the shape of the listener's ears, head, and torso before reaching the eardrums. The "
            "direction-dependent colouring and inter-aural time and level differences encoded in "
            "HRTFs are the main cues the auditory system uses for 3D localisation. Personalised "
            "HRTFs measured with microphones in the listener's own ears give the most accurate "
            "externalisation, the perception that sound comes from outside the head. Generic HRTFs "
            "derived from an average of many subjects work well for left/right localisation but are "
            "less reliable for elevation cues, which vary more between individuals. For enterprise "
            "XR remote assist, spatial audio localisation helps the field technician identify which "
            "direction a remote expert is speaking from when multiple remote participants are on the "
            "call. Typical HRTF rendering pipelines convolve each audio source with the appropriate "
            "HRTF filter pair and mix the results for the left and right ear outputs."
        ),
        (
            "Audio latency in XR must remain below approximately 20 ms from the application "
            "triggering a sound event to the sound reaching the user's ears. Higher latency breaks "
            "the correlation between visual events and the corresponding audio feedback, which is "
            "jarring and reduces presence. The latency budget includes application-side sound "
            "scheduling taking 0 to 2 ms, audio thread processing including HRTF convolution taking "
            "3 to 5 ms on a mobile headset, audio driver buffering taking 5 to 10 ms, and headphone "
            "output under 1 ms. A common tuning is a 512-sample buffer at 48 kHz, giving 10.7 ms "
            "per buffer. Reducing audio buffer size decreases latency but increases the probability "
            "of dropout if the CPU is momentarily preempted. For remote assist voice communication, "
            "echo cancellation and noise suppression add 5 to 15 ms, so the total voice round-trip "
            "latency from speaker to microphone to remote ear is typically 80 to 120 ms including "
            "network."
        ),
        (
            "Acoustic echo cancellation is necessary when the headset plays audio back to the user "
            "while also recording from a microphone. Without it, the remote participant on a call "
            "hears their own voice echoed back with a delay of 80 to 200 ms. AEC maintains a model "
            "of the echo path from the speaker to the microphone and subtracts an estimate of the "
            "echo signal from the microphone input. The normalised least mean squares algorithm "
            "updates the echo path estimate adaptively in real time. The challenge is double-talk: "
            "when both parties speak simultaneously, the echo estimate update must be paused to "
            "avoid corrupting the near-end speech signal. XR headsets with on-ear speakers rather "
            "than over-ear speakers have a shorter and simpler echo path, making AEC more reliable. "
            "For bone-conduction speakers the echo path is almost entirely through tissue rather "
            "than air, which simplifies AEC considerably compared to open-air speaker configurations."
        ),
    ],

    "comfort_ergonomics": [
        (
            "Simulator sickness in XR is caused by a sensory conflict between the visual system and "
            "the vestibular system. When the visual display shows motion but the body does not "
            "physically move, the vestibular system detects no acceleration while the visual system "
            "registers it. This mismatch triggers the same neural pathway as food poisoning, causing "
            "nausea. The most effective mitigation is keeping total motion-to-photon latency under "
            "20 ms and maintaining a stable 90 Hz frame rate with no dropped frames. Other "
            "mitigations include restricting artificial locomotion to teleportation rather than "
            "continuous movement, adding a vignette that darkens the peripheral visual field during "
            "artificial motion to reduce optical flow, and using cockpit designs that provide a "
            "static visual reference frame. For enterprise XR, simulator sickness in "
            "teleportation-free apps such as hands-free annotation or document review is rare because "
            "the user's body and display match, as the user moves and the world stays fixed."
        ),
        (
            "Interpupillary distance (IPD) is the distance between the centres of the two pupils, "
            "which varies from about 56 mm to 74 mm in the adult population with a mean of 63 mm. "
            "Correct IPD setting is important in VR because the left and right rendered images must "
            "match the exact optical axes of the lenses for correct stereo fusion. An IPD mismatch "
            "of more than 3 mm causes eye strain and headache during extended use because the eyes "
            "must hold an abnormal vergence angle to fuse the stereo pair. High-end headsets provide "
            "a mechanical IPD adjustment that physically shifts the lenses; consumer headsets "
            "typically offer three discrete steps or a continuous slider. Software IPD adjustment "
            "shifts the rendered eye positions without moving the lenses, correcting convergence "
            "but not the lens offset. For enterprise fleet deployment, each user's IPD should be "
            "recorded and applied when they pick up a headset, ideally via MDM-linked user profile."
        ),
        (
            "Weight distribution in head-mounted displays significantly affects comfort during "
            "extended enterprise shifts. A headset's centre of gravity should be as close to the "
            "centre of the head as possible to minimise the torque on the neck. Front-heavy headsets "
            "common in AR glasses with forward projector modules create a moment arm that becomes "
            "fatiguing after 15 to 20 minutes. Moving battery mass to the rear can shift the centre "
            "of gravity significantly and extend comfortable wear time. Total headset weight targets "
            "are under 400 g for extended-wear enterprise scenarios and under 250 g for lightweight "
            "AR glasses. The crown strap design also matters: a wide crown strap distributes pressure "
            "over a larger area, reducing localised pressure points on the top of the head. Memory "
            "foam reduces peak pressure but traps heat more than open-cell foam alternatives."
        ),
        (
            "Safety guidelines for XR use in enterprise environments address both physical ergonomics "
            "and cognitive limits of extended XR use. Physical guidelines include taking a 10-minute "
            "break every 45 to 60 minutes of continuous headset use, not using XR headsets while "
            "operating moving machinery, and ensuring the physical work area is free of tripping "
            "hazards before starting a XR session. Contraindications for XR use include "
            "photosensitive epilepsy, as flicker from 20 to 60 Hz content can trigger seizures, "
            "recent eye surgery, and severe motion sickness disorders. Most headsets specify a "
            "minimum age of 13 years due to concerns about developing visual systems. ATEX-rated "
            "XR devices are required for use in explosive atmospheres; standard consumer XR headsets "
            "are not certified for these zones and must not be used in them."
        ),
    ],

    "enterprise_xr": [
        (
            "Remote assist systems for XR allow a remote expert to see exactly what a field "
            "technician sees through the headset's passthrough cameras and to draw 3D annotations "
            "that appear in the technician's view anchored to real-world surfaces. The expert "
            "typically uses a web browser or desktop application to view the live video feed and "
            "interact with it. Annotation types include arrows pointing to specific components, "
            "freehand drawing, callout text boxes, and 3D highlights. Annotations must be spatially "
            "anchored so they stay locked to the physical object as the technician moves around it, "
            "which requires the headset to report its real-time pose and the annotation service to "
            "transform annotation geometry accordingly. End-to-end latency from the expert drawing "
            "an annotation to it appearing in the technician's headset needs to be under 500 ms to "
            "feel responsive; most commercial systems achieve 200 to 400 ms over corporate Wi-Fi."
        ),
        (
            "Document retrieval in XR remote assist allows field workers to access technical "
            "manuals, wiring diagrams, and parts lists hands-free while working. A voice query is "
            "transcribed by an on-device ASR model, sent to a document retrieval backend, and the "
            "relevant passage is rendered as a world-locked panel in the technician's field of view. "
            "The retrieval backend uses sparse search via BM25 or dense embedding-based search over "
            "a corpus of technical documents preprocessed into passages. For extractive QA, the "
            "retrieved passage is fed to a transformer model that identifies the specific span "
            "answering the query. The full pipeline from voice input to displayed answer typically "
            "takes 1 to 3 seconds over a good Wi-Fi connection to a local edge server. Caching "
            "recently accessed documents in the headset's local storage reduces retrieval latency "
            "to under 200 ms for repeated queries during a work session."
        ),
        (
            "Digital twins in enterprise XR create a virtual replica of physical assets synchronised "
            "with real-time sensor data. For industrial equipment, OPC-UA or MQTT feeds provide "
            "machine state including speed, temperature, pressure, and fault codes to the digital "
            "twin model. In the XR view, a technician can see a 3D overlay on the physical machine "
            "showing real-time values colour-coded by operating range: green for normal, yellow for "
            "caution, and red for fault. The data ingestion pipeline involves a connector that "
            "subscribes to the sensor feed, maps values to 3D model attributes, and updates a state "
            "database at 1 to 10 Hz. Synchronisation between the physical machine and its digital "
            "twin is the main engineering challenge: sensor polling intervals, network latency, and "
            "state model update rates must all be considered to avoid showing stale data on "
            "safety-critical parameters."
        ),
        (
            "XR training simulations use immersive virtual environments to teach procedural skills "
            "with immediate feedback. Studies show that XR training for complex maintenance tasks "
            "reduces time-to-competency by 30 to 50 percent compared to traditional classroom "
            "training and produces lower error rates on first real-world task execution. The core "
            "design pattern is a step-by-step procedure where each step is explained with spatial "
            "guidance including arrows, highlights, and text panels, and the trainee must perform "
            "the correct physical action to progress. The system detects correct action completion "
            "using hand tracking, controller tracking, or sensor feedback from physical tools "
            "instrumented with IoT hardware. Assessment data including time per step, error count, "
            "and guidance requests is stored and reported to the LMS via xAPI for tracking against "
            "competency frameworks."
        ),
        (
            "Connectivity and offline operation are key design considerations for field service XR "
            "in areas with unreliable network coverage. A well-designed remote assist application "
            "should cache the current work order's procedure, relevant documentation passages, and "
            "3D assets locally so that a Wi-Fi dropout does not interrupt work. The local cache "
            "should be pre-populated before the technician enters a low-coverage zone and updated "
            "automatically when connectivity is restored. For document QA in offline mode, a "
            "locally cached ONNX model on the headset provides retrieval and extraction without "
            "any network requests. The tradeoff is that the on-device model is smaller and may have "
            "lower accuracy than the server-side model; this should be made explicit in the UI so "
            "the technician knows to verify answers through another means in ambiguous cases."
        ),
        (
            "ROI measurement for enterprise XR deployments tracks three categories of metrics: "
            "time savings showing how much faster a task is completed with XR guidance versus a "
            "paper manual, error reduction showing first-time-right rate and rework incidents, and "
            "training throughput showing how many trainees complete a curriculum per month and at "
            "what competency level. Baseline measurement before XR deployment is essential for "
            "credible ROI claims. A typical 6-month ROI study in a manufacturing context shows "
            "20 to 35 percent task time reduction for guided assembly tasks and 40 to 60 percent "
            "reduction in training time for complex procedures. The break-even point for an XR "
            "deployment covering hardware, software licensing, and content creation is typically "
            "12 to 24 months for facilities with 50 or more users and high procedure complexity."
        ),
    ],

    "platform_sdk": [
        (
            "Meta's OpenXR SDK provides OpenXR extensions specific to Quest hardware including "
            "XR_FB_spatial_entity for persistent world-locked content, XR_FB_hand_tracking_mesh "
            "for a rigged hand mesh, and XR_FB_passthrough for colour passthrough compositing. "
            "Scene understanding on Quest uses XR_FB_scene, which provides labelled floor, ceiling, "
            "wall, and furniture mesh components identified by the room setup wizard. The Meta "
            "Presence Platform adds higher-level APIs for co-presence, allowing multiple users in "
            "the same physical space to share virtual content. For enterprise remote assist on Quest, "
            "XR_FB_spatial_entity_sharing allows anchors created by one device to be transferred to "
            "another device in the same session via the network. This is the recommended integration "
            "path for multi-user enterprise scenarios where precise spatial alignment between "
            "participants is required."
        ),
        (
            "OpenXR is the Khronos Group's open standard for XR hardware and software interaction. "
            "An application written to the OpenXR API runs without modification on any conformant "
            "runtime, including Meta Quest, Windows Mixed Reality, Valve SteamVR, and HoloLens. "
            "The API divides into core features that are always present and extensions that are "
            "optional and vendor-specific. Key extensions for enterprise include XR_EXT_hand_tracking "
            "for skeleton hand data, XR_FB_passthrough for mixed reality on Meta devices, and "
            "XR_MSFT_holographic_window_attachment for HoloLens 2D panel integration. The OpenXR "
            "action system allows applications to define logical actions such as grip, confirm, and "
            "scroll, and bind them to device interaction profiles in a binding file. When the user "
            "switches headsets, the action bindings change but the application code does not, which "
            "is critical for enterprise fleet management where different device models may be "
            "deployed to different users."
        ),
        (
            "Enterprise mobile device management for XR headsets enables IT departments to manage "
            "fleets at scale. Key capabilities include remote wipe to delete all user data on a lost "
            "device, policy enforcement to block sideloading and require PIN, OTA update scheduling, "
            "and application deployment. On Quest for Business and HoloLens 2, Microsoft Intune and "
            "other MDM platforms communicate via the OMA-DM protocol. Kiosk mode locks the headset "
            "to a single application, preventing users from leaving the intended workflow. For remote "
            "assist deployments where workers use shared headsets, a user profile switch mechanism "
            "allows personal settings including IPD, language, and dominant eye to be loaded from "
            "the MDM profile when a user logs in. Application logs and crash reports can be "
            "forwarded to a central SIEM for incident investigation."
        ),
    ],

    "content_creation": [
        (
            "Unity's XR Interaction Toolkit provides a component-based system for building XR "
            "interactions that work across OpenXR-compatible headsets without platform-specific code. "
            "The core components are XR Controller mapping device inputs to Unity events, XR "
            "Interactable for objects that can be grabbed or gazed at, and XR Interactor for the "
            "controller or hand that initiates interactions. The Universal Render Pipeline is the "
            "recommended pipeline for XR because it supports single-pass instanced rendering natively "
            "and has smaller shader overhead than HDRP. For enterprise apps targeting standalone "
            "headsets, a fixed function budget of 100 draw calls per frame, 200 thousand triangles, "
            "and 4 MB of texture memory per scene is a reasonable starting point. Unity MARS extends "
            "XRI with semantic understanding components that can query scene understanding APIs on "
            "HoloLens and ARKit/ARCore devices."
        ),
        (
            "3D asset optimisation for XR requires different targets than desktop games because the "
            "GPU is weaker and the frame budget is tighter. Character meshes should be under 5,000 "
            "polygons for secondary characters at normal working distance of 1 to 3 m. Hero objects "
            "that the user works directly with can use 10,000 to 20,000 polygons. Texture resolution "
            "should be capped at 1K or 2K for most assets, as 4K textures are wasteful because the "
            "rendering resolution of current headsets does not resolve fine texture detail at typical "
            "viewing distances. Texture atlasing combines multiple small textures into a single atlas, "
            "reducing draw call count for scene objects that share material properties. For "
            "photogrammetry-captured assets of real-world equipment, a decimation pass reduces raw "
            "mesh polygon counts by 95 to 99 percent while preserving surface detail baked into a "
            "normal map."
        ),
        (
            "Photogrammetry captures 3D geometry from overlapping photographs, making it practical "
            "to create high-fidelity digital twins of real equipment for XR training and maintenance "
            "scenarios. A typical capture workflow uses 50 to 200 photographs taken on a smartphone "
            "or mirrorless camera from overlapping viewpoints at 30 to 45 degree increments around "
            "the subject. Software such as RealityCapture or Metashape aligns the photos using "
            "structure-from-motion, builds a dense point cloud, and generates a textured mesh. Output "
            "mesh sizes before optimisation are 1 to 5 million polygons. For real-time XR rendering "
            "the mesh must be decimated to 5,000 to 50,000 polygons and surface detail baked to a "
            "2K or 4K texture via cage projection. NeRF and Gaussian splatting are newer alternatives "
            "that can produce higher visual fidelity from less structured input but are harder to "
            "integrate into standard game engines as interactive objects with collision meshes."
        ),
    ],
}


# ---------------------------------------------------------------------------
# QA pair generation
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(s.split())


def _extract_qa_pairs(topic: str, text: str) -> list[dict]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 40]
    pairs = []

    number_re = re.compile(r"\d+[\.,]?\d*\s*(?:mm|ms|Hz|fps|GB|MB|kHz|dB|nits|m/s|%|W|Mbps)")
    definition_re = re.compile(r"\b(is|are|requires|enables|allows|provides|achieves)\b")

    for sent in sentences:
        if number_re.search(sent):
            q = _to_question(sent, topic)
            if q:
                pairs.append({"question": q, "answer": sent.strip()})

    for sent in sentences:
        if definition_re.search(sent) and sent not in [p["answer"] for p in pairs]:
            q = _to_question(sent, topic)
            if q:
                pairs.append({"question": q, "answer": sent.strip()})

    return pairs[:3]


def _to_question(sentence: str, topic: str) -> str | None:
    s = sentence.strip().rstrip(".")
    m = re.match(r"^([A-Z][^,]{5,40}) (is|are) ([^,]{10,80})", s)
    if m:
        subj = m.group(1).strip()
        verb = m.group(2)
        return f"What {verb} {subj.lower()}?"
    m = re.match(r"^([A-Z][^,]{5,40}) (achieves|enables|allows|provides|requires) ", s)
    if m:
        subj = m.group(1).strip()
        return f"What does {subj.lower()} do?"
    if re.search(r"\d", s):
        words = s.split()[:6]
        stub = " ".join(words[:4]).rstrip(",")
        return f"What is the value for {stub.lower()}?"
    return None


def _build_squad(passages: list[dict]) -> dict:
    all_pairs: list[tuple] = []
    for p in passages:
        for pair in _extract_qa_pairs(p["topic"], p["text"]):
            all_pairs.append((p, pair["question"], pair["answer"]))

    random.shuffle(all_pairs)
    n_train = int(0.85 * len(all_pairs))
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:]

    by_topic: dict[str, list] = collections.defaultdict(list)
    for split_pairs in (train_pairs, val_pairs):
        for p, question, answer_text in split_pairs:
            context = p["text"]
            start = context.lower().find(answer_text[:40].lower())
            if start == -1:
                continue
            qas_id = f"{p['topic']}-{p['id']}-{len(by_topic[p['topic']])}"
            by_topic[p["topic"]].append({
                "context": context,
                "qas": [{
                    "id": qas_id,
                    "question": question,
                    "answers": [{"text": answer_text, "answer_start": start}],
                    "is_impossible": False,
                }],
            })

    squad = {"version": "docpilot-v1", "data": []}
    for topic, paragraphs in by_topic.items():
        squad["data"].append({"title": topic, "paragraphs": paragraphs})

    return squad


def main():
    passages = []
    pid = 0
    for topic, texts in PASSAGES.items():
        for text in texts:
            passages.append({"id": pid, "topic": topic, "source": "demo", "text": text.strip()})
            pid += 1

    total_words = sum(len(p["text"].split()) for p in passages)
    print(f"Built {len(passages)} passages across {len(PASSAGES)} topics")
    print(f"Approx. {total_words:,} words (~{total_words // 300} pages at 300 words/page)")

    corpus_path = OUT_DIR / "corpus.json"
    corpus_path.write_text(
        json.dumps(passages, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {corpus_path}")

    squad = _build_squad(passages)
    qa_path = OUT_DIR / "qa_dataset.json"
    qa_path.write_text(
        json.dumps(squad, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    n_qa = sum(len(a["paragraphs"]) for a in squad["data"])
    print(f"Wrote {qa_path}  ({n_qa} QA pairs)")


if __name__ == "__main__":
    main()
