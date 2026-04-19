"""
Generates a 100-passage XR knowledge corpus and SQuAD-format QA pairs.

    python data/generate_corpus.py

Outputs:
    data/xr_corpus.json      – raw passages
    data/xr_qa_squad.json    – train/val split in SQuAD v2.0 format
"""

import json
import random
import re
from pathlib import Path

random.seed(42)

TOPICS = {
    "display_optics": [
        "Waveguide displays in XR headsets use diffractive optical elements to couple light "
        "from a projector into a thin glass substrate and redirect it toward the eye. Modern "
        "waveguides achieve a field-of-view of up to 55° diagonally with an exit pupil of "
        "10 mm, enabling comfortable viewing without precise headset alignment.",

        "Pancake lens stacks replace traditional Fresnel lenses in compact VR headsets by "
        "folding the optical path via polarisation-selective reflective surfaces. This reduces "
        "the lens-to-display distance from ~40 mm to ~15 mm, shrinking the optical module by "
        "roughly 40% while improving edge-to-edge sharpness.",

        "Micro-LED arrays are emerging as the preferred emissive display for AR glasses due to "
        "their brightness exceeding 1,000,000 nits peak, sub-microsecond response time, and "
        "compatibility with transparent substrates. Pixel pitches below 4 µm are now "
        "achievable in research prototypes.",

        "Eye-tracking integrated into XR displays enables foveated rendering, where full "
        "resolution is rendered only within the ~5° foveal zone tracked in real time. This "
        "reduces GPU workload by up to 3× on typical scenes without perceptible quality loss.",

        "Varifocal displays dynamically adjust the focal distance of the presented image to "
        "match the depth of gaze, directly reducing the vergence-accommodation conflict (VAC) "
        "that causes visual fatigue in fixed-focus stereoscopic headsets. VAC occurs because "
        "the eyes must converge at the virtual object depth while accommodating to the fixed "
        "focal plane of the display optics.",

        "Holographic optical elements (HOEs) recorded in photopolymer films serve as "
        "wavelength-selective mirrors and combiners in AR headsets. A single HOE layer can "
        "achieve >90% diffraction efficiency for a specific design wavelength.",

        "Retinal projection displays bypass the eye's optics entirely by scanning a laser "
        "directly onto the retina. This approach is inherently accommodative because the "
        "image is always in focus regardless of the user's refractive error.",

        "Birdbath combiners use a curved half-mirror and a reflective element to fold the "
        "optical path of a projector engine. Though bulkier than waveguides, they offer "
        "higher optical efficiency (>50%) and minimal colour non-uniformity.",

        "Liquid crystal on silicon (LCoS) microdisplays operate in reflective mode and "
        "support resolutions up to 4K in a 0.9-inch diagonal form factor, making them "
        "popular in high-end enterprise AR projector modules.",

        "Dynamic dimming in mixed-reality headsets uses a spatial light modulator placed in "
        "front of the see-through combiner to attenuate real-world light selectively, "
        "increasing perceived contrast of virtual overlays in bright environments.",
    ],

    "tracking_slam": [
        "Inside-out tracking in modern XR headsets uses visual-inertial odometry (VIO), "
        "fusing data from four or more wide-angle monochrome cameras with a 6-DOF IMU at "
        "1000 Hz. The fused estimate achieves positional drift below 1 mm/m traveled.",

        "Simultaneous Localisation and Mapping (SLAM) builds a sparse point-cloud map of "
        "the environment and localises the headset within it. Feature descriptors such as "
        "ORB or SIFT are extracted at 30 Hz and matched against the persistent map.",

        "Hand tracking pipelines typically run a two-stage CNN: a palm detector on the full "
        "image followed by a landmark regressor on the cropped palm region. 21 key-points "
        "are estimated per hand at 30 fps with sub-centimetre accuracy.",

        "Scene understanding in XR involves semantic segmentation of the camera feed to "
        "classify surfaces (floor, wall, table) enabling physics-based anchoring of virtual "
        "objects and occlusion rendering.",

        "Controller-free interaction in XR relies on hand pose estimation and gesture "
        "recognition. A finite-state machine converts continuous landmark trajectories into "
        "discrete events: pinch-start, pinch-hold, pinch-release.",

        "Room-scale guardian boundaries are computed from floor-plane detection and user "
        "walk-through calibration. The headset renders a virtual grid when the user "
        "approaches within 0.5 m of a boundary polygon.",

        "Relocalization is the process of recovering the headset's position after tracking "
        "loss. Bag-of-words indexing over ORB descriptors enables sub-second relocalization "
        "to a saved map with a recall rate above 95%.",

        "Depth cameras – structured light or time-of-flight (ToF) – supplement RGB "
        "cameras in XR for accurate surface reconstruction at up to 30 fps with 1–3 mm "
        "precision at 1 m range.",

        "Inertial Measurement Units (IMUs) in XR headsets combine a 3-axis accelerometer "
        "and 3-axis gyroscope sampled at 500–2000 Hz. IMU pre-integration provides "
        "rotational estimates between visual frames with < 0.1° error per second of drift.",

        "Persistent content anchoring stores spatial anchors as 6-DOF poses relative to "
        "map landmarks, enabling virtual objects to remain fixed in the physical world "
        "across sessions and devices sharing the same map.",
    ],

    "rendering_engines": [
        "Forward rendering in XR evaluates lighting per-fragment for each light source, "
        "which scales as O(geometry × lights). For the typical single-directional-light XR "
        "scenario, forward rendering is preferred over deferred for its lower bandwidth "
        "overhead on mobile GPUs.",

        "Asynchronous Timewarp (ATW) re-projects the last rendered frame to the current "
        "head orientation before display, compensating for GPU frame-time variance. This "
        "eliminates judder when frame time exceeds the display refresh budget of 11.1 ms "
        "at 90 Hz.",

        "Variable Rate Shading (VRS) tiles the render target and assigns a coarser shading "
        "rate to peripheral regions identified by the eye tracker, reducing fragment shader "
        "invocations by 40–60% compared to full-rate rendering.",

        "Reprojection techniques use the depth buffer to warp rendered pixels to new "
        "viewpoints, allowing the display system to interpolate frames between full renders "
        "and achieve effective 120 fps output from a 60 fps render loop.",

        "Level-of-detail (LOD) streaming in XR dynamically substitutes high-poly meshes "
        "with lower-resolution proxies based on the angular size of objects in the field of "
        "view, keeping draw call counts below 1000 per eye per frame.",

        "Render graphs decouple resource declaration from execution ordering in modern XR "
        "engines, enabling automatic hazard detection, aliasing of transient resources, and "
        "multi-GPU scheduling across the application CPU and DSP.",

        "Single-Pass Stereo Rendering draws both eye views in a single draw call by "
        "broadcasting geometry to two render targets via a geometry shader or via "
        "multiview extensions (OVR_multiview2 / GL_OVR_multiview), halving CPU draw-call "
        "overhead.",

        "Physically Based Rendering (PBR) in XR uses the Cook-Torrance BRDF with metallic "
        "and roughness parameters, enabling consistent appearance of virtual objects under "
        "varying real-world lighting conditions captured by an environment HDR probe.",

        "Occlusion culling in XR uses hardware occlusion queries or software rasterization "
        "of a depth hierarchy (HZB) to skip rendering objects fully hidden behind opaque "
        "surfaces, reducing GPU vertex work by 20–50% in cluttered scenes.",

        "Compute shaders on the XR GPU handle particle systems, cloth simulation, and "
        "post-process effects (bloom, chromatic aberration correction for lens distortion) "
        "asynchronously, overlapping with the graphics queue to improve frame pacing.",
    ],

    "audio_spatial": [
        "Head-Related Transfer Functions (HRTFs) encode how the pinnae, head, and torso "
        "modify incoming sound for each ear. Personalised HRTFs derived from ear "
        "photographs improve externalisation of virtual sound sources by ~30% versus "
        "generic defaults in listening studies.",

        "Room Impulse Response (RIR) estimation in XR uses acoustic ray-casting against "
        "the reconstructed room geometry to simulate early reflections and late reverberation, "
        "anchoring virtual audio to the physical space.",

        "Ambisonics is a scene-based spatial audio format that encodes a sound field as "
        "spherical harmonic coefficients (B-format). First-order Ambisonics uses four "
        "channels (W, X, Y, Z); third-order uses 16 channels with higher spatial resolution.",

        "Bone conduction transducers in XR headsets deliver audio through the skull, "
        "preserving open-ear situational awareness while providing moderate sound pressure "
        "levels for spoken notifications and spatial cues.",

        "Acoustic occlusion in XR attenuates sound sources blocked by geometry, applying "
        "low-pass filtering to model the frequency-dependent absorption of walls and "
        "furniture between source and listener.",

        "Lip-sync driven avatar animation uses phoneme extraction from the microphone "
        "signal at 20 ms frames and drives a set of viseme blend shapes on the avatar face, "
        "reducing the uncanny valley effect in social XR.",

        "Echo cancellation in XR headsets uses adaptive filters (NLMS or RLS algorithms) "
        "to subtract the headset's own speaker output from the microphone signal, enabling "
        "voice commands even while audio is playing.",

        "Spatial audio rendering latency must remain below 10 ms end-to-end to avoid "
        "perceptible desynchronisation between head rotation and sound-field update, which "
        "causes simulation sickness in sensitive users.",

        "Near-field audio effects model the increase in low-frequency energy (proximity "
        "effect) and intensity as a virtual source approaches within 1 m of the listener, "
        "improving presence for close-up interactions.",

        "Audio LOD in XR reduces the convolution cost of HRTF rendering by switching from "
        "full HRIR convolution to parametric panning for sources beyond 10 m or below a "
        "perceptual threshold of -40 dBFS relative to the loudest source.",
    ],

    "networking_webrtc": [
        "WebRTC enables real-time peer-to-peer audio, video, and data exchange in XR "
        "applications. ICE candidate gathering, STUN traversal, and DTLS-SRTP encryption "
        "are handled transparently, achieving sub-200 ms round-trip latency on LAN.",

        "QUIC transport underlies modern XR streaming protocols, providing multiplexed "
        "streams over UDP with congestion control and packet-loss recovery without "
        "head-of-line blocking between independent media streams.",

        "Predictive networking for XR uses Kalman-filtered head-pose extrapolation to "
        "pre-fetch tile-based panoramic video segments likely to enter the viewport within "
        "the next 500 ms, reducing stall probability by 60%.",

        "Network slicing in 5G NR reserves a dedicated logical network with guaranteed "
        "latency ≤ 5 ms and bandwidth ≥ 100 Mbps for XR traffic, isolating it from "
        "best-effort consumer data.",

        "Adaptive bitrate (ABR) streaming in cloud-XR monitors available bandwidth and "
        "adjusts the codec quantisation parameter every GOP (≈ 500 ms) to keep the buffer "
        "above 200 ms while maximising visual quality.",

        "Point-cloud streaming compresses 3D sensor output using geometry-based MPEG "
        "G-PCC or video-based V-PCC codecs, achieving 10:1 compression of LiDAR frames "
        "at 30 fps for real-time telepresence.",

        "Edge computing for XR offloads rendering or AI inference to a MEC (Multi-Access "
        "Edge Computing) server co-located with the 5G base station, reducing round-trip "
        "from ~50 ms (cloud) to ~5 ms (edge).",

        "Multiplayer state synchronisation in XR uses delta-compression and interest "
        "management to limit the per-client bandwidth for a 50-player shared space to "
        "under 200 kbps while maintaining positional accuracy within 2 cm.",

        "Forward Error Correction (FEC) in XR video streams adds redundant packets to "
        "recover from burst packet loss of up to 10% without retransmission, keeping "
        "display continuity for the viewer.",

        "Jitter buffers in XR audio pipelines absorb network timing variation by "
        "adaptively sizing the playout delay between 20 ms and 80 ms, balancing "
        "smoothness against added latency.",
    ],

    "comfort_ergonomics": [
        "Simulator sickness in XR is caused by sensory conflict between visual motion "
        "and vestibular signals. Vection – the illusion of self-motion from optic flow – "
        "is the primary trigger; reducing field-of-view dynamically during locomotion "
        "mitigates nausea in susceptible users.",

        "Interpupillary distance (IPD) adjustment in XR headsets is critical: a 5 mm "
        "mismatch between the user's IPD and the headset's lens separation increases "
        "eye strain and reduces stereo depth accuracy by up to 25%.",

        "Weight distribution in XR headsets affects user comfort during extended sessions. "
        "A centre-of-gravity positioned close to the forehead reduces neck torque; "
        "rear-mounted batteries counterbalance front-heavy optical modules.",

        "Thermal management in tethered XR headsets routes waste heat from the compute "
        "module through a vapour chamber to a rear aluminium heat spreader, keeping the "
        "user-contact surfaces below 43°C (ASTM C1055 threshold for discomfort).",

        "Haptic feedback via eccentric rotating mass (ERM) or linear resonant actuator "
        "(LRA) motors in XR controllers conveys contact events with 10–200 Hz vibration "
        "profiles. LRAs achieve sharper transient response (< 10 ms rise time) than ERMs.",

        "Prescription lens inserts for XR headsets are CNC-machined from CR-39 or "
        "polycarbonate and magnetically attach inside the headset frame, allowing users "
        "with refractive errors to use the device without contact lenses.",

        "Extended session guidelines for XR recommend breaks of 10–15 minutes every "
        "30 minutes of continuous use, based on clinical studies showing significant "
        "reduction in eye strain, headache, and disorientation with scheduled rest.",

        "Face gasket materials in XR headsets use antimicrobial foam with a moisture-wicking "
        "textile cover. Replaceable gaskets reduce hygiene concerns in shared device "
        "deployments in enterprise and arcade settings.",

        "Optical distortion correction in XR applies a pre-warp to the rendered image "
        "that inversely matches the barrel distortion profile of the lens system, computed "
        "per headset unit from factory calibration data stored in onboard EEPROM.",

        "Passthrough latency – the end-to-end delay from real-world event to display of "
        "the camera-captured image – must be below 20 ms to preserve the sense of "
        "presence in mixed-reality tasks such as typing on a physical keyboard.",
    ],

    "ml_inference": [
        "Transformer-based question answering models fine-tuned on domain corpora use a "
        "linear span-extraction head on top of the final hidden states to predict start "
        "and end token positions of the answer within a provided context passage.",

        "ONNX (Open Neural Network Exchange) format provides a hardware-agnostic "
        "intermediate representation for deep learning models. The ONNX Runtime executes "
        "the graph via pluggable execution providers: CPU, CUDA, TensorRT, CoreML.",

        "INT8 post-training quantization maps 32-bit floating-point weights and activations "
        "to 8-bit integers using per-channel scale factors calibrated on a small "
        "representative dataset, reducing model size by 4× and improving inference "
        "throughput by 2–3× on hardware with INT8 SIMD support.",

        "Dynamic quantization quantizes only the weights offline while keeping activations "
        "in float at runtime, trading a smaller speedup (1.5–2×) for the elimination of "
        "the calibration dataset requirement.",

        "Knowledge distillation trains a compact student model to mimic the output "
        "distribution of a larger teacher, achieving 95–98% of teacher accuracy with "
        "30–50% fewer parameters, suitable for on-device XR inference.",

        "TensorRT optimises ONNX graphs for NVIDIA GPUs by fusing layers, selecting "
        "optimal CUDA kernels, and applying FP16 or INT8 precision. Typical speedup "
        "over vanilla PyTorch: 2–5× for transformer inference at batch size 1.",

        "Model pruning removes weights with magnitude below a threshold, introducing "
        "structured sparsity in attention heads and feed-forward layers. 50% unstructured "
        "sparsity achieves near-lossless accuracy with 1.3–1.8× latency gain on CPUs "
        "with sparse BLAS support.",

        "Latency profiling of transformer inference identifies the self-attention "
        "computation (O(n²) in sequence length) and the linear projections as dominant "
        "contributors. Sequence lengths above 384 tokens cause super-linear latency growth.",

        "Batch inference in XR QA systems queues multiple user queries and processes them "
        "together, exploiting GPU parallelism to improve throughput at the cost of added "
        "queuing latency. A batch size of 4–8 typically maximises GPU utilisation.",

        "Embedding caching stores pre-computed context encodings for frequently accessed "
        "passages, avoiding redundant encoder forward passes and reducing end-to-end "
        "latency for repeated queries against static documents.",
    ],

    "content_creation": [
        "Photogrammetry reconstructs 3D models from overlapping photographs using "
        "Structure-from-Motion (SfM) and Multi-View Stereo (MVS). A 100-photo capture "
        "of a room-sized object yields mesh accuracy within 2 mm at 4K texture resolution.",

        "Gaussian Splatting represents scenes as a set of anisotropic 3D Gaussians "
        "optimised to reproduce training views. Real-time rendering at 1080p achieves "
        "100+ fps on an RTX 3080, making it attractive for XR telepresence.",

        "Procedural generation in XR uses grammar-based or noise-driven algorithms to "
        "produce infinite non-repeating environments. Wave Function Collapse generates "
        "tile-based layouts consistent with hand-authored adjacency constraints.",

        "glTF 2.0 is the standard interchange format for 3D assets in XR pipelines. "
        "It supports PBR material definitions, skeletal animation, morph targets, and "
        "extensions for XR-specific features (KHR_materials_transmission, EXT_mesh_gpu_instancing).",

        "Volumetric video captures a performer using a rig of 50–100 RGB-D cameras and "
        "reconstructs a textured mesh per frame at 30 fps. Compressed bitrates of "
        "50–200 Mbps are required for full-body at 5 mm resolution.",

        "AI-generated textures using diffusion models conditioned on text prompts can "
        "produce tileable PBR material maps (albedo, normal, roughness, metallic) at "
        "2K resolution in under 5 seconds on an RTX 3090.",

        "Level design for XR prioritises movement economy: walkable surfaces are kept "
        "within a 4 m × 3 m guardian space and vertical interaction zones are bounded "
        "between 0.5 m and 2.0 m to accommodate seated and standing users.",

        "Shader graphs in XR engines provide a node-based visual interface for authoring "
        "GLSL/HLSL shaders without hand-coding, enabling artists to create iridescent, "
        "holographic, and energy-shield surface effects.",

        "Dynamic occlusion masking for mixed reality requires real-time mesh segmentation "
        "to identify foreground objects (hands, furniture) and render virtual content "
        "behind them, preserving depth ordering between real and virtual elements.",

        "Animation retargeting maps motion capture data recorded on a performer skeleton "
        "to differently proportioned avatar rigs using inverse kinematics and bone-length "
        "normalisation, preserving the intent of the motion across body shapes.",
    ],

    "platform_sdk": [
        "OpenXR is the Khronos cross-platform API standard for XR runtimes. Applications "
        "call xrBeginFrame / xrEndFrame and submit swapchain images; the runtime handles "
        "reprojection, display synchronisation, and tracking without platform-specific code.",

        "ARKit on iOS provides markerless world tracking, plane detection, scene "
        "reconstruction, face tracking, and LiDAR-based occlusion through a unified "
        "ARSession API, updated annually with new WWDC capability announcements.",

        "ARCore on Android provides a feature-equivalent AR platform with Motion Tracking, "
        "Environmental Understanding, Light Estimation, and Cloud Anchors for "
        "multi-user shared AR sessions across Android devices.",

        "Meta Presence Platform SDK provides body tracking, eye tracking, face tracking, "
        "Scene API for mesh and boundary access, and Interaction SDK for hand and "
        "controller input with physics-based grab and throw on Quest headsets.",

        "Unity XR Interaction Toolkit (XRI) provides a component-based framework for "
        "XR interaction: ray interactors, direct grab, socket interactors, and UI "
        "interaction, abstracting over OpenXR, ARKit, and ARCore backends.",

        "WebXR Device API exposes VR and AR capabilities to web applications via "
        "navigator.xr.requestSession(), enabling immersive experiences in Chrome, "
        "Edge, and Firefox without native app installation.",

        "Spatial anchors in cross-platform XR SDKs persist 6-DOF pose references tied "
        "to physical locations. Azure Spatial Anchors and ARCore Cloud Anchors support "
        "multi-user, multi-device localisation to a shared anchor within 2–5 cm accuracy.",

        "Passthrough API in Meta Quest SDK provides camera feed access as a layer "
        "composited beneath the application's rendered content, enabling mixed-reality "
        "experiences with software-controlled opacity and colour balance.",

        "Scene understanding mesh from LiDAR (ARKit) or structured light (HoloLens 2) "
        "is delivered as a triangulated world mesh updated at 1–10 Hz, suitable for "
        "physics collision, occlusion, and navigation mesh generation.",

        "Input action systems in XR SDKs abstract hardware-specific button layouts "
        "into semantic actions (Grip, Trigger, PrimaryButton) mapped at runtime to "
        "device capabilities, enabling single-codebase support for 10+ controller types.",
    ],

    "enterprise_xr": [
        "Remote assistance XR applications stream a first-person camera view from a "
        "field technician's headset to an expert's desktop, who annotates the live feed "
        "with spatial markers visible in the technician's AR overlay.",

        "Digital twin integration connects XR headsets to IoT sensor streams and "
        "enterprise ERP/MES systems, overlaying real-time KPIs (temperature, pressure, "
        "throughput) on physical machinery as floating AR labels.",

        "Training simulation in XR achieves 75–80% knowledge retention rates versus "
        "40% for conventional e-learning, attributed to procedural practice in realistic "
        "3D environments with immediate corrective feedback.",

        "CAD visualisation in XR imports STEP and PLM files via translation pipelines "
        "(PTC Creo, Siemens NX, CATIA export) and renders assembly models at "
        "millimetre accuracy for design review and virtual prototyping.",

        "Safety training scenarios in XR simulate hazardous environments (confined "
        "spaces, electrical switchgear, chemical spills) that are impractical or "
        "dangerous to reproduce physically, reducing workplace incident rates by "
        "up to 43% in longitudinal enterprise studies.",

        "Multi-user collaboration in enterprise XR supports up to 50 co-located or "
        "remote avatars sharing a persistent spatial workspace, with voice, gesture, "
        "and annotation tools synchronised over a dedicated TURN relay.",

        "Wayfinding and navigation overlays in XR use floor-projected path arrows and "
        "door-frame highlights to guide workers through large facilities, reducing "
        "pick-and-pack errors in warehouse logistics by 25%.",

        "XR device management at enterprise scale uses MDM (Mobile Device Management) "
        "platforms to push firmware updates, configure Wi-Fi profiles, and enforce "
        "content restrictions across fleets of hundreds of headsets over-the-air.",

        "Compliance and audit logging in enterprise XR records session timestamps, "
        "operator IDs, procedure steps completed, and pass/fail outcomes to an "
        "immutable audit trail for regulatory reporting.",

        "ROI measurement for enterprise XR deployments tracks training time reduction, "
        "error rate improvement, and travel cost savings, with average payback periods "
        "of 12–18 months reported across manufacturing and healthcare verticals.",
    ],
}


def _first_noun(sentence: str) -> str:
    return " ".join(sentence.split()[:5]).rstrip(".,;:").lower()


def build_qa_pairs(topics: dict) -> list[dict]:
    templates = [
        lambda ctx, s: (f"What is {_first_noun(s)}?", s),
        lambda ctx, s: (f"How does {_first_noun(s)} work in XR?", s),
        lambda ctx, s: (f"What are the benefits of {_first_noun(s)} in XR?", s),
        lambda ctx, s: (f"What is the purpose of {_first_noun(s)}?", s),
        lambda ctx, s: (f"Describe {_first_noun(s)} in the context of XR.", s),
    ]
    pairs = []
    for topic, passages in topics.items():
        for passage in passages:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage) if len(s.strip()) > 30]
            for sent in sentences[:3]:
                question, answer_text = random.choice(templates)(passage, sent)
                start = passage.find(answer_text)
                if start == -1:
                    continue
                pairs.append({
                    "id": f"{topic}_{len(pairs)}",
                    "context": passage,
                    "question": question,
                    "answers": {"text": [answer_text], "answer_start": [start]},
                })
    return pairs


def make_squad_format(pairs: list[dict], val_ratio: float = 0.15) -> dict:
    random.shuffle(pairs)
    split = int(len(pairs) * (1 - val_ratio))

    def wrap(items):
        return {
            "version": "v2.0",
            "data": [
                {
                    "title": p["id"],
                    "paragraphs": [{
                        "context": p["context"],
                        "qas": [{"id": p["id"], "question": p["question"],
                                 "answers": p["answers"], "is_impossible": False}],
                    }],
                }
                for p in items
            ],
        }

    return {"train": wrap(pairs[:split]), "validation": wrap(pairs[split:])}


def main():
    out_dir = Path(__file__).parent
    corpus = [
        {"topic": topic, "passage_id": i, "text": text}
        for topic, passages in TOPICS.items()
        for i, text in enumerate(passages)
    ]
    (out_dir / "xr_corpus.json").write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    print(f"Wrote {len(corpus)} passages")

    pairs = build_qa_pairs(TOPICS)
    squad = make_squad_format(pairs)
    (out_dir / "xr_qa_squad.json").write_text(json.dumps(squad, indent=2), encoding="utf-8")
    n_train = len(squad["train"]["data"])
    n_val = len(squad["validation"]["data"])
    print(f"Wrote {n_train} train / {n_val} val QA pairs")


if __name__ == "__main__":
    main()
