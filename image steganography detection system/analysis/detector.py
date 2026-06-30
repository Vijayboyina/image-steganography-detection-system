"""
Core steganography detection algorithms.
LSB, Chi-square, RS, Histogram, DCT/FFT analysis.

Empirical calibration (512x512 gradient+noise image):
  RS d_rm:    clean=−0.004, 10%=−0.017, 30%=−0.043, 80%=−0.115, 100%=−0.145
  LSB corr:   clean=0.052,  30%=0.045,  80%=0.009,  100%=0.001
  Chi p:      clean=0.000,  80%=0.661,  100%=1.000
  Hist nd:    clean=0.162,  80%=0.049,  100%=0.032
  DCT hf:     clean=0.918,  100%=0.916  ← no signal, used for display only

Primary detector:   RS d_rm
Secondary:          LSB correlation (low corr only counts if RS also sees signal)
Tertiary:           Chi-square, Histogram (high payload only)
DCT:                Display/report only — not used in detection logic
"""
import numpy as np
from PIL import Image
from scipy import stats
from scipy.fft import fft2, fftshift
import warnings
warnings.filterwarnings('ignore')


class StegoAnalyzer:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.image_rgb = self.image.convert("RGB")
        self.np_image = np.array(self.image_rgb)
        self.results = {}

    # ------------------------------------------------------------------
    # 1. LSB Analysis — spatial correlation
    # ------------------------------------------------------------------
    def lsb_analysis(self) -> dict:
        """
        Measures horizontal + vertical spatial correlation of LSB planes.
        Natural images: structured LSBs → higher correlation (≈0.05).
        Stego images:   random LSBs    → near-zero correlation (≈0.001).
        Only used as a secondary signal when RS also detects something.
        """
        channels = {"Red": 0, "Green": 1, "Blue": 2}
        channel_results = {}

        for name, idx in channels.items():
            lsb = (self.np_image[:, :, idx] & 1).astype(np.float32)
            h_corr = float(np.corrcoef(
                lsb[:, :-1].flatten(), lsb[:, 1:].flatten())[0, 1])
            v_corr = float(np.corrcoef(
                lsb[:-1, :].flatten(), lsb[1:, :].flatten())[0, 1])
            avg_corr = (abs(h_corr) + abs(v_corr)) / 2
            ones_ratio = float(np.mean(lsb))
            _, counts = np.unique(lsb, return_counts=True)
            probs = counts / counts.sum()
            entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
            channel_results[name] = {
                "h_correlation":   h_corr,
                "v_correlation":   v_corr,
                "avg_correlation": avg_corr,
                "ones_ratio":      ones_ratio,
                "entropy":         entropy,
            }

        avg_corr_all = float(np.mean(
            [v["avg_correlation"] for v in channel_results.values()]))
        suspicion_score = float(np.mean(
            [v["entropy"] for v in channel_results.values()]))

        # Threshold set low — only fires at 80%+ payload where corr < 0.010
        # Not used standalone; weighted scoring applies RS gate
        detected = avg_corr_all < 0.010

        self.results["lsb"] = {
            "channels":        channel_results,
            "avg_correlation": avg_corr_all,
            "suspicion_score": suspicion_score,
            "detected":        detected,
            "description": (
                "LSB correlation analysis measures spatial correlation of "
                "least-significant bits. Natural images have correlated LSBs; "
                "steganographic embedding destroys this correlation."
            ),
        }
        return self.results["lsb"]

    # ------------------------------------------------------------------
    # 2. Chi-Square Test
    # ------------------------------------------------------------------
    def chi_square_analysis(self) -> dict:
        """
        PoV chi-square test.
        LSB embedding equalises pair counts: hist[2k] ≈ hist[2k+1].
        High p-value = pairs are suspiciously equal = evidence of embedding.
        Reliable only at 70%+ payload. Chi fires at p > 0.30.
        """
        channel_names = ["Red", "Green", "Blue"]
        channel_results = {}

        for idx, name in enumerate(channel_names):
            ch = self.np_image[:, :, idx].flatten().astype(int)
            hist = np.bincount(ch, minlength=256).astype(float)
            obs, exp = [], []
            for v in range(0, 256, 2):
                t = hist[v] + hist[v + 1]
                if t > 0:
                    obs.extend([hist[v], hist[v + 1]])
                    exp.extend([t / 2.0, t / 2.0])
            obs = np.array(obs, dtype=float)
            exp = np.array(exp, dtype=float)
            mask = exp > 0
            chi2 = float(np.sum(((obs[mask] - exp[mask]) ** 2) / exp[mask]))
            df = int(np.sum(mask) - 1)
            p = float(1 - stats.chi2.cdf(chi2, df)) if df > 0 else 0.0
            channel_results[name] = {"chi2_statistic": chi2, "p_value": p}

        avg_p = float(np.mean([v["p_value"]
                      for v in channel_results.values()]))
        avg_chi2 = float(np.mean([v["chi2_statistic"]
                         for v in channel_results.values()]))
        detected = avg_p > 0.30   # fires only at heavy embedding

        self.results["chi_square"] = {
            "channels":        channel_results,
            "average_p_value": avg_p,
            "average_chi2":    avg_chi2,
            "detected":        detected,
            "description": (
                "Chi-square test checks whether pixel-value pair counts are "
                "equalised — a hallmark of LSB embedding at high payload."
            ),
        }
        return self.results["chi_square"]

    # ------------------------------------------------------------------
    # 3. RS Analysis — primary detector
    # ------------------------------------------------------------------
    def rs_analysis(self) -> dict:
        """
        RS analysis (Fridrich et al.).
        d_rm = rm − rm_  (negative in stego, near-zero in clean).

        Empirical calibration on 512×512 gradient+noise:
          clean    d_rm ≈ −0.004
          10%      d_rm ≈ −0.017
          30%      d_rm ≈ −0.043
          50%      d_rm ≈ −0.074
          80%      d_rm ≈ −0.115
          100%     d_rm ≈ −0.145

        Threshold: −0.015 catches ~8% payload while avoiding clean images.
        """
        def smoothness(g):
            return float(np.sum(np.abs(np.diff(g.astype(float)))))

        def apply_f1(g, mask):
            out = g.copy().astype(np.int16)
            out[mask == 1] ^= 1
            return out.astype(np.uint8)

        def apply_fm1(g, mask):
            out = g.copy().astype(np.int16)
            for i, m in enumerate(mask):
                if m == 1:
                    out[i] = (out[i] - 1) % 256 if out[i] % 2 == 0 \
                        else (out[i] + 1) % 256
            return out.astype(np.uint8)

        mask_pos = np.array([0, 1, 0, 1])
        group_size = 4
        channel_results = {}

        for idx, name in enumerate(["Red", "Green", "Blue"]):
            ch = self.np_image[:, :, idx].flatten().astype(np.uint8)
            n = (len(ch) // group_size) * group_size
            groups = ch[:n].reshape(-1, group_size)
            Rm = Sm = Rm_ = Sm_ = 0
            total = len(groups)

            for g in groups:
                d0 = smoothness(g)
                dp = smoothness(apply_f1(g, mask_pos))
                dm = smoothness(apply_fm1(g, mask_pos))
                if dp > d0:
                    Rm += 1
                elif dp < d0:
                    Sm += 1
                if dm > d0:
                    Rm_ += 1
                elif dm < d0:
                    Sm_ += 1

            if total == 0:
                channel_results[name] = {"payload_estimate": 0.0, "d_rm": 0.0}
                continue

            rm = Rm / total
            sm = Sm / total
            rm_ = Rm_ / total
            sm_ = Sm_ / total
            d_rm = rm - rm_
            d_sm = sm_ - sm

            denom = abs(d_rm) + d_sm
            payload = float(np.clip(abs(d_rm) / (denom + 1e-10), 0.0, 1.0)) \
                if d_rm < 0 else 0.0

            channel_results[name] = {
                "payload_estimate": payload,
                "d_rm": d_rm, "d_sm": d_sm,
                "rm": rm, "sm": sm, "rm_": rm_, "sm_": sm_,
            }

        avg_payload = float(np.mean(
            [v["payload_estimate"] for v in channel_results.values()]))
        avg_d_rm = float(np.mean(
            [v["d_rm"] for v in channel_results.values()]))

        # −0.015 safely above clean noise floor (−0.004) yet catches 10% stego
        detected = avg_d_rm < -0.015

        self.results["rs_analysis"] = {
            "channels":                 channel_results,
            "average_payload_estimate": avg_payload,
            "avg_d_rm":                 avg_d_rm,
            "detected":                 detected,
            "description": (
                "RS analysis measures asymmetry between Regular and Singular "
                "pixel groups under positive and negative bit-flip masks. "
                "Clean images are symmetric (d_rm ≈ 0); "
                "embedding creates a negative d_rm proportional to payload."
            ),
        }
        return self.results["rs_analysis"]

    # ------------------------------------------------------------------
    # 4. Histogram Analysis
    # ------------------------------------------------------------------
    def histogram_analysis(self) -> dict:
        """
        Measures how equalised pixel-value pairs are.
        Natural images:  large pair-count differences (norm_diff ≈ 0.16+).
        High payload:    pairs nearly equal          (norm_diff < 0.05).
        Only fires at 70%+ payload.
        """
        channel_results = {}

        for idx, name in enumerate(["Red", "Green", "Blue"]):
            ch = self.np_image[:, :, idx].flatten().astype(int)
            hist = np.bincount(ch, minlength=256).astype(float)
            even = hist[0::2]
            odd = hist[1::2]
            norm_diff = float(np.mean(np.abs(even - odd)) /
                              (np.mean(hist) + 1e-10))
            pov_ratios = [hist[v] / (hist[v] + hist[v+1])
                          for v in range(0, 256, 2)
                          if hist[v] + hist[v+1] > 10]
            sym_std = float(np.std(pov_ratios)) if pov_ratios else 0.5
            channel_results[name] = {
                "norm_diff":        norm_diff,
                "pov_symmetry_std": sym_std,
                "histogram":        hist.tolist(),
            }

        avg_norm_diff = float(np.mean(
            [v["norm_diff"] for v in channel_results.values()]))
        avg_sym_std = float(np.mean(
            [v["pov_symmetry_std"] for v in channel_results.values()]))

        # Both conditions must hold: pairs nearly equal AND low variance
        detected = avg_norm_diff < 0.05 and avg_sym_std < 0.04

        self.results["histogram"] = {
            "channels":             channel_results,
            "avg_norm_diff":        avg_norm_diff,
            "avg_pov_symmetry_std": avg_sym_std,
            "avg_comb_score":       1.0 - avg_norm_diff,  # compat with report
            "detected":             detected,
            "description": (
                "Histogram analysis measures equalisation of pixel-value pairs. "
                "Natural images show large pair-count differences; "
                "high-payload embedding collapses these differences."
            ),
        }
        return self.results["histogram"]

    # ------------------------------------------------------------------
    # 5. DCT / FFT Analysis  (display only — not used in detection)
    # ------------------------------------------------------------------
    def dct_analysis(self) -> dict:
        """
        Computes frequency-domain metrics for display in the report.
        Empirically, hf_ratio does NOT change meaningfully with LSB payload
        and therefore cannot reliably detect steganography.
        Result is always detected=False; shown for completeness only.
        """
        gray = np.array(self.image.convert("L")).astype(float)
        f = fftshift(fft2(gray))
        mag = np.abs(f)
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        y_g, x_g = np.ogrid[:h, :w]
        dist = np.sqrt((y_g - cy) ** 2 + (x_g - cx) ** 2)

        r_low = min(h, w) // 8
        r_mid = min(h, w) // 4
        r_high = min(h, w) // 2

        mid_mask = (dist >= r_low) & (dist < r_mid)
        high_mask = (dist >= r_mid) & (dist < r_high)

        mid_mean = float(np.mean(mag[mid_mask])) if mid_mask.any() else 1.0
        high_mean = float(np.mean(mag[high_mask])) if high_mask.any() else 0.0
        high_min = float(np.min(mag[high_mask])) if high_mask.any() else 0.0
        hf_ratio = high_mean / (mid_mean + 1e-10)
        hf_flatness = high_min / (high_mean + 1e-10)

        log_mag = np.log1p(mag)
        flat_norm = log_mag.flatten() / (log_mag.sum() + 1e-10)
        spec_ent = float(-np.sum(flat_norm * np.log2(flat_norm + 1e-12)))

        # Always False — DCT hf_ratio is not a reliable LSB stego detector
        detected = False

        self.results["dct"] = {
            "high_frequency_ratio": hf_ratio,
            "hf_flatness":          hf_flatness,
            "spectral_entropy":     spec_ent,
            "detected":             detected,
            "magnitude_spectrum":   log_mag.tolist(),
            "description": (
                "DCT/FFT frequency-domain analysis. The log-magnitude spectrum "
                "is shown for visual inspection. Note: high-frequency energy ratio "
                "does not reliably distinguish LSB steganography from natural "
                "image content and is provided for reference only."
            ),
        }
        return self.results["dct"]

    # ------------------------------------------------------------------
    # 6. Overall Assessment — RS-driven + extraction-aware scoring
    # ------------------------------------------------------------------
    def overall_assessment(self) -> dict:
        for m in ["lsb", "chi_square", "rs_analysis", "histogram", "dct"]:
            if m not in self.results:
                getattr(self, f"{m}_analysis")()

        # Run message extraction as a detection signal
        if not hasattr(self, "_extraction_cache"):
            self._extraction_cache = self.extract_lsb_message(max_chars=300)
        ext = self._extraction_cache
        best_ratio = ext["best_result"]["printable_ratio"]
        likely_msg = ext["best_result"]["likely_message"]

        det = {
            "LSB Analysis":        self.results["lsb"]["detected"],
            "Chi-Square Test":     self.results["chi_square"]["detected"],
            "RS Analysis":         self.results["rs_analysis"]["detected"],
            "Histogram Analysis":  self.results["histogram"]["detected"],
            "DCT Analysis":        self.results["dct"]["detected"],
            "Message Extraction":  likely_msg,
        }

        rs_d_rm = self.results["rs_analysis"]["avg_d_rm"]
        lsb_corr = self.results["lsb"]["avg_correlation"]
        lsb_det = self.results["lsb"]["detected"]

        # ── RS continuous score (primary) ────────────────────────────
        # d_rm noise floor = -0.006, full payload = -0.145
        rs_noise_floor = -0.006
        rs_full = -0.145
        rs_score = float(np.clip(
            (rs_d_rm - rs_noise_floor) / (rs_full - rs_noise_floor),
            0.0, 1.0))

        # ── Extraction score (catches small payloads RS misses) ──────
        # printable_ratio on clean images: typically 0.25–0.45
        # printable_ratio with readable message: 0.70+
        # Score ramps from 0 at ratio=0.55 to 1.0 at ratio=0.95
        if best_ratio >= 0.70:
            ext_score = float(np.clip((best_ratio - 0.55) / 0.40, 0.0, 1.0))
        else:
            ext_score = 0.0

        # ── Support signals (require RS or extraction to be non-zero) ─
        rs_sees_signal = rs_d_rm < -0.012
        ext_sees_signal = ext_score > 0.0
        support = 0
        if rs_sees_signal or ext_sees_signal:
            if det["Chi-Square Test"]:
                support += 1
            if det["Histogram Analysis"]:
                support += 1
            if lsb_det and lsb_corr < 0.008:
                support += 1

        # ── Final risk score ─────────────────────────────────────────
        # RS and extraction together, support tests amplify
        combined_primary = float(np.clip(
            max(rs_score, ext_score * 0.80),   # extraction slightly discounted
            0.0, 1.0))

        if combined_primary > 0.0:
            risk_score = float(np.clip(
                combined_primary * 0.75 + support * 0.083,
                0.0, 1.0))
        else:
            # Neither RS nor extraction fired — need Chi AND Histogram
            both = det["Chi-Square Test"] and det["Histogram Analysis"]
            risk_score = 0.25 if both else 0.0

        # ── Risk level ───────────────────────────────────────────────
        if risk_score >= 0.35:
            risk_level = "HIGH"
            verdict = "Steganographic content LIKELY present"
        elif risk_score >= 0.10:
            risk_level = "MEDIUM"
            verdict = "Steganographic content POSSIBLY present"
        else:
            risk_level = "LOW"
            verdict = "No steganographic content detected"

        return {
            "risk_score":            risk_score,
            "risk_level":            risk_level,
            "verdict":               verdict,
            "detections":            det,
            "positive_count":        sum(det.values()),
            "total_tests":           len(det),
            "estimated_payload_pct": self.results["rs_analysis"]["average_payload_estimate"] * 100,
            "extraction_ratio":      best_ratio,
            "image_size":            self.image.size,
            "image_mode":            self.image.mode,
            "image_format":          self.image.format or "Unknown",
        }

    def run_all(self) -> dict:
        self.lsb_analysis()
        self.chi_square_analysis()
        self.rs_analysis()
        self.histogram_analysis()
        self.dct_analysis()
        return self.overall_assessment()

    # ------------------------------------------------------------------
    # 7. LSB Message Extraction
    # ------------------------------------------------------------------
    def extract_lsb_message(self, max_chars: int = 500) -> dict:
        def bits_to_text(bits):
            chars = []
            for i in range(0, len(bits) - 7, 8):
                bv = int(''.join(str(b) for b in bits[i:i+8]), 2)
                if bv == 0:
                    break
                chars.append(chr(bv) if 32 <= bv <= 126 else '.')
                if len(chars) >= max_chars:
                    break
            text = ''.join(chars)
            pr = sum(1 for c in text if c != '.') / max(1, len(text))
            return {"text": text, "length": len(text),
                    "printable_ratio": round(pr, 3),
                    "likely_message":  pr > 0.7 and len(text) > 8}

        max_bits = max_chars * 8 + 16
        fr = self.np_image[:, :, 0].flatten()
        fg = self.np_image[:, :, 1].flatten()
        fb = self.np_image[:, :, 2].flatten()
        results = {}
        for ch_name, ch_flat in [("Red", fr), ("Green", fg), ("Blue", fb)]:
            results[ch_name] = bits_to_text(
                [int(v) & 1 for v in ch_flat[:max_bits]])
        il = []
        for i in range(min(len(fr), max_bits // 3)):
            il += [int(fr[i]) & 1, int(fg[i]) & 1, int(fb[i]) & 1]
        results["RGB_interleaved"] = bits_to_text(il)
        best_label, best_data = max(
            results.items(), key=lambda x: x[1]["printable_ratio"])
        return {"channels": results, "best_channel": best_label,
                "best_result": best_data, "extraction_attempted": True}
