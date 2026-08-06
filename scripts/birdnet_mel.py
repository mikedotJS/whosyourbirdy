"""BirdNET's ``MelSpecLayerSimple``, transcribed and then algebraically folded.

Background
----------
BirdNET v2.4 computes its two mel spectrograms *inside* the model graph. The layer
is not serialised with its code, so loading the official SavedModel requires us to
supply the class. The transcription below is taken from the reference
implementation the BirdNET team ships with the model itself
(``BirdNET_GLOBAL_6K_V2.4_Model_TFJS/static/main.js``), not reverse-engineered:

    input  = input - min(input)
    input  = input / (max(input) + 1e-6)
    input  = (input - 0.5) * 2
    spec   = tf.signal.stft(input, frame_length, frame_step, frame_length, hann)
    spec   = cast(spec, float32)            # real part, NOT the magnitude
    spec   = spec @ mel_filterbank
    spec   = spec ** 2
    spec   = spec ** (1 / (1 + exp(magnitude_scaling)))
    spec   = reverse(spec, -1); transpose; expand_dims(-1)

Note the ``cast`` from complex to float: BirdNET feeds the **real part** of the
STFT into the mel filterbank, not the magnitude. That is unusual, it is what the
weights were trained against, and it is also why the graph cannot be converted by
the stock tf2onnx RFFT handler, which only knows the ``RFFT2D -> ComplexAbs``
(magnitude) pattern.

Why the fold is exact
---------------------
Three consecutive operations are linear in the framed signal:

    windowing        w[n]
    real DFT         Re(F)[t,k] = sum_n x[t,n] * cos(2*pi*n*k/N)
    mel filterbank   mel[t,m]   = sum_k Re(F)[t,k] * MB[k,m]

Composing them gives a single matrix, independent of the input:

    mel[t,m] = sum_n frames[t,n] * ( w[n] * sum_k cos(2*pi*n*k/N) * MB[k,m] )
             = sum_n frames[t,n] * folded[n,m]

and "slide a fixed matrix over a signal with a stride" is exactly a strided 1-D
convolution. So the entire STFT + mel front-end collapses into one ``Conv1D``
whose kernel is a constant derived from the model's own hann window and its own
mel filterbank. Nothing is re-derived, re-tuned, or approximated: the mel filter
banks, the frame sizes and the magnitude scaling all come from the official
serialised config and checkpoint.

This is not a reimplementation of the mel spectrogram; it is the same linear
operator, materialised. ``FOLD_VS_STFT`` in ``convert_to_onnx.py`` asserts the two
forms agree inside TensorFlow, and the parity harness then checks the exported
ONNX against the official TFLite interpreter end to end.

The fold is also what makes the model usable in a browser: it removes the FFT
(unsupported by the WASM execution provider) and replaces ~1 GFLOP of dense DFT
with a 96-channel convolution.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def folded_kernel(
    mel_filterbank: np.ndarray, frame_length: int
) -> np.ndarray:
    """Fold hann window, real-DFT basis and mel filterbank into one Conv1D kernel.

    Returns an array of shape ``[frame_length, 1, n_mel]`` laid out for
    ``tf.nn.conv1d`` (``[filter_width, in_channels, out_channels]``).

    Computed in float64 and cast once at the end, so the stored kernel is as close
    to the exact operator as float32 allows.
    """
    mb = np.asarray(mel_filterbank, dtype=np.float64)
    n_bins, n_mel = mb.shape
    expected_bins = frame_length // 2 + 1
    if n_bins != expected_bins:
        raise ValueError(
            f"mel filterbank has {n_bins} bins but frame_length={frame_length} "
            f"implies {expected_bins}"
        )

    # Same window tf.signal.stft uses by default (periodic hann).
    window = tf.signal.hann_window(frame_length, periodic=True).numpy().astype(np.float64)

    n = np.arange(frame_length, dtype=np.float64)[:, None]      # [N, 1]
    k = np.arange(n_bins, dtype=np.float64)[None, :]            # [1, K]
    cos_basis = np.cos(2.0 * np.pi * n * k / frame_length)      # [N, K] = Re(DFT)

    folded = (cos_basis @ mb) * window[:, None]                 # [N, n_mel]
    return folded.astype(np.float32)[:, None, :]                # [N, 1, n_mel]


class MelSpecLayerSimple(tf.keras.layers.Layer):
    """The official layer, in either its literal (``stft``) or folded (``conv``) form.

    Both modes are numerically equivalent; ``stft`` exists so the equivalence can
    be asserted rather than asserted-by-hand, and ``conv`` is what gets exported.
    """

    # Set on the class before load_model() so every instance picks it up; the
    # serialised config comes from BirdNET and must not be edited to carry a mode.
    export_mode = "conv"

    def __init__(
        self,
        sample_rate: int = 48000,
        spec_shape: tuple[int, int] = (96, 511),
        frame_step: int = 278,
        frame_length: int = 2048,
        fmin: int = 0,
        fmax: int = 15000,
        data_format: str = "channels_last",
        mel_filterbank: list | np.ndarray | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.spec_shape = tuple(spec_shape)
        self.frame_step = frame_step
        self.frame_length = frame_length
        self.fmin = fmin
        self.fmax = fmax
        self.data_format = data_format
        if mel_filterbank is None:
            raise ValueError("mel_filterbank missing from the serialised layer config")
        self.mel_filterbank = np.asarray(mel_filterbank, dtype=np.float32)
        self.mode = type(self).export_mode

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            sample_rate=self.sample_rate,
            spec_shape=list(self.spec_shape),
            frame_step=self.frame_step,
            frame_length=self.frame_length,
            fmin=self.fmin,
            fmax=self.fmax,
            data_format=self.data_format,
            mel_filterbank=self.mel_filterbank.tolist(),
        )
        return config

    def build(self, input_shape) -> None:
        # Name and initial value must match the checkpoint, or the restored
        # magnitude scaling silently falls back to 1.23.
        self.mag_scale = self.add_weight(
            name="magnitude_scaling",
            shape=(),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(1.23),
            trainable=True,
        )
        self._kernel = tf.constant(
            folded_kernel(self.mel_filterbank, self.frame_length), dtype=tf.float32
        )
        self._melbank = tf.constant(self.mel_filterbank, dtype=tf.float32)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        n_frames = 1 + (int(input_shape[-1]) - self.frame_length) // self.frame_step
        return (input_shape[0], self.mel_filterbank.shape[1], n_frames, 1)

    @staticmethod
    def _normalize(x: tf.Tensor) -> tf.Tensor:
        """Per-window min-max to [-1, 1]. Part of the model, not of our pipeline.

        This is why callers must NOT peak- or RMS-normalise the audio beforehand:
        the model already does it, per 3-second window.
        """
        x = x - tf.reduce_min(x, axis=-1, keepdims=True)
        x = x / (tf.reduce_max(x, axis=-1, keepdims=True) + 1e-6)
        x = x - 0.5
        return x * 2.0

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self._normalize(inputs)

        if self.mode == "conv":
            # Framing + windowing + real-DFT + mel, as one strided convolution.
            spec = tf.nn.conv1d(
                tf.expand_dims(x, axis=-1),          # [B, T, 1]
                self._kernel,                        # [frame_length, 1, n_mel]
                stride=self.frame_step,
                padding="VALID",
            )                                        # [B, n_frames, n_mel]
        elif self.mode == "stft":
            spec = tf.signal.stft(
                x,
                frame_length=self.frame_length,
                frame_step=self.frame_step,
                fft_length=self.frame_length,
                window_fn=tf.signal.hann_window,
            )
            # complex -> float keeps the real part; this is deliberate in BirdNET.
            spec = tf.cast(tf.math.real(spec), tf.float32)
            spec = tf.matmul(spec, self._melbank)
        else:
            raise ValueError(f"unknown mode {self.mode!r}")

        spec = spec * spec
        spec = tf.pow(spec, 1.0 / (1.0 + tf.exp(self.mag_scale)))
        spec = tf.reverse(spec, axis=[-1])
        spec = tf.transpose(spec, perm=[0, 2, 1])
        return tf.expand_dims(spec, axis=-1)
