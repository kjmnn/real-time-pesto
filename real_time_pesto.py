import argparse
import functools

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
import pesto
import pesto.utils.hcqt
import pyaudio
import torch


class PlotWrapper:
    def update(self, time, pitch, confidence, activations):
        raise NotImplementedError


class LinePlotWrapper(PlotWrapper):
    line: plt.Line2D

    def __init__(self, crop_y=True, confidence_threshold=0):
        plt.style.use("Solarize_Light2")
        plt.gca().set_ylim(21, 109)
        self.line = plt.plot([], [])[0]
        self.crop_y = crop_y
        self.confidence_threshold = confidence_threshold

    def update(self, time, pitch, confidence, activations):
        if len(time) == 0:
            return
        if self.crop_y:
            plt.gca().set_ylim(min(pitch) - 1, max(pitch) + 1)
        plt.gca().set_xlim(time[0], time[-1])
        if self.confidence_threshold > 0:
            pitch = torch.where(confidence > self.confidence_threshold, pitch, torch.nan)
        self.line.set_data(time, pitch)


class BitmapPlotWrapper(PlotWrapper):
    bitmap: mpimg.AxesImage

    def __init__(self, predicted_only=False, confidence_threshold=0, cmap="viridis"):
        self.bitmap = plt.imshow([[]], aspect="auto", origin="lower", cmap=cmap)
        plt.tight_layout()
        self.predicted_only = predicted_only
        self.confidence_threshold = confidence_threshold
        bps = 384 // 128
        self.ylims = (21, 109)
        self.act_crop = (21 * bps, 109 * bps)

    def update(self, time, pitch, confidence, activations):
        if len(time) == 0:
            return
        activations = activations[:, self.act_crop[0] : self.act_crop[1]]
        if self.predicted_only:
            activations = activations * confidence[:, None]

        self.bitmap.set_data(activations.T)
        self.bitmap.set_extent((time[0], time[-1]) + self.ylims)
        # plt.gca().set_ylim(min(pitch) - 1, max(pitch) + 1)
        # plt.gca().set_xlim(time[0], time[-1])


def init_plot(type="line", crop_y=True, predicted_only=False, confidence_threshold=0, cmap="viridis"):
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (semitones)")
    plt.tight_layout()
    match type:
        case "line":
            return LinePlotWrapper(crop_y, confidence_threshold)
        case "bitmap":
            return BitmapPlotWrapper(predicted_only, confidence_threshold, cmap)


class Stuff:
    def __init__(
        self,
        model: pesto.core.PESTO,
        plot: PlotWrapper,
        frames_per_buffer: int,
        step_size: float,
        history_length: int,
    ):
        self.model = model
        self.preds = torch.zeros(0, dtype=torch.float)
        self.confidences = torch.zeros(0, dtype=torch.float)
        self.activations = torch.zeros((0, 384), dtype=torch.float)
        self.plot = plot
        self.frames_per_buffer = frames_per_buffer
        self.step_size = step_size / 1000
        self.history_length = history_length

    def pa_callback(self, in_data, _frame_count, _time_info, _status):
        audio = torch.frombuffer(in_data, dtype=torch.int16).type(torch.float32)
        with torch.inference_mode():
            preds, confidences, activations = self.model(audio, return_activations=True)
        self.preds = torch.cat([self.preds, preds], dim=0)
        self.confidences = torch.cat([self.confidences, confidences], dim=0)
        self.activations = torch.cat([self.activations, activations], dim=0)
        return None, pyaudio.paContinue

    def redraw(self, _=None):
        if self.history_length is not None:
            window = min(len(self.preds), self.history_length)
        else:
            window = len(self.preds)
        time = torch.linspace(len(self.preds) - window, len(self.preds), window) * self.step_size
        self.plot.update(
            time.cpu(), self.preds[-window:].cpu(), self.confidences[-window:].cpu(), self.activations[-window:].cpu()
        )


def main(args: argparse.Namespace):
    chunk_length = args.sampling_rate * args.step_size * args.steps_per_chunk // 1000

    if chunk_length <= 65536:
        # Normally, the CQT algorithm uses reflection padding.
        # However, the max kernel size with default settings is 131072,
        # which means that reflection padding can't be used for buffers shorter than 65537.
        match args.short_chunk_workaround:
            case "constant_pad":
                # We can get around this at the cost of some accuracy by using constant padding.
                pesto.utils.hcqt.CQT = functools.partial(pesto.utils.hcqt.CQT, pad_mode="constant")
                model = pesto.load_model("mir-1k", args.step_size, args.sampling_rate)
    else:
        model = pesto.load_model("mir-1k", args.step_size, args.sampling_rate)
    model = functools.partial(model, sr=args.sampling_rate)

    pa = pyaudio.PyAudio()
    plot = init_plot(args.plot_type, args.crop_y, args.predicted_only, args.confidence_threshold, args.cmap)

    stuff = Stuff(
        model,
        plot,
        chunk_length,
        args.step_size,
        args.history_length,
    )

    # Open stream from default input device
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=args.sampling_rate,
        input=True,
        output=False,
        frames_per_buffer=chunk_length,
        stream_callback=stuff.pa_callback,
    )

    # Animate the plot and loop
    ani = animation.FuncAnimation(
        plt.gcf(), stuff.redraw, interval=args.step_size * args.steps_per_chunk, cache_frame_data=False
    )
    plt.show()

    pa.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling_rate", type=int, default=48000, help="Input sampling rate.")
    parser.add_argument("--step_size", type=float, default=50, help="Prediction step size in milliseconds.")
    parser.add_argument(
        "--steps_per_chunk",
        type=int,
        default=2,
        help="Number of steps per chunk (and plot redraw). Lower will result in smoother animation, "
        "but the predictions will be less accurate due to workarounds required for short chunks.",
    )
    parser.add_argument("--history_length", type=int, default=60, help="Number of past steps to show.")
    parser.add_argument(
        "--short_chunk_workaround",
        type=str,
        choices=["constant_pad"],  # I tried different workarounds but they didn't work.
        default="constant_pad",
        help="Workaround for short chunks.",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["line", "bitmap"],
        default="bitmap",
        help='Output type ("line" for line graph, "bitmap" for PESTO CLI style).',
    )
    parser.add_argument(
        "--crop_y", action="store_true", help='Crop y-axis to fit the data (no effect with "--plot_type bitmap")'
    )
    parser.add_argument(
        "--predicted_only",
        action="store_true",
        help='Only show predicted pitches (like the PESTO CLI does; no effect with "--plot_type line")',
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0, help="Confidence threshold for displaying pitch predictions."
    )
    parser.add_argument(
        "--cmap", type=str, default="viridis", help='Color map for bitmap plot (PESTO CLI uses "inferno".)'
    )
    args = parser.parse_args()
    main(args)
