from reactpy import component, html, hooks
from reactpy.backend.fastapi import configure
from fastapi import FastAPI
import math
import uvicorn

global data

# ReactPy Component
@component
def InteractiveGraph():
    global data
    freq, set_freq = hooks.use_state(440)  # Default tone frequency in Hz
    points, set_points = hooks.use_state(data)  # Graph points
    audio_context_started, set_audio_context_started = hooks.use_state(False)  # Track if AudioContext is started

    def redraw_graph(change_x, change_y):
        def update():
            current_points = points
            new_points = [(x + change_x, y + change_y) for x, y in current_points]
            set_points(new_points)  # Update points state
            new_freq = max(220, min(880, freq + 20 * change_y))  # Update tone frequency
            set_freq(new_freq)
            return new_freq

        return update

    # Key press handler
    def handle_keypress(event):
        if not audio_context_started:
            return  # Prevent keypress updates if audio context isn't started
        key = event["key"]
        if key == "ArrowUp":
            redraw_graph(0, 1)()
        elif key == "ArrowDown":
            redraw_graph(0, -1)()
        elif key == "ArrowLeft":
            redraw_graph(-1, 0)()
        elif key == "ArrowRight":
            redraw_graph(1, 0)()

    # Start/Stop audio context
    def toggle_audio_context(event=None):
        if not audio_context_started:
            set_audio_context_started(True)
        else:
            html.script("window.stopAudio();")
            set_audio_context_started(False)

    # Inject script using use_effect
    hooks.use_effect(
        lambda: html.script("""
            if (!window.audioCtx) {
                window.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                window.oscillator = null;
                console.log("AudioContext created");
            }
            window.startAudio = function(freq) {
                if (window.oscillator) {
                    window.oscillator.stop();
                    console.log("Oscillator stopped");
                }
                window.oscillator = window.audioCtx.createOscillator();
                window.oscillator.type = 'sine';
                window.oscillator.frequency.setValueAtTime(freq, window.audioCtx.currentTime);
                window.oscillator.connect(window.audioCtx.destination);
                window.oscillator.start();
                console.log("Oscillator started with frequency:", freq);
            };
            window.stopAudio = function() {
                if (window.oscillator) {
                    window.oscillator.stop();
                    console.log("Oscillator stopped");
                }
            };
        """),
        []  # Empty dependency list ensures this runs once on component mount
    )


    # Render
    return html.div(
        {
            "tabIndex": 0,
            "onKeyDown": handle_keypress,
            "style": {"outline": "none"},
        },
        html.button(
            {"onClick": toggle_audio_context},
            "Start Audio" if not audio_context_started else "Stop Audio",
        ),
        html.script(f"if (window.startAudio && {audio_context_started}) window.startAudio({freq});"),
        html.canvas(
            {
                "id": "graph",
                "width": 400,
                "height": 200,
                "style": {"border": "1px solid black"},
            },
            html.script(
                f"""
                const canvas = document.getElementById('graph');
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.beginPath();
                {''.join(f'ctx.lineTo({20 * x}, {100 - 20 * y});' for x, y in points)}
                ctx.stroke();
                """
            ),
        ),
    )


def main():
    # Prepare data
    global data
    data = [(x, math.sin(x)) for x in range(0, 10)]

    # Create FastAPI app and integrate ReactPy
    app = FastAPI()
    configure(app, InteractiveGraph)

    # Run the server programmatically
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
