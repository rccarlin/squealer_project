from reactpy import component, use_state, html, run



@component
def TonePlayer():
    is_playing, set_is_playing = use_state(False)
    frequency, set_frequency = use_state(440)  # Default frequency is A4

    def toggle_tone():
        if is_playing:
            # Stop the tone
            stop_tone_js()
        else:
            # Start the tone
            play_tone_js(frequency)
        set_is_playing(not is_playing)

    def change_frequency(new_frequency):
        set_frequency(new_frequency)
        if is_playing:
            # Update the frequency while playing
            update_frequency_js(new_frequency)

    # JavaScript for playing, stopping, and updating the tone
    js_code = """
    let oscillator;
    let audioContext;

    function playTone(frequency) {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        oscillator = audioContext.createOscillator();
        oscillator.type = "sine";
        oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime);
        oscillator.connect(audioContext.destination);
        oscillator.start();
    }

    function stopTone() {
        if (oscillator) {
            oscillator.stop();
            oscillator.disconnect();
            oscillator = null;
        }
    }

    function updateFrequency(frequency) {
        if (oscillator) {
            oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime);
        }
    }
    """

    return html.div(
        html.script(js_code),  # Inject JavaScript into the page
        html.h1("Tone Player"),
        html.button({"onClick": toggle_tone}, "Toggle Tone"),
        html.div(
            html.button({"onClick": lambda event: change_frequency(880)}, "Set Frequency to 880 Hz"),
            html.button({"onClick": lambda event: change_frequency(330)}, "Set Frequency to 330 Hz"),
        ),
        html.p(f"Current frequency: {frequency} Hz"),
    )


run(TonePlayer)
