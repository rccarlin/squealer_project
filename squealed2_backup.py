import reactpy
import plotly
import plotly.io as pio
import numpy as np
from reactpy.svg import marker
from scipy import stats
# import sounddevice as sd
import json
from reactpy import component, html, run, use_ref, use_effect, hooks

global data
# i would like to be coding mostly in python as I'm much more comfortable with it, so can I only use javascript for the noises?


def make_plot():
    # assumes that the handlebars are added to the bottom
    x = data[0:-1][0]  # fixme i think I made this nd proof....
    y = data[-1]

    # this is currently just linear regression, but could potentially change this to be more fancy?
    # want to put extra weight on the two handlebars
    weights = np.ones_like(x)
    # make the line really want to be with handlebars
    weights[-1] = 1000  # maybe change this if too crazy
    weights[-2] = 1000
    # or maybe do legrange to get it through your points...
    coef = np.polyfit(x, y, 1, w=weights)  # can change degree to be bigger to fit fancier...

    # fixme make this more dynamic
    best_fit_line = coef[0] * np.array(x) + coef[1]
    resid_squared = (y - best_fit_line)**2

    # plotting what we were given...
    scatter_trace = plotly.graph_objects.Scatter(x=x[0:-2], y=y[0:-2], mode='markers',marker=dict(color= resid_squared, colorscale="Viridis", colorbar=dict(title= "Residual Squared", x=1.1, y=.5, len=.5)), name='Data Points')
    best_fit_trace = plotly.graph_objects.Scatter(x=x[0:-2], y=best_fit_line, mode='lines', name='Best Fit Line')

    # I want to augment the data with two new points as the handlebars
    # for now these won't do anything, but I want them on the graph
    # x_handlebars = [np.percentile(x, 25), np.percentile(x, 75)]  # You could choose other specific points if needed
    # y_handlebars = [slope * x_handlebars[0] + intercept, slope * x_handlebars[1] + intercept]
    # user-defined handlebars
    x_handlebars = [x[-1], x[-2]]
    y_handlebars = [y[-1], y[-2]]

    # plotting
    points_on_line_trace = plotly.graph_objects.Scatter(
        x=x_handlebars, y=y_handlebars,
        mode='markers', name='Extra Points on Line',
        marker=dict(color='red', size=10)  # Customizing color and size
    )

    fig = plotly.graph_objects.Figure(data=[scatter_trace, best_fit_trace, points_on_line_trace])

    # Add click event handling in Plotly to play a sound when a point is clicked
    fig.update_layout(clickmode='event+select')

    return json.loads(pio.to_json(fig))


@component
def InteractiveGraph():
    # ref = use_ref(None)
    # use_effect(lambda: ref.current.focus(), [])
    graph_json = make_plot()
    # inside? # const plotlyScript = document.createElement('script');
    # plotlyScript.src = 'https://cdn.plot.ly/plotly-latest.min.js';
    # document.head.appendChild(plotlyScript);
    # hooks.use_effect(lambda: inject_key_listener(), [])

    script= f'''
            function renderPlot() {{
                console.log("Rendering Plotly graph...");
                let graphDiv = document.getElementById('plot');
                if (graphDiv) {{
                    let plotData = {json.dumps(graph_json['data'])};
                    let plotLayout = {json.dumps(graph_json['layout'])};
                    Plotly.newPlot(graphDiv, plotData, plotLayout);

                }} else {{
                    console.error("Plot div not found");
                }}
            }}

            // Load Plotly and then render the plot
            var plotlyScript = document.createElement('script');
            plotlyScript.src = 'https://cdn.plot.ly/plotly-latest.min.js';
            plotlyScript.onload = renderPlot;
            document.head.appendChild(plotlyScript);
            
            // Global keydown event listener
            document.addEventListener('keydown', function(event){{console.log('Global key press detected:', event.key);
            // Call Python backend via pywebview or another communication method
            // pywebview.api.handle_keypress(event.key);
        }});'''

    # def inject_key_listener():
    #     # Inject custom JavaScript into the DOM to listen globally for key presses
    #     js_code = """
    #     document.addEventListener('keydown', function(event) {
    #         // Log the key press event to Python's console through ReactPy's backend
    #         pywebview.api.handle_keypress(event.key);
    #     });
    #     """
    #     # This will inject the JavaScript into the DOM
    #     html.script(js_code)

    # hooks.use_effect(lambda:update_point())
    def update_point(point, dx, dy):
        print("")
        global data
        data[0][-point] += dx
        data[1][-point] += dy
        print(data[0][-point], data[1][-point])
        # fixme I think data would make more sense as pandas............

        # redraw the plot, making sure the new line goes through both handlebars
        # make noise corresponding to error/ loss

        # okay sorry about this, smashing two functions together, want to redraw now
        new_plot = make_plot()
        # fixme check to see if this in the same format as the last time u did this (OG)
        # maybe its a name issue?
        js_code = f"""
                Plotly.react('plot', {new_plot['data']}, {new_plot['layout']});  
                """
        return html.script({"type": "text/javascript"}, js_code)

    # let
    # graphDiv = document.getElementById('plot');
    # if (graphDiv) {{
    # let plotData = {json.dumps(graph_json['data'])};
    # let plotLayout = {json.dumps(graph_json['layout'])};
    # Plotly.newPlot(graphDiv, plotData, plotLayout);
    #
    # }} else {{
    # console.error("Plot div not found");
    # }}


    # Handle keypress events
    def handle_key_down(event):
        print(f"Key pressed: {event['key']}")

        # 1 so I can use -1 to get the 20th percentile, etc
        delta = .1  # fixme, customize, always be positive
        if event["key"] == "ArrowUp":
            update_point(2, 0, delta)
        elif event["key"] == "ArrowDown":
            update_point(2, 0, -delta)
        elif event["key"] == "ArrowLeft":
            update_point(2, -delta, 0)
        elif event["key"] == "ArrowRight":
            update_point(2, delta, 0)
        elif event["key"] == "w":
            update_point(1, dx=0, dy=delta)
        elif event["key"] == "a":
            update_point(1, dx=0, dy=-delta)
        elif event["key"] == "s":
            update_point(1, dx=-delta, dy=0)
        elif event["key"] == "d":
            update_point(1, dx=delta, dy=0)

    def handle_focus(event):
        # Log when the div is focused
        print("Div is focused and ready to capture key presses.")


    return html.div(
        [
            html.div({"id": "plot", "style": {"width": "600px", "height": "400px"}}),
            html.script(script),  # JavaScript to load Plotly and render the chart
            html.div({
                "onFocus": handle_focus,  # Log when focused
                "tabIndex": 0,  # Makes the div focusable to receive key events
                "onKeyDown": handle_key_down,  # Attach the keydown event listener
                "style": {"border": "1px solid black", "padding": "10px", "width": "300px"}
                #"ref": ref
                    }, "Click here to focus and press WASD or Arrow keys.")
        ]
    )


@component
def GraphWithSoundControl():
    def on_click(event):
        return html.div(
            [
                html.h1("Adjust Tone Pitch"),
                # html.button({"onclick": lambda _: play_tone()}, "Play Tone"),
                html.button({"id": "playTone"}, "Play Tone"),
                html.input({"type": "range", "id": "pitchSlider", "min": "200", "max": "1000", "step": "50", "value": "440"}),
                html.label({"for": "pitchSlider"}, "Tone Frequency (Pitch)"),
                html.input(
                    {"type": "range", "id": "volumeSlider", "min": "0", "max": "1", "step": "0.1", "value": "0.5"}),
                html.label({"for": "volumeSlider"}, "Volume (0 to 1)"),
                html.script({
                    "src": "https://cdnjs.cloudflare.com/ajax/libs/howler/2.2.4/howler.min.js"
                }),
                html.script("""
                                    window.onload = function() {
                                        document.getElementById('playTone').onclick = function() {
                                            console.log("Play Tone button clicked!");

                                            // Retrieve slider values for pitch and volume
                                            let pitch = document.getElementById('pitchSlider').value;
                                            let volume = document.getElementById('volumeSlider').value;

                                            console.log("Pitch set to: " + pitch);
                                            console.log("Volume set to: " + volume);

                                            // Create a new audio context
                                            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                                            const oscillator = audioContext.createOscillator();
                                            const gainNode = audioContext.createGain();

                                            // Set the oscillator type and frequency
                                            oscillator.type = 'sine';  // You can change to 'square', 'sawtooth', or 'triangle'
                                            oscillator.frequency.setValueAtTime(pitch, audioContext.currentTime); // Set frequency to the slider value

                                            // Set the volume
                                            gainNode.gain.setValueAtTime(volume, audioContext.currentTime);

                                            // Connect the oscillator to the gain node, then to the audio context
                                            oscillator.connect(gainNode);
                                            gainNode.connect(audioContext.destination);

                                            // Start the oscillator
                                            oscillator.start();

                                            // Stop the oscillator after 1 second (you can adjust this)
                                            oscillator.stop(audioContext.currentTime + 1);

                                            console.log("Sound played!");
                                        };
                                    };
                                """),
            ]
        )
    return html.div(on_click(None))


def play_tone():
    return html.script(
        """
        var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        var oscillator = audioCtx.createOscillator();
        var gainNode = audioCtx.createGain(); // Create a gain node for volume control


        // Get slider values
        var frequency = document.getElementById('pitchSlider').value;
        var volume = document.getElementById('volumeSlider').value; // Get volume from the slider

        console.log("Playing tone with frequency: " + frequency + " Hz and volume: " + volume);

        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(frequency, audioCtx.currentTime); // Frequency in Hertz

        gainNode.gain.setValueAtTime(volume, audioCtx.currentTime); // Set gain value

        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);

        oscillator.start();
        // Make sure the tone lasts long enough to hear (2 seconds)
        setTimeout(function() {
            oscillator.stop();
        }, 2000); // Stop after 2 seconds
        """
    )


# Run the application
# run(GraphWithSoundControl)

def main():
    # replace this with taking in data irl
    x = np.random.uniform(0, 10, 20)
    y = 2 * x + np.random.uniform(0, 3, 20)

    # want to find initial handlebars
    slope, intercept, _, _, _ = stats.linregress(x, y)  # replace with target model
    # best_fit_line = slope * np.array(x) + intercept

    top = np.percentile(x, 80)
    bottom = np.percentile(x, 20)
    x = np.append(x, [top, bottom])
    y = np.append(y, [slope * top + intercept, slope * bottom + intercept])

    # print(np.sort(x))
    # print(np.percentile(x, 90), np.percentile(x, 10))
    global data
    data = np.array([x, y])
    # okay, so now the last two elements of data are the handlebars
    # run(InteractiveGraph)
    run(GraphWithSoundControl)


main()


# from interactive graph script before the else (inside the if)
# graphDiv.on('plotly_click', function(data) {{
                    #     if (data.points.length > 0) {{
                    #         const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    #         const oscillator = audioContext.createOscillator();
                    #         const gainNode = audioContext.createGain();
                    #         oscillator.type = 'sine';
                    #
                    #         // to customize as you move the line around
                    #         oscillator.frequency.setValueAtTime(440, audioContext.currentTime); // Set frequency to the slider value
                    #         gainNode.gain.setValueAtTime(.3, audioContext.currentTime)
                    #         oscillator.connect(gainNode);
                    #         gainNode.connect(audioContext.destination);
                    #         oscillator.start();
                    #         oscillator.stop(audioContext.currentTime + 1);  // eventually i don't want it to stop
                    #
                    #     }}
                    # }});

