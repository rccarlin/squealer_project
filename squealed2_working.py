import reactpy
import plotly
import plotly.io as pio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from reactpy.svg import marker
from scipy import stats
# import sounddevice as sd
import json
from reactpy import component, html, run, use_ref, use_effect, hooks
from reactpy.backend.fastapi import configure
import uvicorn
from fastapi.responses import JSONResponse
import plotly.graph_objs as go
from contextlib import asynccontextmanager
import sys




global data
# i would like to be coding mostly in python as I'm much more comfortable with it, so can I only use javascript for the noises?

app = FastAPI()
# fixme make graph bigger, make it a fixed range/ domain

# calculate the line, return yhats and loss
def line_fit(points):
    x = points[0:-1][0]
    y = points[-1]

    # this is currently just linear regression, but could potentially change this to be more fancy?
    # want to put extra weight on the two handlebars
    weights = np.ones_like(x)
    # make the line really want to be with handlebars
    weights[-1] = 100  # maybe change this if too crazy
    weights[-2] = 100
    # or maybe do legrange to get it through your points...
    coef = np.polyfit(x, y, 1, w=weights)  # can change degree to be bigger to fit fancier...

    best_fit_line = coef[0] * np.array(x) + coef[1]
    resid_squared = (y - best_fit_line) ** 2

    return best_fit_line, resid_squared, coef


def make_plot(data):
    # assumes that the handlebars are added to the bottom

    x = data[0:-1][0]  # fixme make tuples!!!
    y = data[-1]

    best_fit_line, resid_squared, _ = line_fit(data)
    # plotting what we were given...
    scatter_trace = plotly.graph_objects.Scatter(x=x[0:-2], y=y[0:-2], mode='markers',marker=dict(color= resid_squared, colorscale="Viridis", colorbar=dict(title= "Residual Squared", x=1.1, y=.5, len=.5)), name='Data Points')
    best_fit_trace = plotly.graph_objects.Scatter(x=x[0:-2], y=best_fit_line, mode='lines', name='Best Fit Line')

    # the handlebars were added in main and are the last two points
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

def make_line_chart(points):
    temp = [go.Scatter(x=points[1], y=points[0], mode="markers")]
    # temp = [go.Scatter(x=[1, 2, 3], y=[4, 1, 2], mode="markers")]
    layout = go.Layout(title="Coefficients Tried", xaxis_title="Intercept", yaxis_title="Slope")
    fig = go.Figure(data=temp, layout=layout)
    # return pio.to_html(fig, full_html=False)
    return json.loads(pio.to_json(fig))


@component
def LineChart():
    line_chart_html = make_line_chart()
    return html.div(dangerously_set_inner_html=line_chart_html)

@component
def InteractiveGraph():
    global data
    points, set_points = hooks.use_state(data)
    pitch, set_pitch = hooks.use_state(440)
    graph_json = make_plot(points)
    best_fit_line, resid_squared, coef = line_fit(data)
    try_list = [[coef[0]], [coef[1]]]
    tries, set_tries = hooks.use_state(try_list)
    graph_temp = make_line_chart(tries)

    script= f'''
            function renderPlot() {{
                console.log("Rendering Plotly graph...");
                
                // graph 1
                let graphDiv1 = document.getElementById('plot1');
                if (graphDiv1) {{
                    let plotData1 = {json.dumps(graph_json['data'])};
                    let plotLayout1 = {json.dumps(graph_json['layout'])};
                    Plotly.newPlot(graphDiv1, plotData1, plotLayout1);

                }} else {{
                    console.error("Plot 1 div not found");
                }}
                
                // Graph 2
                let graphDiv2 = document.getElementById('plot2');
                if (graphDiv2) {{
                    let plotData2 = {json.dumps(graph_temp['data'])};
                    let plotLayout2 = {json.dumps(graph_temp['layout'])};
                    Plotly.newPlot(graphDiv2, plotData2, plotLayout2);

                }} else {{
                    console.error("Plot 2 div not found");
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


    def update_point(all_points, point, dx, dy, freq):
        new_points = all_points[:]
        # new_points[-point]["x"] += dx
        # new_points[-point]["y"] += dy

        new_points[0][-point] += dx
        new_points[1][-point] += dy

        # okay what pitch should we do?
        line, resid, coef = line_fit(new_points)
        temp = resid.sum()
        if temp > 1000:  # figure out what the max error should be?
            temp = 1000
        new_pitch = temp * 1200 / 5000 + 300

        return new_points, new_pitch, coef, line

    def redraw(all_points):
        try:
            new_plot = make_plot(all_points)
            js_code = f"""
                    Plotly.react('plot1', {new_plot['data']}, {new_plot['layout']});
                    """
            return html.script({"type": "text/javascript"}, js_code)
        except WebSocketDisconnect:
            print("disconnect caught in redraw")

    # Handle keypress events
    def handle_key_down(event):
        try:
            # print(f"Key pressed: {event['key']}")
            nonlocal pitch

            # 1 so I can use -1 to get the 20th percentile, etc
            delta = .1  # fixme, customize, always be positive

            pressed = False
            point = 0
            dx = 0
            dy = 0
            freq = pitch

            if event["key"] == "ArrowUp":
                point = 2
                dy = delta
                pressed = True
            elif event["key"] == "ArrowDown":
                point = 2
                dy = -delta
                pressed = True
            elif event["key"] == "ArrowLeft":
                point = 2
                dx = -delta
                pressed = True
            elif event["key"] == "ArrowRight":
                point = 2
                dx = delta
                pressed = True
            elif event["key"] == "w":
                point = 1
                dy = delta
                pressed = True
            elif event["key"] == "a":
                point = 1
                dx = -delta
                pressed = True
            elif event["key"] == "s":
                point = 1
                dy = -delta
                pressed = True
            elif event["key"] == "d":
                point = 1
                dx = delta
                pressed = True
            # else:
            #     new_points = points
            #     new_pitch = pitch
            # print("new pitch:", new_pitch, "curr:", pitch)

            # set_points(new_points)  # is this what's bugging

            if pressed:
                new_points, new_pitch, coef, line  = update_point(points, point, dx, dy, freq)
                set_pitch(new_pitch)

        except WebSocketDisconnect as e:
            print("closed window caught by handler for reason {e.reason}")

    hooks.use_effect(lambda: redraw(points), [points])

    def play_tone(loss):
        return html.script(
            f"""
                (function() {{
                    const audio = new AudioContext();
                    const oscillator = audio.createOscillator();
                    oscillator.frequency.value = {loss};
                    oscillator.connect(audio.destination);
                    oscillator.start();
                    setTimeout(() => oscillator.stop(), 200);
                }})();
                """)

    def handle_focus(event):
        # Log when the div is focused
        print("Div is focused and ready to capture key presses.")

    # tone_test = play_tone(500)
    return html.div(
        [
            html.div({"id": "plot1", "style": {"width": "600px", "height": "400px"}}),
            html.div({"id": "plot2", "style": {"width": "600px", "height": "400px"}}),
            html.script(script),  # JavaScript to load Plotly and render the chart
            html.div({
                "onFocus": handle_focus,  # Log when focused
                "tabIndex": 0,  # Makes the div focusable to receive key events
                "onKeyDown": handle_key_down,  # Attach the keydown event listener
                "style": {"border": "1px solid black", "padding": "10px", "width": "300px"}
                #"ref": ref
                    }, "Click here to focus and press WASD or Arrow keys."),
            play_tone(pitch)
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

# @app.websocket("/disconnect-endpoint")
# async def disconnect_endpoint(websocket: WebSocket):
#     out = await websocket.receive_text()
#     reason = json.loads(out).get("reason", "Unknown reason")
#     print(f"Client disconnected with reason: {reason}")
#     # Handle the disconnection logic
#     await websocket.close()

@app.post("/disconnect-endpoint")
async def disconnect_endpoint(request: Request):
    try:
        out = await request.json()
        reason = out.get("reason", "Unknown reason")
        print(f"Client disconnected with reason: {reason}")
        # Handle any additional logic here, like cleanup or logging
        return JSONResponse(status_code=200, content={"message": "Disconnect handled"})
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid request data")


configure(app, InteractiveGraph)

# runs the program
def main():
    # replace this with taking in data irl
    x = np.random.uniform(0, 10, 20)
    y = 2 * x + np.random.uniform(-3, 3, 20)

    # fixme another model, logistic

    # want to find initial handlebars
    slope, intercept, _, _, _ = stats.linregress(x, y)  # replace with target model

    top = np.percentile(x, 80)
    bottom = np.percentile(x, 20)
    x = np.append(x, [top, bottom])
    y = np.append(y, [slope * top + intercept, slope * bottom + intercept])

    global data
    data = np.array([x, y])
    # data = zip(x, y)



    uvicorn.run(app, host="127.0.0.1", port=8000)
    # run(InteractiveGraph)

    # Run the application
    # run(GraphWithSoundControl)


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

