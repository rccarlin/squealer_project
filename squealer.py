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
import math
from sklearn.linear_model import LogisticRegression
import pandas as pd




global data
global model
global og_fit_line

app = FastAPI()

# calculate the line, return (some sense of) loss, and the coefficients needed to make the line
def line_fit(points):
    x = points[0:-1]
    y = points[-1]

    # okay so the handlebars have already been fit to the line, so can't we just extract those points to make a line and then calculate loss...

    # if model is linear, the handlebars are in x[0] and y
    # and if it's logistic, it's in x[0] and x[1], I believe...

    # get the handlebars so we can draw a line between them
    start = (0,0)
    stop = (0,0)
    global model  # fixme, if change this to a state later, pass that in
    # I actually think we can avoid this by doing points[0], points[1], and points[2], no?
    if model == "linear":
        start = (x[0][-1], y[-1])
        stop = (x[0][-2], y[-2])
    elif model == "logistic":
        start = (x[0][-1], x[1][-1])
        stop = (x[0][-2], x[1][-2])

    # now we have a line
    slope = (stop[1] - start[1]) / (stop[0] - start[0])
    intercept = start[1] - slope * start[0]

    coef = [slope, intercept]  # his is based off of the np.polyfit coef returns, where intercept is last

    # now to calculate the error/ loss.
    # for linear regression, this is the residuals, but for logistic, I want to calculate the uhhhhhh

    resids = 0
    best_fit_line = list()
    if model == "linear":
        best_fit_line = coef[0] * np.array(x[0]) + coef[1]  # fixme, now that x != points[0:-1][0], I think I need that extra [0]... we'll see
        resids = y - best_fit_line
    elif model == "logistic":
        best_fit_line = coef[0] * x[0] + coef[1]
        resids = y * (x[1][0:-2] - coef[0] * x[0][0:-2] - coef[1]) / (coef[0] ** 2 + 1) ** .5  # this is actually margins


    # old code...
    # # this is currently just linear regression, but could potentially change this to be more fancy?
    # # want to put extra weight on the two handlebars
    # weights = np.ones_like(x)
    # # make the line really want to be with handlebars
    # weights[-1] = 100  # maybe change this if too crazy
    # weights[-2] = 100
    # # or maybe do legrange to get it through your points...
    # coef = np.polyfit(x, y, 1, w=weights)  # can change degree to be bigger to fit fancier...
    #
    # best_fit_line = coef[0] * np.array(x) + coef[1]
    # # resid_squared = (y - best_fit_line) ** 2
    # resids = y - best_fit_line

    return best_fit_line, resids, coef


def make_plot(data):
    # assumes that the handlebars are added to the bottom

    x = data[0:-1]  # when this was just linear regression add [0]
    y = data[-1]

    best_fit_line, resids, coef = line_fit(data)
    x_handlebars = []
    y_handlebars = []
    symbol_map = {-1: "circle", 1: "cross"}


    if model == "linear":
        color = resids**2
        # plotting what we were given...
        scatter_trace = plotly.graph_objects.Scatter(x=x[0][0:-2], y=y[0:-2], mode='markers',
                                                     marker=dict(color=color, colorscale="portland",
                                                                 colorbar=dict(title="Residual Squared", x=1.1, y=.5,
                                                                               len=.5)), name='Data Points')
        best_fit_trace = plotly.graph_objects.Scatter(x=x[0][0:-2], y=best_fit_line, mode='lines', name='Best Fit Line')
        x_handlebars = [x[0][-1], x[0][-2]]
        y_handlebars = [y[-1], y[-2]]

        # # plotting the handlebars
        # points_on_line_trace = plotly.graph_objects.Scatter(
        #     x=x_handlebars, y=y_handlebars,
        #     mode='markers', name='Handlebars',
        #     marker=dict(color='red', size=10)  # Customizing color and size
        # )
        #
        # fig = plotly.graph_objects.Figure(data=[scatter_trace, best_fit_trace, points_on_line_trace])

    elif model == "logistic":
        color = resids  # because this is margins... hopefully
        # plotting what we were given...

        # want y labels to be conveyed by shapes

        class_1 = [x[0][0:-2][y == -1], x[1][0:-2][y == -1]]
        class_2 = [x[0][0:-2][y == 1], x[1][0:-2][y == 1]]

        # class_1_points = plotly.graph_objects.Scatter(x=class_1[0], y=class_1[1], mode='markers',
        #                                              marker=dict(symbol="circle", color=color, colorscale="portland",
        #                                                          colorbar=dict(title="Margins", x=1.1, y=.5,
        #                                                                        len=.5)), name='Class 1')
        #
        # class_2_points = plotly.graph_objects.Scatter(x=class_2[0], y=class_2[1], mode='markers',
        #                                               marker=dict(symbol="cross", color=color, colorscale="portland",
        #                                                           colorbar=dict(title="Margins", x=1.1, y=.5,
        #                                                                         len=.5)), name='Class 2')

        scatter_trace = plotly.graph_objects.Scatter(x=x[0][0:-2], y=x[1][0:-2], mode='markers',
                                                     marker=dict(symbol=np.vectorize(symbol_map.get)(y), color=color, colorscale="portland",
                                                                 colorbar=dict(title="Margins", x=1.1, y=.5,
                                                                               len=.5)), name = "w", showlegend=False)

        x_handlebars = [x[0][-1], x[0][-2]]
        y_handlebars = [x[1][-1], x[1][-2]]


    # both types of models need the original line and a current line
    global og_fit_line
    og_fit_trace = plotly.graph_objects.Scatter(x=x[0][0:-2], y=og_fit_line, mode='lines', name='Original Fit Line')
    curr_fit_trace = plotly.graph_objects.Scatter(x=x[0][0:-2], y=best_fit_line, mode='lines', name='Current Fit Line')
    # plotting the handlebars
    points_on_line_trace = plotly.graph_objects.Scatter(
        x=x_handlebars, y=y_handlebars,
        mode='markers', name='Handlebars',
        marker=dict(color='red', size=10)  # Customizing color and size
    )

    # fig = plotly.graph_objects.Figure(data=[class_1_points, class_2_points, best_fit_trace, points_on_line_trace])
    fig = plotly.graph_objects.Figure(data=[scatter_trace, curr_fit_trace, og_fit_trace, points_on_line_trace])
    fig.update_layout(title="Data")

    if model == "logistic":
        for label in symbol_map:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],  # No data points, just for the legend
                mode='markers',
                marker=dict(symbol=symbol_map[label], color='red'),  # Transparent color
                name=f"Data with Y = {label}",  # Name in the legend
            ))

    return json.loads(pio.to_json(fig))


def make_prog_chart(coef, likelihood):
    size_list = np.ones(len(likelihood)) * 10
    size_list[-1] = 25
    temp = [go.Scatter(x=coef[1], y=coef[0], mode="lines+markers", marker=dict(size= size_list, color=likelihood, colorscale="Viridis", colorbar=dict(title= "Approx Log Likelihood", x=1.1, y=.5, len=.5)), name="Tries")]
    layout = go.Layout(title="Coefficients Tried (Current Try Larger)", xaxis_title="Intercept", yaxis_title="Slope")
    fig = go.Figure(data=temp, layout=layout)
    return json.loads(pio.to_json(fig))

def log_likelihood(resids, y = None):
    global model
    if model == "linear":
        resid_squared = resids**2
        rss = resid_squared.sum()
        resid_var = np.var(resids, ddof=2)
        n = len(resids)
        return -n / 2 * np.log(2 * math.pi * resid_var) - rss / (2 * resid_var)
    elif model == "logistic":
        return np.sum(-np.log(1 + np.exp(-resids)))


# @component
# def LineChart():
#     line_chart_html = make_line_chart()
#     return html.div(dangerously_set_inner_html=line_chart_html)

@component
def InteractiveGraph():
    global data
    points, set_points = hooks.use_state(data)  # data
    pitch, set_pitch = hooks.use_state(440)  # tone
    graph_json = make_plot(points)
    best_fit_line, resid, coef = line_fit(data)

    # in addition to graphing the data, we will also keep track of the lines tried so far
    try_list = [[coef[0]], [coef[1]]]
    tries, set_tries = hooks.use_state(try_list)
    likelihood_list = [log_likelihood(resid, data[-1])]  # added the data[-1] for the labels on logistic regression, is unused for linear
    likelihood, set_likelihood = hooks.use_state(likelihood_list)
    graph_temp = make_prog_chart(tries, likelihood)

    # adjust stepsize
    step, set_step = hooks.use_state(.1)

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


    def update_point(all_points, point, dx, dy):
        # this update is fine for 1d linear or 2d logistic!
        new_points = all_points[:]
        new_points[0][-point] += dx
        new_points[1][-point] += dy

        # okay what pitch should we do?
        line, resids, coef = line_fit(new_points)

        temp = []
        global model
        if model == "linear":
            temp = resids
            maxErr = 3500
        elif model == "logistic":
            temp = resids[resids < 0]  # bad margins
            maxErr = 800  # what is a reasonable error for this?

        temp = temp ** 2  # okay so like this, I'm squaring the bad margins. I probably don't need to...
        temp = temp.sum()

        if temp > maxErr:
            temp = maxErr
        minHz = 300
        maxHz = 1200
        new_pitch =  minHz + (temp / maxErr) * (maxHz - minHz)

        return new_points, new_pitch, coef, resids


    # Handle keypress events
    def handle_key_down(event):
        try:
            # print(f"Key pressed: {event['key']}")
            nonlocal pitch

            # 1 so I can use -1 to get the 20th percentile, etc

            pressed = False
            point = 0
            dx = 0
            dy = 0
            # print(step)
            delta = float(step)

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

            if pressed:
                new_points, new_pitch, coef, resids  = update_point(points, point, dx, dy)

                temp_tries = tries[:]
                temp_tries[0].append(coef[0])
                temp_tries[1].append(coef[1])

                # trying something
                # set_tries(temp_tries)  # update the states
                set_pitch(new_pitch)
                set_likelihood(likelihood + [log_likelihood(resids, new_points[-1])])  # again, only need the labels for logistic, unused for linear

        except WebSocketDisconnect as e:
            print("closed window caught by handler for reason {e.reason}")


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

    def handle_slider_change(event):
        set_step(event["target"]["value"])



    return html.div(
        [
            html.div({"id": "plot1", "style": {"width": "800px", "height": "600px"}}),
            html.script(script),  # JavaScript to load Plotly and render the chart
            html.div({
                "tabIndex": 0,  # Makes the div focusable to receive key events
                "autofocus": True,
                "onKeyDown": handle_key_down,  # Attach the keydown event listener
                "style": {"border": "1px solid black", "padding": "10px", "width": "300px"}
                }, "Use Arrow Keys to adjust the line from the right, and WASD to adjust from the left."),
            html.div(
                {
                    "style": {"display": "flex", "alignItems": "center", "gap": "10px"}
                },
                "Slider: ",
                html.input(
                    {
                        "type": "range",
                        "min": 0,
                        "max": 10,  # fixme, make more dynamic?
                        "value": step,
                        "step": .25,
                        "onInput": handle_slider_change,
                    }),  html.span(f"Value: {step}"),
            ),
            html.div({"id": "plot2", "style": {"width": "600px", "height": "400px"}}),
            play_tone(pitch)
        ]
    )

# @app.post("/disconnect-endpoint")
# async def disconnect_endpoint(request: Request):
#     try:
#         out = await request.json()
#         reason = out.get("reason", "Unknown reason")
#         print(f"Client disconnected with reason: {reason}")
#         # Handle any additional logic here, like cleanup or logging
#         return JSONResponse(status_code=200, content={"message": "Disconnect handled"})
#     except Exception as e:
#         raise HTTPException(status_code=400, detail="Invalid request data")
#

def generate_data():
    global model

    if model == "linear":
        x = np.random.uniform(0, 10, 30)
        y = 2 * x + np.random.uniform(-3, 3, 30)

        # fit line and add handle bars
        coef =  np.polyfit(x, y, 1)
        slope = coef[0]
        intercept = coef[1]

        top = np.percentile(x, 80)
        bottom = np.percentile(x, 20)
        x = [np.append(x, [top, bottom])]
        y = np.append(y, [slope * top + intercept, slope * bottom + intercept])
    elif model == "logistic":
        locations = [np.random.uniform(-3, -1), np.random.uniform(1,3) ]

        rng = np.random.default_rng()
        cluster1 = np.random.normal(loc=[locations[0], locations[0]], size=(15,2)) # for 2d
        label1 = rng.choice(a= np.array([-1, 1]), size=15, p= [.9, .1])


        cluster2 = np.random.normal(loc=[locations[1], locations[1]], size=(15,2)) # for 2d
        label2 = rng.choice(a= [-1, 1], size=15, p= [.2, .8])

        x = np.vstack((cluster1, cluster2))
        y = np.hstack((label1, label2)).ravel()

        # now to fit a line and add handlebars
        mod = LogisticRegression()
        mod.fit(x, y)

        coef = mod.coef_[0]
        intercept = mod.intercept_[0]

        if abs(coef[1]) > abs(coef[0]):
            top_x1 = np.percentile(x[:, 0], 80)
            bottom_x1 = np.percentile(x[:, 0], 20)
            right_x2 = -(coef[0] * top_x1 + intercept) / coef[1]
            left_x2 = -(coef[0] * bottom_x1 + intercept) / coef[1]
            x = np.vstack((x, np.array([top_x1, right_x2])))
            x = np.vstack((x, np.array([bottom_x1, left_x2])))
            # print("1>0")
            # print(top_x1, right_x2)
            # print(bottom_x1, left_x2)
        else:
            # fixme so if coef1 is really small, we should not be dividing by it
            # so now we need to find the line in terms of x2 instead of x1
            # which means I need to be careful about what order I add these points to data
            # to make sure that the point with the larger x1 value is added first...
            top_x2 = np.percentile(x[:, 1], 80)
            bottom_x2 = np.percentile(x[:, 1], 20)
            right_x1 = -(coef[1] * top_x2 + intercept) / coef[0]
            left_x1 = -(coef[1] * bottom_x2 + intercept) / coef[0]

            # print("0>1")
            # print(right_x1, top_x2)
            # print(left_x1, bottom_x2)

            # I need to add the right most x1 first to be consistent with rest of code
            if right_x1 > left_x1:
                x = np.vstack((x, np.array([right_x1, top_x2])))
                x = np.vstack((x, np.array([left_x1, bottom_x2])))
            else:
                x = np.vstack((x, np.array([left_x1, bottom_x2])))
                x = np.vstack((x, np.array([right_x1, top_x2])))

        # # print(top, right_x2, bottom, left_x2)
        # print("coef", mod.coef_)
        # print("intercept", mod.intercept_)

        x = [x[:,0], x[:,1]]


    return x, y



configure(app, InteractiveGraph)

# runs the program
def main():
    # replace this with taking in data irl
    global model
    model = "logistic"  # logistic or linear, eventually make this a button, make this easier to change

    x, y = generate_data()

    global data
    # data = np.array([x, y])  # old code just in case
    # functions assume the data comes in the form of [x1, x2, ..., y]
    data = x
    data.append(y)

    line, _, _ = line_fit(data)
    global og_fit_line
    og_fit_line = line

    uvicorn.run(app, host="127.0.0.1", port=8000)
    # run(InteractiveGraph)

main()


