import reactpy
import plotly
import plotly.io as pio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from reactpy.svg import marker
from scipy import stats
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
import random
import pandas as pd


global data
global model
global og_fit_line

app = FastAPI()

# takes in the data (including handle bars), finds the line/ coefficients defined by the handle bars
# also returns the "residuals" which are residuals for linear regression but are signed margins for logistic regression
def line_fit(points):
    x = points[0:-1]
    y = points[-1]

    global model  # fixme, if change this to a state later, pass that in

    # step one: make a line with the handlebars
    # (in previous iterations, I used weights to ensure the line stayed with the handlebars. That is unnecessary for
    # these simple linear examples, but just an idea for when expanding to more complicated models...)
    # if model is linear, the handlebars are in x[0] and y (which is points[1])
    # and if it's logistic, it's in x[0] and x[1] (points[0] and points[1])
    start = (points[0][-1], points[1][-1])
    stop = (points[0][-2], points[1][-2])

    # now we can find the line
    slope = (stop[1] - start[1]) / (stop[0] - start[0])
    intercept = start[1] - slope * start[0]

    coef = [slope, intercept]  # this is based off of the np.polyfit coef returns, where intercept is last

    # step two: calculate fit statistic (residuals or signed margins)
    resids = 0
    fit_line = list()
    if model == "linear":
        fit_line = coef[0] * np.array(x[0]) + coef[1]
        resids = y - fit_line  # residuals
    elif model == "logistic":
        fit_line = coef[0] * x[0] + coef[1]
        resids = y * (x[1][0:-2] - coef[0] * x[0][0:-2] - coef[1]) / (coef[0] ** 2 + 1) ** .5  # signed margins

    return fit_line, resids, coef


# takes in all data (with handle bars at the end) to plot the data points, handle bars, original best fit line, and the
# current line defined by the handlebars
def make_plot(data):
    x = data[0:-1]
    y = data[-1]

    fit_line, resids, _ = line_fit(data)

    # handlebars will be plotted separately so they can stand out
    x_handlebars = [data[0][-1], data[0][-2]]
    y_handlebars = [data[1][-1], data[1][-2]]
    symbol_map = {-1: "circle", 1: "cross"}  # this is needed for plotting the labels for logistic regression but is
    # currently outside of the if statement because it's used again after the figure is made...

    global model  # again, change this if model is a state/ passed in
    if model == "linear":
        color = resids**2  # the color of the points will be the residual squared
        scatter_trace = plotly.graph_objects.Scatter(x= x[0][0:-2], y= y[0:-2], mode= 'markers', marker= dict(color= color,
                                                        colorscale= "portland", colorbar= dict(title= "Residual Squared",
                                                        x= 1.1, y= .5, len= .5)), name= 'Data Points')

    elif model == "logistic":
        color = resids  # the color of the points is just their signed margin

        # want y labels to be conveyed by shapes
        scatter_trace = plotly.graph_objects.Scatter(x= x[0][0:-2], y= x[1][0:-2], mode= 'markers',
                                                        marker= dict(symbol= np.vectorize(symbol_map.get)(y), color= color,
                                                        colorscale= "portland", colorbar= dict(title= "Margins", x= 1.1,
                                                        y= .5, len= .5)), name= "Data Points", showlegend= False)


    # both types of models need the original line, a current line, and handlebars
    global og_fit_line
    og_fit_trace = plotly.graph_objects.Scatter(x= x[0][0:-2], y= og_fit_line, mode= 'lines', name= 'Original Fit Line')
    curr_fit_trace = plotly.graph_objects.Scatter(x= x[0][0:-2], y= fit_line, mode= 'lines', name= 'Current Fit Line')

    # plotting the handlebars
    points_on_line_trace = plotly.graph_objects.Scatter(x= x_handlebars, y=y_handlebars, mode= 'markers',
                                                            name= 'Handlebars', marker= dict(color= 'red', size= 10))

    # making the figure with all of the scatter traces
    fig = plotly.graph_objects.Figure(data= [scatter_trace, curr_fit_trace, og_fit_trace, points_on_line_trace])
    fig.update_layout(title= "Data")

    if model == "logistic":  # add labels for the different symbols
        for label in symbol_map:
            fig.add_trace(go.Scatter(
                x= [None], y= [None],  # No data points, just for the legend
                mode= 'markers',
                marker= dict(symbol= symbol_map[label], color= 'red'),
                name= f"Data with Y = {label}",
            ))

    return json.loads(pio.to_json(fig))


# plots the intercepts and slopes of the model/ lines attempted so far, color is determined by the model's log likelihood
def make_prog_chart(coef, likelihood):
    # make the current/ most recent point bigger
    size_list = np.ones(len(likelihood)) * 10
    size_list[-1] = 25

    # plot the attempts
    temp = [go.Scatter(x=coef[1], y=coef[0], mode="lines+markers", marker=dict(size= size_list, color= likelihood,
                            colorscale= "Viridis", colorbar= dict(title= "Approx Log Likelihood", x=1.1, y=.5, len=.5)),
                            name= "Tries")]
    layout = go.Layout(title= "Coefficients Tried (Current Try Larger)", xaxis_title= "Intercept", yaxis_title= "Slope")
    fig = go.Figure(data= temp, layout= layout)

    return json.loads(pio.to_json(fig))


# calculates the log likelihood of the current model using the residuals or margins
def log_likelihood(resids):
    global model
    if model == "linear":
        resid_squared = resids**2
        rss = resid_squared.sum()
        resid_var = np.var(resids, ddof=2)
        n = len(resids)
        return -n / 2 * np.log(2 * math.pi * resid_var) - rss / (2 * resid_var)
    elif model == "logistic":
        return np.sum(-np.log(1 + np.exp(-resids)))


# this component facilitates the plotting, updating, and replotting of data. Events such as key presses, sliders,
# and tones played are also handled here
@component
def InteractiveGraph():
    global data

    # declare states (akin to global variables but for the UI specifically)
    points, set_points = hooks.use_state(data)  # data
    pitch, set_pitch = hooks.use_state(440)  # tone

    _, resid, coef = line_fit(data)  # need these to populate the initial tries and likelihoods

    # try_list = [[coef[0]], [coef[1]]]
    tries, set_tries = hooks.use_state([[coef[0]], [coef[1]]])  # the slope and intercepts tried so far
    # likelihood_list = [log_likelihood(resid)]
    likelihood, set_likelihood = hooks.use_state([log_likelihood(resid)])  # log likelihood of model tried so far

    # adjust stepsize
    step, set_step = hooks.use_state(.25)

    # create the plots
    graph_json = make_plot(points)
    graph_temp = make_prog_chart(tries, likelihood)

    # this script is the html to actually display the graphs and prepare for key presses
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


    # takes in the points, the index  (from the end) of the handlebar being updated, and the target handlebar's change
    # in x and y (or x1 and x2, in the case of logistic regression)
    # this could probably be rewritten to not need all points passed in, but this way has worked so far
    def update_point(all_points, index, dx, dy):

        # this update is fine for 1d linear or 2d logistic!
        new_points = all_points[:]
        new_points[0][-index] += dx
        new_points[1][-index] += dy

        # calculate the residuals/ margins to determine the tone
        _, resids, _ = line_fit(new_points)

        global model
        if model == "linear":
            temp = resids
            maxErr = 3500  # for these synthetic examples, residuals are often much higher than the bad margins...
        elif model == "logistic":
            temp = resids[resids < 0]  # bad margins
            maxErr = 150  # what is a reasonable error for this?

        temp = temp ** 2
        temp = temp.sum()  # this gets the rss or the sum of squared bad margins

        # if the range of possible error statistics is too large, then the change in tones may not be noticeable...
        if temp > maxErr:
            temp = maxErr
        minHz = 300
        maxHz = 1200
        new_pitch =  minHz + (temp / maxErr) * (maxHz - minHz)  # scales the pitch to be between 300 and 1200 Hz

        return new_points, new_pitch, coef, resids


    # Looks for arrow key or wasd presses and calls update_point() accordingly
    def handle_key_down(event):
        nonlocal pitch

        pressed = False
        index = 0  # will either have a value of 1 (left handlebar) or 2 (right handlebar)
        dx = 0
        dy = 0
        delta = float(step)  # makes it so the amount changed (dx or dy) is determined by the step size, which is
        # determined by the slider

        # for each of the valid key presses, sets index and change in x or y (never both) to intended value
        # also sets pressed to True so that and update and redraw only happens when a handlebar is moved
        if event["key"] == "ArrowUp":
            index = 2
            dy = delta
            pressed = True
        elif event["key"] == "ArrowDown":
            index = 2
            dy = -delta
            pressed = True
        elif event["key"] == "ArrowLeft":
            index = 2
            dx = -delta
            pressed = True
        elif event["key"] == "ArrowRight":
            index = 2
            dx = delta
            pressed = True
        elif event["key"] == "w":
            index = 1
            dy = delta
            pressed = True
        elif event["key"] == "a":
            index = 1
            dx = -delta
            pressed = True
        elif event["key"] == "s":
            index = 1
            dy = -delta
            pressed = True
        elif event["key"] == "d":
            index = 1
            dx = delta
            pressed = True

        if pressed:
            # updates points, the tries, and the pitch
            new_points, new_pitch, coef, resids  = update_point(points, index, dx, dy)

            temp_tries = tries[:]
            temp_tries[0].append(coef[0])
            temp_tries[1].append(coef[1])

            set_pitch(new_pitch)
            set_likelihood(likelihood + [log_likelihood(resids)])  # again, only need the labels for logistic, unused for linear


    # plays a short tone at the given frequency (which will be some function of residuals/ margins)
    def play_tone(err):
        return html.script(
            f"""
                (function() {{
                    const audio = new AudioContext();
                    const oscillator = audio.createOscillator();
                    oscillator.frequency.value = {err};
                    oscillator.connect(audio.destination);
                    oscillator.start();
                    setTimeout(() => oscillator.stop(), 200);
                }})();
                """)


    # updates the step size according to changes with the slider
    def handle_slider_change(event):
        set_step(event["target"]["value"])

    # the component returns the html necessary to display the graphs, handle key presses and sliders, and play tone
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


# returns random (x, y) data for linear regression and random (x1, x2, y) data for logistic regression
def generate_data():
    global model

    if model == "linear":
        # make data
        x = np.random.uniform(0, 10, 30)
        m = np.random.uniform(-3, 3)
        y = m * x + np.random.uniform(-3, 3, 30)

        # fit line
        coef =  np.polyfit(x, y, 1)
        slope = coef[0]
        intercept = coef[1]

        # put handlebars on said line and append to x and y
        top = np.percentile(x, 80)  # putting the handlebars on the 80th and 20th percentile x values
        bottom = np.percentile(x, 20)
        x = [np.append(x, [top, bottom])]
        y = np.append(y, [slope * top + intercept, slope * bottom + intercept])

    elif model == "logistic":
        locations = [np.random.uniform(-3, -1), np.random.uniform(1,3)]  # where will the clusters be?
        # right now, the clusters will have similar x1 and x2 values, but that can be changed by having different
        # random numbers for x1 center and x2 center

        # creates clusters and assigns labels (for now, only small % chance of having a different label than your
        # cluster mates... this can easily be edited to make the task easier or harder)
        rng = np.random.default_rng()
        cluster1 = np.random.normal(loc=[locations[0], locations[0]], size=(15,2))
        label1 = rng.choice(a= np.array([-1, 1]), size=15, p= [.9, .1])

        cluster2 = np.random.normal(loc=[locations[1], locations[1]], size=(15,2))
        label2 = rng.choice(a= [-1, 1], size=15, p= [.2, .8])

        # put the clusters together
        x = np.vstack((cluster1, cluster2))
        y = np.hstack((label1, label2)).ravel()

        # fit a line and add handlebars
        mod = LogisticRegression()
        mod.fit(x, y)
        coef = mod.coef_[0]
        intercept = mod.intercept_[0]

        # but now to make the line into x2 = m*x1 + b
        # if coef[1] is really small, the equation becomes unstable, so let's handle the two cases
        if abs(coef[1]) > abs(coef[0]):
            top_x1 = np.percentile(x[:, 0], 80)  # again doing 80th and 20th percentile of x(1)
            bottom_x1 = np.percentile(x[:, 0], 20)
            right_x2 = -(coef[0] * top_x1 + intercept) / coef[1]
            left_x2 = -(coef[0] * bottom_x1 + intercept) / coef[1]
            x = np.vstack((x, np.array([top_x1, right_x2])))
            x = np.vstack((x, np.array([bottom_x1, left_x2])))
        else:
            # so now we need to find the line in terms of x2 instead of x1
            top_x2 = np.percentile(x[:, 1], 80)
            bottom_x2 = np.percentile(x[:, 1], 20)
            top_x1 = -(coef[1] * top_x2 + intercept) / coef[0]
            bottom_x1 = -(coef[1] * bottom_x2 + intercept) / coef[0]

            # I need to add the point with larger x1 first to be consistent with rest of code
            if top_x1 > bottom_x1:
                x = np.vstack((x, np.array([top_x1, top_x2])))
                x = np.vstack((x, np.array([bottom_x1, bottom_x2])))
            else:
                x = np.vstack((x, np.array([bottom_x1, bottom_x2])))
                x = np.vstack((x, np.array([top_x1, top_x2])))

        x = [x[:,0], x[:,1]]  # this is done so that the ending dataset with be [x1, x2, y]

    return x, y


configure(app, InteractiveGraph)

# runs the program
def main():
    global model
    model = "logistic"  # logistic or linear, eventually make this a button, make this easier to change

    x, y = generate_data()

    global data
    # functions assume the data comes in the form of [x1, x2, ..., y]
    data = x
    data.append(y)

    # getting the starting line so we can always plot it as a baseline
    line, _, _ = line_fit(data)
    global og_fit_line
    og_fit_line = line

    uvicorn.run(app, host="127.0.0.1", port=8000)  # runs the component

main()


