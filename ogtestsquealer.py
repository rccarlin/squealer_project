import reactpy
import plotly
import plotly.io as pio
import numpy as np
from fastapi import FastAPI
from reactpy import component, html, hooks
from reactpy.backend.fastapi import configure
import uvicorn
import plotly.graph_objs as go
import json
import math
from sklearn.linear_model import LogisticRegression
import random

# Global variables for storing the current dataset, model mode, and the original best-fit line.
global data
global model
global og_fit_line

app = FastAPI()

# takes in the data (including handle bars), finds the line/ coefficients defined by the handle bars
# also returns the "residuals" which are residuals for linear regression but are signed margins for logistic regression
def line_fit(points, is_linear):
    
    x_all = points[0]
    y_all = points[1]

    x_main = x_all[:-2]
    y_main = y_all[:-2]

    right = (x_all[-1], y_all[-1])
    left  = (x_all[-2], y_all[-2])

    slope = (right[1] - left[1]) / (right[0] - left[0])
    intercept = right[1] - slope * right[0]
    coef = [slope, intercept]


    if is_linear:
        # Compute the fitted values and residuals on the main data.
        fit_line = coef[0] * np.array(x_all) + coef[1]
        resids = y_all - fit_line
    
    else:
        
        # Logistic mode: points = [x1, x2, y]
        x_2 = points[2]
        fit_line = slope * x_main + intercept

        fit_line = coef[0] * x_all + coef[1]
        resids = x_2 * (y_main - coef[0] * x_main - coef[1]) / (coef[0] ** 2 + 1) ** .5  # signed margins
        
    return fit_line, resids, coef

# takes in all data (with handle bars at the end) to plot the data points, handle bars, original best fit line, and the
# current line defined by the handlebars
def make_plot(data, is_linear):

    # linear mode:
    if is_linear:


        x_all = data[0]
        y_all = data[1]
        x_main = x_all[:-2]
        y_main = y_all[:-2]
        fit_line, resids, coef = line_fit(data, is_linear)

        # convert to lists for plotly compatibility
        x_main_list = x_main.tolist()
        y_main_list = y_main.tolist()
        fit_line_list = fit_line.tolist()
        color_list = (resids**2).tolist()
        scatter_trace = go.Scatter(
            x=x_main_list,
            y=y_main_list,
            mode='markers',
            marker=dict(
                color=color_list,
                colorscale="portland",
                colorbar=dict(title="Residual Squared", x=1.1, y=0.5, len=0.5)
            ),
            name="Data Points"
        )
        curr_fit_trace = go.Scatter(
            x=x_main_list,
            y=fit_line_list,
            mode='lines',
            name='Current Fit Line'
        )
        # the two handlebar points.
        handlebars_trace = go.Scatter(
            x=[x_all[-1], x_all[-2]],
            y=[y_all[-1], y_all[-2]],
            mode='markers',
            name='Handlebars',
            marker=dict(color='red', size=10)
        )
        # plot the original best-fit line based on handlebars (fixed).
        og_fit_trace = go.Scatter(
            x=x_main_list,
            y=og_fit_line.tolist() if isinstance(og_fit_line, np.ndarray) else og_fit_line,
            mode='lines',
            name='Original Fit Line'
        )

        fig = go.Figure(data=[scatter_trace, curr_fit_trace, og_fit_trace, handlebars_trace])
        fig.update_layout(title="Data")
        return json.loads(fig.to_json())
    
    # logistic 
    else:
        # FIX THIS!!! Logistic mode ###########
        x1_all = data[0]
        x2_all = data[1]
        y_labels = data[2]
        x1_main = x1_all[:-2]
        x2_main = x2_all[:-2]
        fit_line, resids, coef = line_fit(data, is_linear)

        x1_main_list = x1_main.tolist()
        x2_main_list = x2_main.tolist()
        fit_line_list = fit_line.tolist()
        color_list = resids.tolist()
        symbol_map = {-1: "circle", 1: "cross"}

        scatter_trace = go.Scatter(
            x=x1_main_list,
            y=x2_main_list,
            mode='markers',
            marker=dict(
                symbol=[symbol_map[int(lbl)] for lbl in y_labels[:-2]],
                color=color_list,
                colorscale="portland",
                colorbar=dict(title="Margins", x=1.1, y=0.5, len=0.5)
            ),
            name="Data Points",
            showlegend=True
        )

        # plot the original best-fit line based on handlebars (fixed)
        # og_fit_trace = go.Scatter(
        #     x=x1_main_list,
        #     y=og_fit_line.tolist() if isinstance(og_fit_line, np.ndarray) else og_fit_line,
        #     mode='lines',
        #     name='Original Fit Line'
        # )

        fit_line_trace = go.Scatter(
            x=x1_main_list,
            y=fit_line_list,
            mode='lines',
            name='Current Fit Line'
        )
        handlebars_trace = go.Scatter(
            x=[x1_all[-1], x1_all[-2]],
            y=[x2_all[-1], x2_all[-2]],
            mode='markers',
            name='Handlebars',
            marker=dict(color='red', size=10)
        )
        fig = go.Figure(data=[scatter_trace, fit_line_trace, handlebars_trace])
        for lbl, sym in symbol_map.items():
            dummy_trace = go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(symbol=sym, color='red'),
                name=f"Data with Y = {lbl}"
            )
            fig.add_trace(dummy_trace)
        fig.update_layout(title="Data")
        return json.loads(fig.to_json())

# plots the intercepts and slopes of the model/ lines attempted so far, color is determined by the model's log likelihood
def make_prog_chart(coef_history, likelihood_history):
    size_list = [10] * len(likelihood_history)
    size_list[-1] = 25
    trace = go.Scatter(
        x=coef_history[1],
        y=coef_history[0],
        mode="lines+markers",
        marker=dict(
            size=size_list,
            color=likelihood_history,
            colorscale="Viridis",
            colorbar=dict(title="Approx Log Likelihood", x=1.1, y=0.5, len=0.5)
        ),
        name="Tries"
    )
    layout = go.Layout(
        title="Coefficients Tried (Current Try Larger)",
        xaxis_title="Intercept",
        yaxis_title="Slope"
    )
    fig = go.Figure(data=[trace], layout=layout)
    return json.loads(fig.to_json())

# calculates the log likelihood of the current model using the residuals or margins
def log_likelihood(resids, is_linear):
    if is_linear:
        resid_squared = resids**2
        rss = resid_squared.sum()
        resid_var = np.var(resids, ddof=2)
        n = len(resids)
        return -n/2 * np.log(2*math.pi*resid_var) - rss/(2*resid_var)
    else:
        return np.sum(-np.log(1+np.exp(-resids)))



# this component facilitates the plotting, updating, and replotting of data. Events such as key presses, sliders,
# and tones played are also handled here
@component
def InteractiveGraph():
    global data
    global model

    # declare states (akin to global variables but for the UI specifically)
    points, set_points = hooks.use_state(data)           # data
    pitch, set_pitch = hooks.use_state(440)              # tone
    is_linear, set_is_linear = hooks.use_state(model)    # model type (true=linear, false=logistic)

    # compute  current best fit line from position of handlebars
    fit_line, resids, coef = line_fit(points, is_linear)
    tries, set_tries = hooks.use_state([[coef[0]], [coef[1]]])
    likelihood_history, set_likelihood = hooks.use_state([log_likelihood(resids, is_linear)])
    step, set_step = hooks.use_state(0.25)

    # graphs
    graph_json = make_plot(points, is_linear)
    prog_chart_json = make_prog_chart(tries, likelihood_history)

    # this script is the html to actually display the graphs and prepare for key presses
    script = html.script(f"""
        function renderPlot() {{
            if (typeof Plotly !== "undefined") {{
                Plotly.newPlot('plot1', {json.dumps(graph_json['data'])}, {json.dumps(graph_json['layout'])});
                Plotly.newPlot('plot2', {json.dumps(prog_chart_json['data'])}, {json.dumps(prog_chart_json['layout'])});
            }} else {{
                console.error("Plotly not loaded");
            }}
        }}
        var plotlyScript = document.createElement('script');
        plotlyScript.src = 'https://cdn.plot.ly/plotly-latest.min.js';
        plotlyScript.onload = renderPlot;
        document.head.appendChild(plotlyScript);
    """)

    # takes in the points, the index  (from the end) of the handlebar being updated, and the target handlebar's change
    # in x and y (or x1 and x2, in the case of logistic regression)
    # this could probably be rewritten to not need all points passed in, but this way has worked so far
    def update_point(all_points, index, dx, dy):
        
        # this update is fine for 1d linear or 2d logistic!        
        new_points = [np.copy(arr) for arr in all_points]
        new_points[0][-index] += dx
        new_points[1][-index] += dy
        fit_line_new, resids_new, coef_new = line_fit(new_points, is_linear)
        
        if is_linear:
            temp = resids_new
            maxErr = 3500 # for these synthetic examples, residuals are often much higher than the bad margins...
        else:
            temp = resids_new[resids_new < 0] if np.any(resids_new < 0) else resids_new # bad margins
            maxErr = 100 # what is a reasonable error for this?
        
        err_val = np.sum(temp**2)
        if err_val > maxErr:
            err_val = maxErr
        minHz = 300
        maxHz = 1200
        new_pitch = minHz + (err_val / maxErr) * (maxHz - minHz) # scales the pitch to be between 300 and 1200 Hz
        return new_points, new_pitch, coef_new, resids_new


    # Looks for arrow key or wasd presses and calls update_point() accordingly
    def handle_key_down(event):
        nonlocal pitch

        pressed = False
        index = 0 # will either have a value of 1 (left handlebar) or 2 (right handlebar)
        dx = 0
        dy = 0
        delta = float(step) # makes it so the amount changed (dx or dy) is determined by the step size, which is
        key = event["key"]
        
        # for each of the valid key presses, sets index and change in x or y (never both) to intended value
        # also sets pressed to True so that and update and redraw only happens when a handlebar is moved
        if key == "ArrowUp":
            index = 2
            dy = delta
            pressed = True
        elif key == "ArrowDown":
            index = 2
            dy = -delta
            pressed = True
        elif key == "ArrowLeft":
            index = 2
            dx = -delta
            pressed = True
        elif key == "ArrowRight":
            index = 2
            dx = delta
            pressed = True
        elif key == "w":
            index = 1
            dy = delta
            pressed = True
        elif key == "s":
            index = 1
            dy = -delta
            pressed = True
        elif key == "a":
            index = 1
            dx = -delta
            pressed = True
        elif key == "d":
            index = 1
            dx = delta
            pressed = True


        if pressed:
            # updates points, the tries, and the pitch
            new_points, new_pitch, new_coef, new_resids = update_point(points, index, dx, dy)
            set_points(new_points)
            new_tries = [tries[0] + [new_coef[0]], tries[1] + [new_coef[1]]]
            set_tries(new_tries)
            new_likelihood = likelihood_history + [log_likelihood(new_resids, is_linear)] # again, only need the labels for logistic, unused for linear
            set_likelihood(new_likelihood)
            set_pitch(new_pitch)


# updates the step size according to changes with the slider
    def handle_slider_change(event):
        set_step(event["target"]["value"])


    # this function resets the data, tone, model attempts, etc. as it switches to/ from a linear/ logistic model
    def switch_modes(event=None):
        set_is_linear(lambda prev: not prev)
        temp_mode = not is_linear # state doesn't get updated right away, so for the rest of the function let's use this
        
        # now that I've switched models, I need new data...
        new_data = generate_data(temp_mode)
        global data
        data = new_data
        set_points(new_data)

        # getting the starting line so we can always plot it as a baseline        
        fit_line_new, resids_new, coef_new = line_fit(new_data, temp_mode)
        global og_fit_line

        # reset all of the trackers        
        set_pitch(440)
        set_likelihood([log_likelihood(resids_new, temp_mode)])
        set_tries([[coef_new[0]], [coef_new[1]]])


    # plays a short tone at the given frequency (which will be some function of residuals/ margins)
    def play_tone(err):
        return html.script(f"""
            (function() {{
                const audio = new AudioContext();
                const oscillator = audio.createOscillator();
                oscillator.frequency.value = {err};
                oscillator.connect(audio.destination);
                oscillator.start();
                setTimeout(() => oscillator.stop(), 200);
            }})();
        """)
    
    # the component returns the html necessary to display the graphs, handle key presses and sliders, and play tone
    return html.div([
        html.div({"id": "plot1", "style": {"width": "800px", "height": "600px"}}),
        script,
        html.div({
            "tabIndex": 0,
            "autofocus": True,
            "onKeyDown": handle_key_down,
            "style": {"border": "1px solid black", "padding": "10px", "width": "300px"}
        }, "Use ARROW KEYS (right handlebar) and WASD (left handlebar) to move the handlebars independently."),
        html.div({"style": {"display": "flex", "alignItems": "center", "gap": "10px"}},
            [
                "Slider: ",
                html.input({
                    "type": "range",
                    "min": 0,
                    "max": 10,
                    "value": step,
                    "step": 0.25,
                    "onInput": handle_slider_change,
                }),
                html.span(f"Value: {step}")
            ]
        ),
        html.div({"id": "plot2", "style": {"width": "600px", "height": "400px"}}),
        html.button({
            "onClick": switch_modes,
            "style": {"fontSize": "20px"}
        }, "Linear" if not is_linear else "Logistic"),
        play_tone(pitch)
    ])



# returns random (x, y) data for linear regression and random (x1, x2, y) data for logistic regression
def generate_data(is_linear):
    if is_linear:
        x = np.random.uniform(0, 10, 30)
        m = np.random.uniform(-3, 3)
        y = m * x + np.random.uniform(-3, 3, 30)
        # Compute the original best-fit line from the main data.
        coef = np.polyfit(x, y, 1)
        slope = coef[0]
        intercept = coef[1]
        top = np.percentile(x, 80)
        bottom = np.percentile(x, 20)
        # Append two handlebar points computed from the original line.
        x_full = np.append(x, [top, bottom])
        y_full = np.append(y, [slope * top + intercept, slope * bottom + intercept])
        return [x_full, y_full]
    else:
        # (Logistic mode data generation – not modified here.)
        locations = [np.random.uniform(-3, -1), np.random.uniform(1, 3)]
        rng = np.random.default_rng()
        cluster1 = np.random.normal(loc=[locations[0], locations[0]], size=(15,2))
        label1 = rng.choice(a=[-1, 1], size=15, p=[0.9, 0.1])
        cluster2 = np.random.normal(loc=[locations[1], locations[1]], size=(15,2))
        label2 = rng.choice(a=[-1, 1], size=15, p=[0.2, 0.8])
        x = np.vstack((cluster1, cluster2))
        y = np.hstack((label1, label2)).ravel()
        mod = LogisticRegression()
        mod.fit(x, y)
        coef = mod.coef_[0]
        intercept = mod.intercept_[0]
        if abs(coef[1]) > abs(coef[0]):
            top_x1 = np.percentile(x[:,0], 80)
            bottom_x1 = np.percentile(x[:,0], 20)
            right_x2 = -(coef[0]*top_x1+intercept)/coef[1]
            left_x2 = -(coef[0]*bottom_x1+intercept)/coef[1]
            x = np.vstack((x, np.array([top_x1, right_x2])))
            x = np.vstack((x, np.array([bottom_x1, left_x2])))
        else:
            top_x2 = np.percentile(x[:,1], 80)
            bottom_x2 = np.percentile(x[:,1], 20)
            top_x1 = -(coef[1]*top_x2+intercept)/coef[0]
            bottom_x1 = -(coef[1]*bottom_x2+intercept)/coef[0]
            if top_x1 > bottom_x1:
                x = np.vstack((x, np.array([top_x1, top_x2])))
                x = np.vstack((x, np.array([bottom_x1, bottom_x2])))
            else:
                x = np.vstack((x, np.array([bottom_x1, bottom_x2])))
                x = np.vstack((x, np.array([top_x1, top_x2])))
        x_list = [x[:,0], x[:,1]]
        return [x_list[0], x_list[1], y]


configure(app, InteractiveGraph)

# runs the program
def main():
    global model, data, og_fit_line
    model = True  # Start in linear mode.
    data = generate_data(model)
    # Compute and store the original best-fit line from the main data.
    x_arr = data[0][:-2]
    y_arr = data[1][:-2]
    og_fit_line = np.polyval(np.polyfit(x_arr, y_arr, 1), x_arr)
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
